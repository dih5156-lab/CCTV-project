"""AI 최초 접수와 감독자 활성화가 공유하는 불변 실행 정보·sequence 경계."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
import uuid
from pathlib import Path

from src.services.edge_orchestrator import RegistryError, _fields, _identifier, _utc
from src.services.edge_ownership import _encode, _epoch, _uuid, validate_result

_APPLICATION_ID = 1162102613


class EdgeResultContext:
    """활성화와 표본 번호 할당을 직렬화하며 이미 붙은 메타데이터를 갱신하지 않는다."""

    def __init__(self, path, *, create=False, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.lock = threading.RLock()
        parent = self.path.parent.lstat()
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid()
            or parent.st_mode & 0o077
        ):
            raise ValueError("result context requires an owner-only directory")
        new = False
        if create:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
                new = True
            except FileExistsError:
                pass
        info = self.path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("result context must be an owner-only regular file")
        self.db = sqlite3.connect(
            self.path.resolve().as_uri() + "?mode=rw",
            uri=True,
            check_same_thread=False,
            timeout=2,
        )
        self.db.row_factory = sqlite3.Row
        try:
            if new:
                self.db.executescript(f"""
                    PRAGMA application_id={_APPLICATION_ID};
                    CREATE TABLE sessions (
                        session_id TEXT PRIMARY KEY, identity TEXT NOT NULL,
                        next_sequence INTEGER NOT NULL DEFAULT 0,
                        activation_id TEXT, sequence_floor INTEGER
                    );
                    CREATE TABLE current (id INTEGER PRIMARY KEY CHECK(id=1), session_id TEXT);
                    CREATE TABLE commands (command_id TEXT PRIMARY KEY, body TEXT NOT NULL, response TEXT NOT NULL);
                """)
            elif (
                self.db.execute("PRAGMA application_id").fetchone()[0]
                != _APPLICATION_ID
            ):
                raise ValueError("not an edge result context database")
            if create:
                # 운영 DB와 분리된 이 DB에서만 접수 표본과 중앙 저장 ACK를 보관한다.
                self.db.executescript("""
                    CREATE TABLE IF NOT EXISTS captures (
                        event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                        event_sequence INTEGER NOT NULL, edge TEXT NOT NULL,
                        UNIQUE(session_id,event_sequence)
                    );
                    CREATE TABLE IF NOT EXISTS node_outbox (
                        event_id TEXT PRIMARY KEY, body TEXT NOT NULL, sha256 TEXT NOT NULL,
                        state TEXT NOT NULL DEFAULT 'pending', receipt TEXT,
                        last_error TEXT, last_attempt REAL NOT NULL DEFAULT 0
                    );
                """)
                if "last_attempt" not in {
                    row[1] for row in self.db.execute("PRAGMA table_info(node_outbox)")
                }:
                    self.db.execute(
                        "ALTER TABLE node_outbox ADD COLUMN last_attempt REAL NOT NULL DEFAULT 0"
                    )
        except BaseException:
            self.db.close()
            raise

    def begin(self, identity):
        _fields(
            identity,
            {
                "node_id",
                "camera_id",
                "assignment_id",
                "epoch",
                "supervisor_boot_id",
                "session_id",
            },
        )
        for key in ("node_id", "camera_id"):
            _identifier(identity[key])
        for key in ("assignment_id", "supervisor_boot_id", "session_id"):
            _uuid(identity[key])
        _epoch(identity["epoch"])
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute(
                "SELECT session_id FROM current WHERE id=1"
            ).fetchone()
            if old and old[0] is not None:
                raise RegistryError(409, "result_session_already_running")
            self.db.execute(
                "INSERT INTO sessions(session_id,identity) VALUES (?,?)",
                (identity["session_id"], _encode(identity)),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO current VALUES (1,?)", (identity["session_id"],)
            )

    def _session(self, session_id):
        row = self.db.execute(
            """SELECT sessions.* FROM sessions JOIN current
            ON current.session_id=sessions.session_id WHERE current.id=1 AND sessions.session_id=?""",
            (session_id,),
        ).fetchone()
        if row is None:
            raise RegistryError(409, "retired_result_session")
        return row

    def end(self, session_id=None):
        with self.lock, self.db:
            if session_id is None:
                # 감독자가 배타 소유 락을 얻은 직후에는 살아 있는 이전 AI가 없음을 전제로 정리한다.
                self.db.execute("UPDATE current SET session_id=NULL WHERE id=1")
            else:
                self.db.execute(
                    "UPDATE current SET session_id=NULL WHERE id=1 AND session_id=?",
                    (session_id,),
                )

    def capture(self, session_id, *, lifetime=30, event_id=None):
        if type(lifetime) not in (int, float) or not 0 < lifetime <= 300:
            raise ValueError("result lifetime must be within (0,300]")
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self._session(session_id)
            event_id = (
                _identifier(event_id) if event_id is not None else str(uuid.uuid4())
            )
            existing = self.db.execute(
                "SELECT session_id,edge FROM captures WHERE event_id=?", (event_id,)
            ).fetchone()
            if existing:
                if existing["session_id"] != session_id:
                    raise RegistryError(409, "capture_identity_conflict")
                return {"event_id": event_id, "edge": json.loads(existing["edge"])}
            sequence = _epoch(row["next_sequence"] + 1)
            self.db.execute(
                "UPDATE sessions SET next_sequence=? WHERE session_id=?",
                (sequence, session_id),
            )
            edge = json.loads(row["identity"])
            now = self.clock()
            edge.update(
                event_sequence=sequence,
                activation_id=row["activation_id"],
                captured_at=_utc(now),
                expires_at=_utc(now + lifetime),
            )
            self.db.execute(
                "INSERT INTO captures VALUES (?,?,?,?)",
                (event_id, session_id, sequence, _encode(edge)),
            )
            return {"event_id": event_id, "edge": edge}

    def check_capture(self, stamp):
        _fields(stamp, {"event_id", "edge"})
        with self.lock:
            row = self.db.execute(
                "SELECT edge FROM captures WHERE event_id=?",
                (_identifier(stamp["event_id"]),),
            ).fetchone()
            if row is None or row["edge"] != _encode(stamp["edge"]):
                raise RegistryError(409, "capture_identity_conflict")

    def enqueue(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("edge"), dict):
            raise RegistryError(400, "invalid_result")
        edge, event_id, _, _, body, digest = validate_result(
            payload.get("edge", {}).get("node_id"), payload, self.clock()
        )
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.check_capture({"event_id": event_id, "edge": edge})
            old = self.db.execute(
                "SELECT sha256 FROM node_outbox WHERE event_id=?", (event_id,)
            ).fetchone()
            if old and old["sha256"] != digest:
                raise RegistryError(409, "result_identity_conflict")
            # 새 결과의 0 우선순위 때문에 과거 재시도가 계속 밀리지 않도록 한다.
            self.db.execute(
                "INSERT OR IGNORE INTO node_outbox(event_id,body,sha256,last_attempt) VALUES (?,?,?,?)",
                (event_id, body, digest, self.clock()),
            )
        return {"event_id": event_id, "sha256": digest}

    def pending(self, limit=20):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("pending limit must be within [1,100]")
        with self.lock:
            return [
                {
                    "event_id": row["event_id"],
                    "sha256": row["sha256"],
                    "payload": json.loads(row["body"]),
                }
                for row in self.db.execute(
                    "SELECT * FROM node_outbox WHERE state='pending' ORDER BY last_attempt,rowid LIMIT ?",
                    (limit,),
                )
            ]

    def confirm(self, receipt):
        if not isinstance(receipt, dict) or receipt.get("stored") is not True:
            raise RegistryError(400, "central_storage_receipt_required")
        event_id = _identifier(receipt.get("event_id"))
        encoded = _encode(receipt)
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT sha256 FROM node_outbox WHERE event_id=?", (event_id,)
            ).fetchone()
            if row is None or row["sha256"] != receipt.get("sha256"):
                raise RegistryError(409, "central_receipt_mismatch")
            self.db.execute(
                "UPDATE node_outbox SET state='committed',receipt=?,last_error=NULL WHERE event_id=?",
                (encoded, event_id),
            )

    def record_error(self, event_id, error_name):
        # 오류 본문에는 주소·비밀번호가 포함될 수 있으므로 분류명만 저장한다.
        _identifier(error_name)
        with self.lock, self.db:
            self.db.execute(
                "UPDATE node_outbox SET last_error=?,last_attempt=? WHERE event_id=? AND state='pending'",
                (error_name, self.clock(), event_id),
            )

    def activate(self, session_id, payload):
        _fields(payload, {"command_id", "expected_activation_id", "activation_id"})
        _identifier(payload["command_id"])
        _uuid(payload["activation_id"])
        if payload["expected_activation_id"] is not None:
            _uuid(payload["expected_activation_id"])
        encoded = _encode({"session_id": session_id, **payload})
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self._session(session_id)
            old = self.db.execute(
                "SELECT body,response FROM commands WHERE command_id=?",
                (payload["command_id"],),
            ).fetchone()
            if old:
                if old["body"] != encoded:
                    raise RegistryError(409, "activation_command_conflict")
                return json.loads(old["response"])
            if row["activation_id"] != payload["expected_activation_id"]:
                raise RegistryError(409, "activation_changed")
            if row["activation_id"] == payload["activation_id"]:
                raise RegistryError(409, "activation_token_reused")
            response = {
                "activation_id": payload["activation_id"],
                "sequence_floor": row["next_sequence"],
            }
            self.db.execute(
                "UPDATE sessions SET activation_id=?,sequence_floor=? WHERE session_id=?",
                (response["activation_id"], response["sequence_floor"], session_id),
            )
            self.db.execute(
                "INSERT INTO commands VALUES (?,?,?)",
                (payload["command_id"], encoded, _encode(response)),
            )
            return response

    def snapshot(self, session_id):
        with self.lock:
            row = self._session(session_id)
            return (
                None
                if row["activation_id"] is None
                else {
                    "activation_id": row["activation_id"],
                    "sequence_floor": row["sequence_floor"],
                }
            )

    def preview_identity(self, session_id):
        """현재 활성 실행의 읽기 전용 식별 정보이며 결과 sequence를 소비하지 않는다."""
        with self.lock:
            row = self._session(session_id)
            if row["activation_id"] is None:
                raise RegistryError(409, "preview_not_activated")
            return {**json.loads(row["identity"]), "activation_id": row["activation_id"]}

    def close(self):
        with self.lock:
            self.db.close()
