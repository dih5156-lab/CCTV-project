"""격리 제어기의 단일 담당, 원본 결과 저장 ACK, 최종 모의 수신 경계."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
import uuid
from pathlib import Path

from src.services.edge_orchestrator import (
    RegistryError,
    _fields,
    _identifier,
    _number,
    _sample_time,
    _utc,
)

_APPLICATION_ID = 1162102607


def _uuid(value):
    try:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise RegistryError(400, "invalid_uuid") from None
    return value


def _epoch(value, *, zero=False):
    if type(value) is not int or not int(not zero) <= value < 2**63:
        raise RegistryError(400, "invalid_epoch")
    return value


def _encode(value):
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, RecursionError):
        raise RegistryError(400, "invalid_json_value") from None
    if len(encoded.encode()) > 49152:
        raise RegistryError(413, "result_too_large")
    return encoded


def validate_result(node_id, payload, now):
    """노드 송신 큐와 중앙 수신에 같은 원본·식별자·유효시간 검증을 적용한다."""
    _identifier(node_id)
    _fields(payload, {"kind", "event", "edge"})
    if payload["kind"] not in ("event", "observation") or not isinstance(
        payload["event"], dict
    ):
        raise RegistryError(400, "invalid_result")
    edge = payload["edge"]
    _fields(
        edge,
        {
            "node_id",
            "camera_id",
            "assignment_id",
            "epoch",
            "supervisor_boot_id",
            "session_id",
            "activation_id",
            "event_sequence",
            "captured_at",
            "expires_at",
        },
    )
    for field in ("node_id", "camera_id"):
        _identifier(edge[field])
    if edge["node_id"] != node_id:
        raise RegistryError(403, "node_identity_mismatch")
    for field in ("assignment_id", "supervisor_boot_id", "session_id"):
        _uuid(edge[field])
    if edge["activation_id"] is not None:
        _uuid(edge["activation_id"])
    _epoch(edge["epoch"])
    _epoch(edge["event_sequence"])
    event = payload["event"]
    event_id = _identifier(event.get("event_id"))
    if event.get("camera_id") != edge["camera_id"]:
        raise RegistryError(400, "camera_identity_mismatch")
    if payload["kind"] == "observation":
        _fields(event, {"event_id", "camera_id", "frames_inferred"})
        _number(event["frames_inferred"])
        if event["frames_inferred"] <= 0:
            raise RegistryError(400, "inference_sample_required")
    captured, expires = (
        _sample_time(edge["captured_at"]),
        _sample_time(edge["expires_at"]),
    )
    if captured > now + 5 or not 0 < expires - captured <= 300:
        raise RegistryError(400, "invalid_result_lifetime")
    body = _encode(payload)
    digest = hashlib.sha256(body.encode()).hexdigest()
    return edge, event_id, captured, expires, body, digest


class EdgeOwnershipStore:
    """DB commit 이전 ACK/출력을 금지하고 재기동마다 활성화 경계를 새로 요구한다."""

    def __init__(self, state_dir, registry, *, clock=time.time, stale_after=10):
        _number(stale_after, 300)
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        self.registry = registry
        self.clock = clock
        self.stale_after = stale_after
        self.boot_id = str(uuid.uuid4())
        self.started_at = clock()
        self.lock = threading.RLock()
        self.closed = False
        self.lock_fd = None
        self.db = None
        root = Path(state_dir)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("ownership state directory must be owner-only")
        try:
            self.lock_fd = os.open(
                root / "ownership.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
            )
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = root / "ownership.sqlite3"
            if path.exists():
                with sqlite3.connect(
                    path.resolve().as_uri() + "?mode=ro", uri=True
                ) as connection:
                    if (
                        connection.execute("PRAGMA application_id").fetchone()[0]
                        != _APPLICATION_ID
                    ):
                        raise ValueError("not an edge ownership database")
            self.db = sqlite3.connect(path, check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript(f"""
                PRAGMA application_id={_APPLICATION_ID};
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY, high_epoch INTEGER NOT NULL,
                    active_id TEXT, pending_id TEXT
                );
                CREATE TABLE IF NOT EXISTS assignments (
                    assignment_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL, node_id TEXT NOT NULL,
                    supervisor_boot_id TEXT NOT NULL, manifest_id TEXT NOT NULL,
                    session_id TEXT, activation_id TEXT NOT NULL,
                    sequence_floor INTEGER, state TEXT NOT NULL,
                    expected_id TEXT, expected_epoch INTEGER NOT NULL,
                    verified_run TEXT, prepared_at REAL NOT NULL, activated_at REAL,
                    sample_id TEXT, sample_at REAL,
                    UNIQUE(camera_id, epoch)
                );
                CREATE TABLE IF NOT EXISTS commands (
                    request_id TEXT PRIMARY KEY, body TEXT NOT NULL, response TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS results (
                    event_id TEXT PRIMARY KEY, assignment_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, event_sequence INTEGER NOT NULL,
                    activation_id TEXT, camera_id TEXT NOT NULL, node_id TEXT NOT NULL,
                    kind TEXT NOT NULL, body TEXT NOT NULL, sha256 TEXT NOT NULL,
                    received_at REAL NOT NULL, expires_at REAL NOT NULL,
                    decision TEXT NOT NULL, reason TEXT NOT NULL,
                    delivery_state TEXT NOT NULL, delivery_reason TEXT,
                    consumed_at REAL,
                    UNIQUE(assignment_id, session_id, event_sequence)
                );
            """)
            # 저장된 담당을 새 실행에서 자동 신뢰하지 않는다. 새 토큰으로 노드 경계를 재대조한다.
            with self.db:
                rows = self.db.execute(
                    "SELECT assignment_id FROM assignments WHERE state IN ('active', 'preparing')"
                ).fetchall()
                for row in rows:
                    self.db.execute(
                        "UPDATE assignments SET activation_id=?, verified_run=NULL, sample_id=NULL, sample_at=NULL WHERE assignment_id=?",
                        (str(uuid.uuid4()), row[0]),
                    )
        except BaseException:
            if self.db is not None:
                self.db.close()
            if self.lock_fd is not None:
                os.close(self.lock_fd)
            raise

    def _service(self, node_id, camera_id, *, fresh_run=False):
        for node in self.registry.list_nodes():
            if node["node_id"] != node_id:
                continue
            from src.services.edge_orchestrator import observed_camera_service

            service = observed_camera_service(node, camera_id)
            now = self.clock()
            if not -5 <= now - _sample_time(service["observed_at"]) < self.stale_after:
                raise RegistryError(409, "node_observation_stale")
            if fresh_run and _sample_time(node["received_at"]) < self.started_at:
                raise RegistryError(409, "observation_predates_controller")
            return service
        raise RegistryError(404, "node_not_registered")

    def _command(self, operation, payload, fields):
        _fields(payload, fields | {"request_id", "controller_boot_id"})
        _identifier(payload["request_id"])
        if payload["controller_boot_id"] != self.boot_id:
            raise RegistryError(409, "stale_controller_boot")
        encoded = _encode({"operation": operation, "payload": payload})
        previous = self.db.execute(
            "SELECT body, response FROM commands WHERE request_id=?",
            (payload["request_id"],),
        ).fetchone()
        if previous:
            if previous["body"] != encoded:
                raise RegistryError(409, "request_id_conflict")
            return encoded, json.loads(previous["response"])
        return encoded, None

    def _remember(self, payload, encoded, response):
        self.db.execute(
            "INSERT INTO commands VALUES (?, ?, ?)",
            (payload["request_id"], encoded, _encode(response)),
        )
        return response

    def _assignment(self, assignment_id):
        row = self.db.execute(
            "SELECT * FROM assignments WHERE assignment_id=?", (assignment_id,)
        ).fetchone()
        if not row:
            raise RegistryError(404, "assignment_not_found")
        return dict(row)

    def camera(self, camera_id):
        _identifier(camera_id)
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM cameras WHERE camera_id=?", (camera_id,)
            ).fetchone()
            active = (
                self._assignment(row["active_id"]) if row and row["active_id"] else None
            )
            pending = (
                self._assignment(row["pending_id"])
                if row and row["pending_id"]
                else None
            )
            return {
                "camera_id": camera_id,
                "controller_boot_id": self.boot_id,
                "active": active,
                "pending": pending,
                "accepting_results": bool(
                    active and active["verified_run"] == self.boot_id
                ),
            }

    def prepare(self, camera_id, payload):
        _identifier(camera_id)
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            encoded, duplicate = self._command(
                "prepare:" + camera_id,
                payload,
                {
                    "node_id",
                    "supervisor_boot_id",
                    "manifest_id",
                    "expected_assignment_id",
                    "expected_epoch",
                },
            )
            if duplicate is not None:
                return duplicate
            for field in ("node_id", "manifest_id"):
                _identifier(payload[field])
            _uuid(payload["supervisor_boot_id"])
            if payload["expected_assignment_id"] is not None:
                _uuid(payload["expected_assignment_id"])
            _epoch(payload["expected_epoch"], zero=True)
            current = self.camera(camera_id)
            owner = current["active"]
            if (
                owner["assignment_id"] if owner else None,
                owner["epoch"] if owner else 0,
            ) != (payload["expected_assignment_id"], payload["expected_epoch"]):
                raise RegistryError(409, "owner_changed")
            if current["pending"]:
                raise RegistryError(409, "preparation_in_progress")
            # 다른 카메라의 준비가 끝나기 전 같은 노드에 추가 인계를 겹치지 않는다.
            if self.db.execute(
                "SELECT 1 FROM assignments WHERE node_id=? AND state='preparing' AND camera_id<>?",
                (payload["node_id"], camera_id),
            ).fetchone():
                raise RegistryError(409, "target_preparation_in_progress")
            service = self._service(payload["node_id"], camera_id)
            if (
                service["camera_id"],
                service["manifest_id"],
                service["supervisor_boot_id"],
            ) != (camera_id, payload["manifest_id"], payload["supervisor_boot_id"]):
                raise RegistryError(409, "node_manifest_changed")
            row = self.db.execute(
                "SELECT high_epoch FROM cameras WHERE camera_id=?", (camera_id,)
            ).fetchone()
            epoch = max(row[0] if row else 0, service["epoch"]) + 1
            _epoch(epoch)
            assignment_id, activation_id = str(uuid.uuid4()), str(uuid.uuid4())
            self.db.execute(
                """INSERT INTO assignments (
                assignment_id,camera_id,epoch,node_id,supervisor_boot_id,manifest_id,activation_id,
                state,expected_id,expected_epoch,prepared_at
                ) VALUES (?,?,?,?,?,?,?,'preparing',?,?,?)""",
                (
                    assignment_id,
                    camera_id,
                    epoch,
                    payload["node_id"],
                    payload["supervisor_boot_id"],
                    payload["manifest_id"],
                    activation_id,
                    payload["expected_assignment_id"],
                    payload["expected_epoch"],
                    self.clock(),
                ),
            )
            self.db.execute(
                """INSERT INTO cameras VALUES (?,?,NULL,?)
                ON CONFLICT(camera_id) DO UPDATE SET high_epoch=excluded.high_epoch,pending_id=excluded.pending_id""",
                (camera_id, epoch, assignment_id),
            )
            return self._remember(payload, encoded, self._assignment(assignment_id))

    def result_progress(self, camera_id):
        """현재 활성화 경계의 최근 저장 사실만 공개하며 영상 본문은 내보내지 않는다."""
        with self.lock:
            current = self.camera(camera_id)
            current["latest_result"] = None
            if not current["accepting_results"]:
                return current
            owner = current["active"]
            now = self.clock()
            row = self.db.execute(
                """SELECT * FROM results
                WHERE assignment_id=? AND session_id=? AND activation_id=?
                  AND node_id=? AND event_sequence>? AND received_at>=?
                  AND received_at>? AND expires_at>?
                  AND decision IN ('accepted','observed')
                ORDER BY event_sequence DESC LIMIT 1""",
                (
                    owner["assignment_id"],
                    owner["session_id"],
                    owner["activation_id"],
                    owner["node_id"],
                    owner["sequence_floor"],
                    owner["activated_at"],
                    now - self.stale_after,
                    now,
                ),
            ).fetchone()
            if row is not None:
                edge = json.loads(row["body"])["edge"]
                current["latest_result"] = {
                    "event_id": row["event_id"],
                    "sha256": row["sha256"],
                    "kind": row["kind"],
                    "event_sequence": row["event_sequence"],
                    "captured_at": edge["captured_at"],
                    "received_at": _utc(row["received_at"]),
                    "expires_at": _utc(row["expires_at"]),
                    # 원격 시계 오차도 보이도록 음수 지연을 임의로 0으로 바꾸지 않는다.
                    "capture_to_receive_seconds": row["received_at"]
                    - _sample_time(edge["captured_at"]),
                }
            return current

    def view(self, camera_id):
        """현재 담당·실행·최근 저장을 함께 확인하며 영상 재생 확인과는 구분한다."""
        with self.lock:
            current = self.result_progress(camera_id)
            owner = current["active"]
            current.update(
                observed_at=_utc(self.clock()),
                status="preparing" if current["pending"] else "unassigned",
                reason=None,
                inference_ready=False,
                media_verified=False,
            )
            if owner is not None:
                current["status"] = "reconciling"
                current["reason"] = "controller_reconciling"
                if current["accepting_results"]:
                    try:
                        service = self._service(owner["node_id"], camera_id, fresh_run=True)
                        if (
                            service.get("status") != "running"
                            or service.get("ready") is not True
                            or service.get("backend") != "deepstream"
                            or service.get("models_verified") is not True
                            or any(
                                service.get(key) != owner[key]
                                for key in (
                                    "assignment_id",
                                    "epoch",
                                    "session_id",
                                    "supervisor_boot_id",
                                    "manifest_id",
                                    "camera_id",
                                )
                            )
                            or service.get("activation")
                            != {
                                "activation_id": owner["activation_id"],
                                "sequence_floor": owner["sequence_floor"],
                            }
                        ):
                            raise RegistryError(409, "owner_execution_unverified")
                        if current["latest_result"] is None:
                            current.update(
                                status="awaiting_results",
                                reason="recent_result_missing",
                            )
                        else:
                            current.update(
                                status="switching" if current["pending"] else "running",
                                reason=None,
                                inference_ready=True,
                            )
                    except RegistryError as error:
                        current.update(status="unavailable", reason=error.code)
            # 현재 배정 상태의 최근20건만 읽으며 원본 영상/이벤트 본문은 공개하지 않는다.
            rows = self.db.execute(
                "SELECT assignment_id,node_id,epoch,manifest_id,state,prepared_at,activated_at "
                "FROM assignments WHERE camera_id=? ORDER BY epoch DESC LIMIT 20",
                (camera_id,),
            ).fetchall()
            current["history"] = [dict(row) for row in rows]
            for item in current["history"]:
                for key in ("prepared_at", "activated_at"):
                    item[key] = _utc(item[key]) if item[key] is not None else None
            return current

    def activate(self, assignment_id, payload):
        _uuid(assignment_id)
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            encoded, duplicate = self._command(
                "activate:" + assignment_id,
                payload,
                {"session_id", "activation_id", "sequence_floor"},
            )
            if duplicate is not None:
                return duplicate
            assignment = self._assignment(assignment_id)
            _uuid(payload["session_id"])
            _uuid(payload["activation_id"])
            _epoch(payload["sequence_floor"], zero=True)
            if assignment["state"] not in ("preparing", "active"):
                raise RegistryError(409, "assignment_not_prepared")
            service = self._service(assignment["node_id"], assignment["camera_id"], fresh_run=True)
            activation = service.get("activation")
            if (
                not service["ready"]
                or service["backend"] != "deepstream"
                or not service["models_verified"]
                or service["assignment_id"] != assignment_id
                or service["epoch"] != assignment["epoch"]
                or service["supervisor_boot_id"] != assignment["supervisor_boot_id"]
                or service["camera_id"] != assignment["camera_id"]
                or service["session_id"] != payload["session_id"]
                or service["manifest_id"] != assignment["manifest_id"]
                or payload["activation_id"] != assignment["activation_id"]
                or activation
                != {
                    "activation_id": payload["activation_id"],
                    "sequence_floor": payload["sequence_floor"],
                }
            ):
                raise RegistryError(409, "activation_not_observed")
            if (
                assignment["sample_id"] is None
                or assignment["session_id"] != payload["session_id"]
                or not 0 <= self.clock() - assignment["sample_at"] < self.stale_after
            ):
                raise RegistryError(409, "central_sample_required")
            current = self.camera(assignment["camera_id"])
            old = current["active"]
            if assignment["state"] == "preparing":
                if (
                    not current["pending"]
                    or current["pending"]["assignment_id"] != assignment_id
                    or (
                        old["assignment_id"] if old else None,
                        old["epoch"] if old else 0,
                    )
                    != (assignment["expected_id"], assignment["expected_epoch"])
                ):
                    raise RegistryError(409, "owner_changed")
                if old:
                    self.db.execute(
                        "UPDATE assignments SET state='retired',verified_run=NULL WHERE assignment_id=?",
                        (old["assignment_id"],),
                    )
                self.db.execute(
                    "UPDATE cameras SET active_id=?,pending_id=NULL WHERE camera_id=?",
                    (assignment_id, assignment["camera_id"]),
                )
            elif not old or old["assignment_id"] != assignment_id:
                raise RegistryError(409, "owner_changed")
            elif assignment["verified_run"] == self.boot_id:
                raise RegistryError(409, "already_active")
            self.db.execute(
                "UPDATE assignments SET state='active',session_id=?,sequence_floor=?,verified_run=?,activated_at=? WHERE assignment_id=?",
                (
                    payload["session_id"],
                    payload["sequence_floor"],
                    self.boot_id,
                    self.clock(),
                    assignment_id,
                ),
            )
            return self._remember(payload, encoded, self._assignment(assignment_id))

    def abort(self, assignment_id, payload):
        _uuid(assignment_id)
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            encoded, duplicate = self._command("abort:" + assignment_id, payload, set())
            if duplicate is not None:
                return duplicate
            assignment = self._assignment(assignment_id)
            if assignment["state"] != "preparing":
                raise RegistryError(409, "assignment_not_prepared")
            self.db.execute(
                "UPDATE assignments SET state='aborted' WHERE assignment_id=?",
                (assignment_id,),
            )
            self.db.execute(
                "UPDATE cameras SET pending_id=NULL WHERE pending_id=?",
                (assignment_id,),
            )
            return self._remember(payload, encoded, self._assignment(assignment_id))

    def _decision(self, edge, expires_at):
        if self.clock() >= expires_at:
            return "expired"
        try:
            assignment = self._assignment(edge["assignment_id"])
        except RegistryError:
            return "unknown_assignment"
        if (
            assignment["node_id"],
            assignment["camera_id"],
            assignment["epoch"],
            assignment["supervisor_boot_id"],
        ) != (
            edge["node_id"],
            edge["camera_id"],
            edge["epoch"],
            edge["supervisor_boot_id"],
        ):
            return "assignment_identity_mismatch"
        current = self.camera(edge["camera_id"])
        if (
            not current["active"]
            or current["active"]["assignment_id"] != edge["assignment_id"]
        ):
            return "not_current_owner"
        if assignment["verified_run"] != self.boot_id:
            return "controller_reconciling"
        if (
            assignment["session_id"] != edge["session_id"]
            or assignment["activation_id"] != edge["activation_id"]
        ):
            return "inactive_execution"
        if edge["event_sequence"] <= assignment["sequence_floor"]:
            return "before_activation_boundary"
        return "current_owner"

    def _ack(self, row):
        return {
            "stored": True,
            "event_id": row["event_id"],
            "sha256": row["sha256"],
            "decision": row["decision"],
            "reason": row["reason"],
            "received_at": _utc(row["received_at"]),
        }

    def receive(self, node_id, payload):
        edge, event_id, captured, expires, body, digest = validate_result(
            node_id, payload, self.clock()
        )
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute(
                "SELECT * FROM results WHERE event_id=?", (event_id,)
            ).fetchone()
            if old:
                if old["node_id"] != node_id or old["sha256"] != digest:
                    raise RegistryError(409, "event_id_conflict")
                return self._ack(old)
            if self.db.execute(
                "SELECT 1 FROM results WHERE assignment_id=? AND session_id=? AND event_sequence=?",
                (edge["assignment_id"], edge["session_id"], edge["event_sequence"]),
            ).fetchone():
                raise RegistryError(409, "sequence_conflict")
            reason = self._decision(edge, expires)
            decision = (
                "accepted"
                if reason == "current_owner" and payload["kind"] == "event"
                else "rejected"
            )
            if payload["kind"] == "observation" and expires > self.clock():
                try:
                    assignment = self._assignment(edge["assignment_id"])
                    service = self._service(node_id, edge["camera_id"], fresh_run=True)
                    if (
                        assignment["state"] in ("active", "preparing")
                        and service["ready"]
                        and (
                            service["assignment_id"],
                            service["epoch"],
                            service["supervisor_boot_id"],
                            service["session_id"],
                            service["manifest_id"],
                            service["camera_id"],
                        )
                        == (
                            edge["assignment_id"],
                            edge["epoch"],
                            edge["supervisor_boot_id"],
                            edge["session_id"],
                            assignment["manifest_id"],
                            edge["camera_id"],
                        )
                        and assignment["node_id"] == node_id
                        and assignment["supervisor_boot_id"]
                        == edge["supervisor_boot_id"]
                        and self.started_at <= captured <= self.clock()
                        and self.clock() - captured < self.stale_after
                    ):
                        self.db.execute(
                            "UPDATE assignments SET session_id=?,sample_id=?,sample_at=? WHERE assignment_id=?",
                            (
                                edge["session_id"],
                                event_id,
                                self.clock(),
                                edge["assignment_id"],
                            ),
                        )
                        decision, reason = "observed", "preparation_sample"
                except RegistryError:
                    pass
            self.db.execute(
                """INSERT INTO results (
                event_id,assignment_id,session_id,event_sequence,activation_id,camera_id,node_id,kind,
                body,sha256,received_at,expires_at,decision,reason,delivery_state
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event_id,
                    edge["assignment_id"],
                    edge["session_id"],
                    edge["event_sequence"],
                    edge["activation_id"],
                    edge["camera_id"],
                    node_id,
                    payload["kind"],
                    body,
                    digest,
                    self.clock(),
                    expires,
                    decision,
                    reason,
                    "pending" if decision == "accepted" else "none",
                ),
            )
            result = self.db.execute(
                "SELECT * FROM results WHERE event_id=?", (event_id,)
            ).fetchone()
            return self._ack(result)

    def receipt(self, node_id, event_id):
        _identifier(node_id)
        _identifier(event_id)
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM results WHERE event_id=? AND node_id=?",
                (event_id, node_id),
            ).fetchone()
            if row is None:
                raise RegistryError(404, "receipt_not_found")
            return self._ack(row)

    def pending(self):
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            results = []
            response_bytes = 32
            for row in self.db.execute(
                "SELECT * FROM results WHERE delivery_state='pending' ORDER BY received_at,event_id LIMIT 100"
            ).fetchall():
                reason = self._decision(
                    json.loads(row["body"])["edge"], row["expires_at"]
                )
                if reason != "current_owner":
                    self.db.execute(
                        "UPDATE results SET delivery_state='discarded',delivery_reason=? WHERE event_id=?",
                        (reason, row["event_id"]),
                    )
                else:
                    item = {
                        "event_id": row["event_id"],
                        "sha256": row["sha256"],
                        "result": json.loads(row["body"]),
                    }
                    size = len(json.dumps(item, ensure_ascii=True).encode()) + 2
                    if response_bytes + size > 60000:
                        break
                    results.append(item)
                    response_bytes += size
            return results

    def consume(self, event_id, digest):
        """최종 모의 수신 기록을 현재 담당 검증과 같은 트랜잭션에서 한 번만 만든다."""
        _identifier(event_id)
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM results WHERE event_id=?", (event_id,)
            ).fetchone()
            if not row or row["sha256"] != digest:
                raise RegistryError(404, "result_not_found")
            if row["delivery_state"] == "delivered":
                return {"consumed": True, "duplicate": True, "event_id": event_id}
            if row["delivery_state"] != "pending":
                return {
                    "consumed": False,
                    "event_id": event_id,
                    "reason": row["delivery_reason"] or row["reason"],
                }
            reason = self._decision(json.loads(row["body"])["edge"], row["expires_at"])
            consumed = reason == "current_owner"
            self.db.execute(
                "UPDATE results SET delivery_state=?,delivery_reason=?,consumed_at=? WHERE event_id=?",
                (
                    "delivered" if consumed else "discarded",
                    reason,
                    self.clock() if consumed else None,
                    event_id,
                ),
            )
            return {
                "consumed": consumed,
                "duplicate": False,
                "event_id": event_id,
                "reason": reason,
            }

    def reject_mqtt(self, topic, payload, reason):
        """잘못된 원본도 저장한 뒤 ACK하여 정상 결과가 같은 지속 세션에서 막히지 않게 한다."""
        digest = hashlib.sha256(topic.encode() + b"\0" + payload).hexdigest()
        with self.lock, self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS rejected_mqtt (
                sha256 TEXT PRIMARY KEY, topic TEXT NOT NULL, payload BLOB NOT NULL,
                total_bytes INTEGER NOT NULL, reason TEXT NOT NULL, received_at REAL NOT NULL
            )""")
            self.db.execute(
                "INSERT OR IGNORE INTO rejected_mqtt VALUES (?,?,?,?,?,?)",
                (
                    digest,
                    topic[:512],
                    payload[:65536],
                    len(payload),
                    reason,
                    self.clock(),
                ),
            )

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.db.close()
                os.close(self.lock_fd)
