"""Durable input receipts. Sensor projection state is the first consumer."""

# Jetson B의 Python 3.8에서도 영속 접수 경로의 타입 표기를 읽는다.
from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock
from uuid import uuid4
from weakref import WeakValueDictionary

_connection_locks = WeakValueDictionary()
_connection_locks_guard = Lock()


def _reset_connection_locks():
    global _connection_locks, _connection_locks_guard
    # fork 자식에는 다른 부모 스레드가 잡은 잠금을 물려주지 않는다.
    _connection_locks = WeakValueDictionary()
    _connection_locks_guard = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_connection_locks)


def _connection_lock(path):
    key = path.resolve()
    with _connection_locks_guard:
        lock = _connection_locks.get(key)
        if lock is None:
            lock = RLock()
            _connection_locks[key] = lock
        return lock


class IngestStore:
    def __init__(self, path: str | Path, *, commit_observer=None):
        self.commit_observer = commit_observer
        self.commit_observer_errors = 0
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3, 4):
                raise ValueError(f"Unsupported ingest schema version: {version}")
            if version == 0:
                if connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchone():
                    raise ValueError(
                        "Unversioned existing database is not an ingest schema"
                    )
                connection.execute(
                    """
                    CREATE TABLE ingest_receipts (
                        ingest_id TEXT PRIMARY KEY NOT NULL,
                        source_namespace TEXT NOT NULL, device_id TEXT,
                        received_at TEXT NOT NULL, raw_format TEXT NOT NULL,
                        raw_payload TEXT NOT NULL, raw_sha256 TEXT NOT NULL,
                        mode TEXT NOT NULL DEFAULT 'live'
                    )
                """
                )
                connection.execute(
                    """
                    CREATE TABLE sensor_ingest_state (
                        ingest_id TEXT PRIMARY KEY REFERENCES ingest_receipts(ingest_id),
                        stage TEXT NOT NULL CHECK(stage IN ('pending','ready','complete','quarantined')),
                        projection_json TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                        next_attempt_at REAL NOT NULL DEFAULT 0, last_error TEXT
                    )
                """
                )
                connection.execute(
                    "CREATE INDEX sensor_ingest_due ON sensor_ingest_state(stage, next_attempt_at)"
                )
                connection.execute("PRAGMA user_version=1")
            else:
                connection.execute("SELECT ingest_id FROM sensor_ingest_state LIMIT 0")
                connection.execute("SELECT raw_payload FROM ingest_receipts LIMIT 0")

    @contextmanager
    def _connect(self):
        # 같은 DB의 로컬 연결 경합을 줄이되, 오래 열린 연결만으로 새 BUSY를 만들지 않는다.
        # ponytail: 250ms 안에 조정하지 못하면 SQLite에 맡긴다. 지속 경합은 별도 쓰기 조정이 필요하다.
        lock = _connection_lock(self.path)
        acquired = lock.acquire(timeout=0.25)
        try:
            # closing은 설정·commit/rollback 실패에도 연결을 닫고, 바깥 finally는 진입 잠금을 푼다.
            with closing(sqlite3.connect(self.path, timeout=0.25)) as connection:
                connection.row_factory = sqlite3.Row
                if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                    raise ValueError("Ingest store requires DELETE journal mode")
                connection.execute("PRAGMA synchronous=EXTRA")
                connection.execute("PRAGMA foreign_keys=ON")
                with connection:
                    yield connection
        finally:
            if acquired:
                lock.release()

    def accept(self, namespace: str, device_id: str, raw: dict) -> str:
        try:
            payload = json.dumps(
                raw, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            raw_format, stage, error = "json", "pending", None
        except ValueError:
            payload = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            raw_format, stage, error = "error_repr", "quarantined", "non_finite_json"
        return self._accept(namespace, device_id, payload, raw_format, stage, error)

    def _notify_commit(self, phase, ingest_id, *, received_at):
        if self.commit_observer is not None:
            try:
                self.commit_observer(phase, ingest_id, received_at=received_at)
            except Exception:
                # An observer failure cannot undo a committed input or request a new receipt.
                self.commit_observer_errors += 1

    def accept_mqtt(self, namespace: str, topic: str, payload: bytes) -> str:
        raw = json.dumps(
            {
                "topic": topic,
                "payload_base64": base64.b64encode(payload).decode("ascii"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return self._accept(namespace, None, raw, "mqtt_bytes_base64", "pending", None)

    def _accept(self, namespace, device_id, payload, raw_format, stage, error) -> str:
        ingest_id = "ing_" + uuid4().hex
        received_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO ingest_receipts "
                "(ingest_id, source_namespace, device_id, received_at, raw_format, raw_payload, raw_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ingest_id,
                    namespace,
                    device_id,
                    received_at,
                    raw_format,
                    payload,
                    hashlib.sha256(payload.encode()).hexdigest(),
                ),
            )
            connection.execute(
                "INSERT INTO sensor_ingest_state(ingest_id, stage, last_error) VALUES (?, ?, ?)",
                (ingest_id, stage, error),
            )
        self._notify_commit(
            "ingest_commit_completed_at", ingest_id, received_at=received_at
        )
        return ingest_id

    def get(self, ingest_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT r.*, s.stage, s.projection_json, s.attempts, s.last_error "
                "FROM ingest_receipts r JOIN sensor_ingest_state s USING (ingest_id) "
                "WHERE ingest_id=?",
                (ingest_id,),
            ).fetchone()
        if row is None:
            raise KeyError(ingest_id)
        result = dict(row)
        result["raw"] = json.loads(result["raw_payload"])
        return result

    def pending(self, limit: int = 100, namespace: str | tuple[str, ...] = "tlv_mqtt") -> list[str]:
        # MQTT와 REST를 함께 소비할 때도 원래 접수 순서와 조회 상한을 유지한다.
        namespaces = (namespace,) if isinstance(namespace, str) else namespace
        placeholders = ",".join("?" for _ in namespaces)
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT s.ingest_id FROM sensor_ingest_state s "
                    "JOIN ingest_receipts r USING (ingest_id) "
                    f"WHERE r.source_namespace IN ({placeholders}) AND s.stage IN ('pending','ready') "
                    "AND s.next_attempt_at <= ? ORDER BY s.rowid LIMIT ?",
                    (*namespaces, time.time(), limit),
                )
            ]

    def prepare(self, ingest_id: str, projection_json: str) -> None:
        json.loads(projection_json)
        with self._connect() as connection:
            connection.execute(
                "UPDATE sensor_ingest_state SET stage='ready', projection_json=?, "
                "last_error=NULL, next_attempt_at=0 WHERE ingest_id=? AND stage='pending'",
                (projection_json, ingest_id),
            )

    def complete(self, ingest_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sensor_ingest_state SET stage='complete', last_error=NULL "
                "WHERE ingest_id=? AND stage='ready'",
                (ingest_id,),
            )

    def quarantine(self, ingest_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sensor_ingest_state SET stage='quarantined', last_error=? "
                "WHERE ingest_id=? AND stage IN ('pending','ready')",
                (error[:512], ingest_id),
            )

    def retry(self, ingest_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sensor_ingest_state SET attempts=attempts+1, last_error=?, "
                "next_attempt_at=? + MIN(60, (1 << MIN(attempts, 6))) "
                "WHERE ingest_id=? AND stage IN ('pending','ready')",
                (error[:512], time.time(), ingest_id),
            )
