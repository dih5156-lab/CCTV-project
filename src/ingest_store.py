"""Durable input receipts. Sensor projection state is the first consumer."""

import base64
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


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
        connection = sqlite3.connect(self.path, timeout=0.25)
        try:
            connection.row_factory = sqlite3.Row
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise ValueError("Ingest store requires DELETE journal mode")
            connection.execute("PRAGMA synchronous=EXTRA")
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

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

    def pending(self, limit: int = 100, namespace: str = "tlv_mqtt") -> list[str]:
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT s.ingest_id FROM sensor_ingest_state s "
                    "JOIN ingest_receipts r USING (ingest_id) "
                    "WHERE r.source_namespace=? AND s.stage IN ('pending','ready') "
                    "AND s.next_attempt_at <= ? ORDER BY s.rowid LIMIT ?",
                    (namespace, time.time(), limit),
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
