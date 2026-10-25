"""Event receipts and atomic normalization; no network or device operations."""

# Jetson B의 Python 3.8에서도 영속 이벤트 저장소의 타입 표기를 읽는다.
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from base64 import b64encode
from datetime import datetime, timezone
from uuid import uuid4

from .ingest_store import IngestStore

logger = logging.getLogger(__name__)


class EventIngestStore(IngestStore):
    def __init__(self, path, *, commit_observer=None):
        super().__init__(path, commit_observer=commit_observer)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("PRAGMA user_version").fetchone()[0] == 1:
                statements = (
                    """CREATE TABLE event_ingest_state (
                        ingest_id TEXT PRIMARY KEY REFERENCES ingest_receipts(ingest_id),
                        stage TEXT NOT NULL DEFAULT 'pending'
                            CHECK(stage IN ('pending','accepted','duplicate','conflict','invalid')),
                        destination TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                        next_attempt_at REAL NOT NULL DEFAULT 0, last_error TEXT)""",
                    """CREATE TABLE canonical_records (
                        source_namespace TEXT NOT NULL, source_record_id TEXT NOT NULL,
                        first_ingest_id TEXT NOT NULL REFERENCES ingest_receipts(ingest_id),
                        source_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL,
                        PRIMARY KEY(source_namespace, source_record_id))""",
                    """CREATE TABLE ingest_revisions (
                        ingest_id TEXT NOT NULL REFERENCES ingest_receipts(ingest_id),
                        revision INTEGER NOT NULL, converter_version TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('accepted','duplicate','conflict','invalid')),
                        payload_json TEXT, errors_json TEXT,
                        PRIMARY KEY(ingest_id, revision))""",
                    """CREATE TABLE delivery_jobs (
                        job_id TEXT PRIMARY KEY, source_namespace TEXT NOT NULL,
                        source_record_id TEXT NOT NULL, destination TEXT NOT NULL,
                        state TEXT NOT NULL CHECK(state IN ('pending','held','sending','submitted','uncertain')),
                        attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
                        UNIQUE(source_namespace, source_record_id, destination),
                        FOREIGN KEY(source_namespace, source_record_id)
                            REFERENCES canonical_records(source_namespace, source_record_id))""",
                    "CREATE INDEX event_ingest_due ON event_ingest_state(stage, next_attempt_at)",
                )
                for statement in statements:
                    connection.execute(statement)
                connection.execute("PRAGMA user_version=2")
            else:
                connection.execute("SELECT destination FROM event_ingest_state LIMIT 0")
                connection.execute("SELECT payload_json FROM canonical_records LIMIT 0")
                connection.execute(
                    "SELECT converter_version FROM ingest_revisions LIMIT 0"
                )
                connection.execute("SELECT state FROM delivery_jobs LIMIT 0")

    def accept_message(
        self, namespace: str, topic: str, payload: bytes, *, destination: str | None
    ) -> str:
        """Commit original bytes before decoding; destination is frozen at receipt."""
        envelope = json.dumps(
            {"topic": topic, "payload_base64": b64encode(payload).decode("ascii")},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        return self._accept_event_raw(
            namespace, envelope, "mqtt_bytes_base64", destination
        )

    def accept_event_json(
        self,
        namespace: str,
        raw: dict,
        *,
        destination: str | None,
        busy_retries: int = 0,
    ) -> str:
        """Freeze input; optionally retry one rolled-back SQLITE_BUSY transaction."""
        try:
            payload = json.dumps(
                raw, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            raw_format = "json"
        except (TypeError, ValueError, OverflowError, RecursionError):
            raw_format = "error_repr"
            try:
                payload = json.dumps(
                    raw, ensure_ascii=False, sort_keys=True, default=repr
                )
            except (TypeError, ValueError, OverflowError, RecursionError):
                payload = json.dumps({"representation": repr(raw)}, ensure_ascii=False)
        return self._accept_event_raw(
            namespace, payload, raw_format, destination, busy_retries=busy_retries
        )

    def _accept_event_raw(
        self, namespace, payload, raw_format, destination, *, busy_retries=0
    ) -> str:
        if type(busy_retries) is not int or busy_retries not in (0, 1):
            raise ValueError("busy_retries must be 0 or 1")
        ingest_id = "ing_" + uuid4().hex
        received_at = datetime.now(timezone.utc).isoformat()
        raw_sha256 = hashlib.sha256(payload.encode()).hexdigest()
        for attempt in range(busy_retries + 1):
            try:
                with self._connect() as connection:
                    connection.execute(
                        "INSERT INTO ingest_receipts "
                        "(ingest_id, source_namespace, received_at, raw_format, raw_payload, raw_sha256) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            ingest_id,
                            namespace,
                            received_at,
                            raw_format,
                            payload,
                            raw_sha256,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO event_ingest_state(ingest_id, destination) VALUES (?, ?)",
                        (ingest_id, destination),
                    )
                break
            except sqlite3.OperationalError as error:
                code = getattr(error, "sqlite_errorcode", None)
                busy = (
                    code & 0xFF == 5
                    if code is not None
                    else str(error) == "database is locked"
                )
                if not busy or attempt >= busy_retries:
                    raise
                # _connect has rolled back and closed; reuse the frozen receipt.
                logger.warning(
                    "Event intake busy; retrying ingest_id=%s retry=%s",
                    ingest_id,
                    attempt + 1,
                )
        self._notify_commit(
            "ingest_commit_completed_at", ingest_id, received_at=received_at
        )
        return ingest_id

    def get_event_receipt(self, ingest_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT r.*, s.stage, s.destination, s.attempts, s.last_error "
                "FROM ingest_receipts r JOIN event_ingest_state s USING (ingest_id) "
                "WHERE ingest_id=?",
                (ingest_id,),
            ).fetchone()
        if row is None:
            raise KeyError(ingest_id)
        result = dict(row)
        result["raw"] = json.loads(result["raw_payload"])
        return result

    def pending_events(self, namespace: str, limit: int = 100) -> list[str]:
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT s.ingest_id FROM event_ingest_state s "
                    "JOIN ingest_receipts r USING (ingest_id) "
                    "WHERE r.source_namespace=? AND s.stage='pending' "
                    "AND s.next_attempt_at<=? ORDER BY s.rowid LIMIT ?",
                    (namespace, time.time(), limit),
                )
            ]

    def record_event(
        self,
        ingest_id: str,
        event: dict,
        *,
        converter_version: str,
        recovery: bool = False,
    ) -> tuple[str, str | None]:
        """Atomically save a revision, canonical record and optional delivery job."""
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("A stable event_id is required")
        return self._record_payload(
            ingest_id,
            event_id,
            event,
            converter_version=converter_version,
            recovery=recovery,
        )

    def record_observation(
        self, ingest_id: str, observation: dict, *, converter_version: str
    ):
        observation_id = observation.get("observation_id")
        if (
            observation.get("message_type") != "sensor_observation"
            or not isinstance(observation_id, str)
            or not observation_id.strip()
        ):
            raise ValueError(
                "A sensor observation with a stable observation_id is required"
            )
        return self._record_payload(
            ingest_id,
            observation_id,
            observation,
            converter_version=converter_version,
            recovery=True,
        )

    def _record_payload(
        self, ingest_id, record_id, record, *, converter_version, recovery
    ):
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            receipt = connection.execute(
                "SELECT r.*, s.stage, s.destination FROM ingest_receipts r "
                "JOIN event_ingest_state s USING (ingest_id) WHERE ingest_id=?",
                (ingest_id,),
            ).fetchone()
            if receipt is None:
                raise KeyError(ingest_id)
            if receipt["stage"] != "pending":
                return receipt["stage"], None
            namespace = receipt["source_namespace"]
            existing = connection.execute(
                "SELECT source_sha256 FROM canonical_records "
                "WHERE source_namespace=? AND source_record_id=?",
                (namespace, record_id),
            ).fetchone()
            status, job_id = "accepted", None
            if existing is not None:
                status = (
                    "duplicate" if existing[0] == receipt["raw_sha256"] else "conflict"
                )
            else:
                connection.execute(
                    "INSERT INTO canonical_records VALUES (?, ?, ?, ?, ?)",
                    (namespace, record_id, ingest_id, receipt["raw_sha256"], payload),
                )
                if receipt["destination"] is not None:
                    job_id = "job_" + uuid4().hex
                    connection.execute(
                        "INSERT INTO delivery_jobs "
                        "(job_id, source_namespace, source_record_id, destination, state) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            job_id,
                            namespace,
                            record_id,
                            receipt["destination"],
                            "held" if recovery else "pending",
                        ),
                    )
            connection.execute(
                "INSERT INTO ingest_revisions "
                "(ingest_id, revision, converter_version, status, payload_json) "
                "VALUES (?, 1, ?, ?, ?)",
                (ingest_id, converter_version, status, payload),
            )
            connection.execute(
                "UPDATE event_ingest_state SET stage=?, last_error=NULL WHERE ingest_id=?",
                (status, ingest_id),
            )
        self._notify_commit(
            "normalize_completed_at", ingest_id, received_at=receipt["received_at"]
        )
        return status, job_id

    def reject_event(
        self, ingest_id: str, errors: list[dict], *, converter_version: str
    ) -> None:
        errors_json = json.dumps(errors, ensure_ascii=False, allow_nan=False)
        with self._connect() as connection:
            updated = connection.execute(
                "UPDATE event_ingest_state SET stage='invalid', last_error=? "
                "WHERE ingest_id=? AND stage='pending'",
                (errors_json[:512], ingest_id),
            ).rowcount
            if updated:
                connection.execute(
                    "INSERT INTO ingest_revisions "
                    "(ingest_id, revision, converter_version, status, errors_json) "
                    "VALUES (?, 1, ?, 'invalid', ?)",
                    (ingest_id, converter_version, errors_json),
                )

    def retry_event(self, ingest_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE event_ingest_state SET attempts=attempts+1, last_error=?, "
                "next_attempt_at=? + MIN(60, (1 << MIN(attempts, 6))) "
                "WHERE ingest_id=? AND stage='pending'",
                (error[:512], time.time(), ingest_id),
            )

    def begin_live_delivery(self, job_id: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "UPDATE delivery_jobs SET state='sending', attempts=attempts+1 "
                    "WHERE job_id=? AND state='pending' "
                    "AND destination NOT LIKE 'record-mqtt:%'",
                    (job_id,),
                ).rowcount
                == 1
            )

    def finish_live_delivery(
        self, job_id: str, *, submitted: bool, error: str | None
    ) -> None:
        """Publisher submission is NOT broker ACK, consumer ACK or device success."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE delivery_jobs SET state=?, last_error=? "
                "WHERE job_id=? AND state='sending' "
                "AND destination NOT LIKE 'record-mqtt:%'",
                ("submitted" if submitted else "uncertain", error, job_id),
            )
