"""Opt-in record delivery. Existing alarm jobs are never replayed here."""

import json
import math
import time
from dataclasses import dataclass
from uuid import uuid4

from .event_ingest_store import EventIngestStore


def record_topic(destination: str) -> str:
    prefix = "record-mqtt:cctv/records/v1/"
    if (
        not destination.startswith(prefix)
        or not destination[len(prefix) :].strip()
        or any(char in destination for char in ("+", "#", "\x00"))
        or len(destination.encode("utf-8")) > 65535
    ):
        raise ValueError(
            "A dedicated record-mqtt:cctv/records/v1/<sink> destination is required"
        )
    return destination.removeprefix("record-mqtt:")


class RecordDeliveryStore(EventIngestStore):
    def __init__(self, path):
        super().__init__(path)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("PRAGMA user_version").fetchone()[0] == 2:
                for statement in (
                    """CREATE TABLE record_delivery_targets (
                        destination TEXT PRIMARY KEY, endpoint TEXT NOT NULL)""",
                    """CREATE TABLE record_delivery_state (
                        job_id TEXT PRIMARY KEY REFERENCES delivery_jobs(job_id),
                        lease_token TEXT, lease_until REAL,
                        next_attempt_at REAL NOT NULL DEFAULT 0,
                        ack_kind TEXT, acked_at REAL)""",
                    """CREATE TABLE record_delivery_attempts (
                        lease_token TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES delivery_jobs(job_id),
                        attempt INTEGER NOT NULL, started_at REAL NOT NULL,
                        finished_at REAL, outcome TEXT NOT NULL
                            CHECK(outcome IN ('sending','uncertain','lease_expired','broker_acked')),
                        error TEXT, UNIQUE(job_id, attempt))""",
                    "CREATE INDEX record_delivery_due ON record_delivery_state(acked_at, next_attempt_at)",
                ):
                    connection.execute(statement)
                connection.execute("PRAGMA user_version=3")
            else:
                connection.execute(
                    "SELECT endpoint FROM record_delivery_targets LIMIT 0"
                )
                connection.execute(
                    "SELECT lease_token, acked_at FROM record_delivery_state LIMIT 0"
                )
                connection.execute(
                    "SELECT outcome FROM record_delivery_attempts LIMIT 0"
                )

    @staticmethod
    def _bind_target(connection, destination, endpoint):
        record_topic(destination)
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("A stable endpoint identity is required")
        connection.execute(
            "INSERT OR IGNORE INTO record_delivery_targets VALUES (?, ?)",
            (destination, endpoint),
        )
        existing = connection.execute(
            "SELECT endpoint FROM record_delivery_targets WHERE destination=?",
            (destination,),
        ).fetchone()[0]
        if existing != endpoint:
            raise ValueError(
                "Destination endpoint changed; use a new destination version"
            )

    def enqueue_pending(
        self, namespace: str, destination: str, endpoint: str, limit: int = 100
    ) -> int:
        """Reconcile canonical records into a separate, idempotent record-only route."""
        if not namespace.strip() or not 1 <= limit <= 1000:
            raise ValueError("namespace and bounded positive limit are required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._bind_target(connection, destination, endpoint)
            records = connection.execute(
                "SELECT c.source_record_id FROM canonical_records c "
                "WHERE c.source_namespace=? AND NOT EXISTS (SELECT 1 FROM delivery_jobs j "
                "WHERE j.source_namespace=c.source_namespace AND j.source_record_id=c.source_record_id "
                "AND j.destination=?) ORDER BY c.rowid LIMIT ?",
                (namespace, destination, limit),
            ).fetchall()
            for record in records:
                job_id = "job_" + uuid4().hex
                connection.execute(
                    "INSERT INTO delivery_jobs(job_id,source_namespace,source_record_id,destination,state) "
                    "VALUES (?, ?, ?, ?, 'pending')",
                    (job_id, namespace, record[0], destination),
                )
                connection.execute(
                    "INSERT INTO record_delivery_state(job_id) VALUES (?)", (job_id,)
                )
            return len(records)

    def claim(
        self,
        destination: str,
        endpoint: str,
        *,
        lease_seconds: float,
        now: float | None = None,
    ) -> dict | None:
        now = time.time() if now is None else now
        if (
            not math.isfinite(now)
            or not math.isfinite(lease_seconds)
            or lease_seconds <= 0
        ):
            raise ValueError("finite time and positive lease are required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._bind_target(connection, destination, endpoint)
            if connection.execute("PRAGMA user_version").fetchone()[0] >= 4:
                if connection.execute(
                    "SELECT 1 FROM monitoring_targets WHERE destination=?",
                    (destination,),
                ).fetchone():
                    return None
            row = connection.execute(
                "SELECT j.*, s.lease_token, c.payload_json FROM delivery_jobs j "
                "JOIN record_delivery_state s USING(job_id) JOIN canonical_records c "
                "ON c.source_namespace=j.source_namespace AND c.source_record_id=j.source_record_id "
                "WHERE j.destination=? AND s.acked_at IS NULL AND s.next_attempt_at<=? "
                "AND (s.lease_token IS NULL OR s.lease_until<=?) "
                "AND j.state IN ('pending','sending','uncertain') "
                "ORDER BY s.next_attempt_at, j.rowid LIMIT 1",
                (destination, now, now),
            ).fetchone()
            if row is None:
                return None
            job = dict(row)
            if job["lease_token"] is not None:
                connection.execute(
                    "UPDATE record_delivery_attempts SET outcome='lease_expired', finished_at=? "
                    "WHERE lease_token=? AND outcome='sending'",
                    (now, job["lease_token"]),
                )
            token = uuid4().hex
            job.update(lease_token=token, attempts=job["attempts"] + 1)
            connection.execute(
                "UPDATE record_delivery_state SET lease_token=?, lease_until=? WHERE job_id=?",
                (token, now + lease_seconds, job["job_id"]),
            )
            connection.execute(
                "UPDATE delivery_jobs SET state='sending', attempts=attempts+1, last_error=NULL WHERE job_id=?",
                (job["job_id"],),
            )
            connection.execute(
                "INSERT INTO record_delivery_attempts(lease_token,job_id,attempt,started_at,outcome) "
                "VALUES (?, ?, ?, ?, 'sending')",
                (token, job["job_id"], job["attempts"], now),
            )
            job["topic"] = record_topic(destination)
            record = json.loads(job["payload_json"])
            id_field = (
                "observation_id"
                if record.get("message_type") == "sensor_observation"
                else "event_id"
            )
            job["wire_payload"] = json.dumps(
                {
                    "schema_version": "1.0",
                    "delivery_id": job["job_id"],
                    "source_namespace": job["source_namespace"],
                    id_field: job["source_record_id"],
                    "record": record,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            return job

    def finish(
        self,
        job: dict,
        *,
        acknowledged: bool,
        error: str | None = None,
        now: float | None = None,
    ) -> bool:
        """Only the unexpired lease owner can save a QoS 1 broker ACK or retry."""
        now = time.time() if now is None else now
        if not math.isfinite(now):
            raise ValueError("finite time is required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT j.attempts FROM delivery_jobs j JOIN record_delivery_state s USING(job_id) "
                "WHERE j.job_id=? AND j.state='sending' AND s.lease_token=? AND s.lease_until>?",
                (job["job_id"], job["lease_token"], now),
            ).fetchone()
            if row is None:
                return False
            delay = min(60, 2 ** min(row["attempts"] - 1, 6))
            connection.execute(
                "UPDATE record_delivery_state SET lease_token=NULL, lease_until=NULL, "
                "next_attempt_at=?, ack_kind=?, acked_at=? WHERE job_id=?",
                (
                    now + delay,
                    "broker_puback" if acknowledged else None,
                    now if acknowledged else None,
                    job["job_id"],
                ),
            )
            connection.execute(
                "UPDATE delivery_jobs SET state=?,last_error=? WHERE job_id=?",
                ("submitted" if acknowledged else "uncertain", error, job["job_id"]),
            )
            connection.execute(
                "UPDATE record_delivery_attempts SET finished_at=?,outcome=?,error=? WHERE lease_token=?",
                (
                    now,
                    "broker_acked" if acknowledged else "uncertain",
                    error,
                    job["lease_token"],
                ),
            )
            return True


@dataclass(frozen=True)
class BrokerAck:
    job_id: str


class RecordDeliveryWorker:
    """One caller-owned worker per destination; network I/O stays outside SQLite."""

    def __init__(
        self,
        store,
        sender,
        *,
        namespace: str,
        destination: str,
        send_timeout: float = 5,
        lease_seconds: float = 30,
    ):
        record_topic(destination)
        if (
            not math.isfinite(send_timeout)
            or send_timeout <= 0
            or not math.isfinite(lease_seconds)
            or lease_seconds <= send_timeout + 2
        ):
            raise ValueError(
                "lease must exceed finite send timeout plus cleanup margin"
            )
        self.store, self.sender = store, sender
        self.namespace, self.destination = namespace, destination
        self.send_timeout, self.lease_seconds = send_timeout, lease_seconds

    def drain_once(self, limit: int = 100) -> int:
        self.store.enqueue_pending(
            self.namespace, self.destination, self.sender.endpoint, limit
        )
        completed = 0
        for _ in range(limit):
            job = self.store.claim(
                self.destination, self.sender.endpoint, lease_seconds=self.lease_seconds
            )
            if job is None:
                break
            acknowledged, error = False, None
            try:
                ack = self.sender.send(job, timeout=self.send_timeout)
                if not isinstance(ack, BrokerAck) or ack.job_id != job["job_id"]:
                    raise ValueError("A matching broker acknowledgement is required")
                acknowledged = True
            except Exception as exc:
                error = type(exc).__name__
            if (
                self.store.finish(job, acknowledged=acknowledged, error=error)
                and acknowledged
            ):
                completed += 1
        return completed
