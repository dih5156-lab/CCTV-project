"""Durable sender state for consumer ACKs; broker history remains independent."""

import hashlib
import json
import math
import time
from uuid import uuid4

from .monitoring_contract import (
    MonitoringBinding,
    MonitoringContractError,
    parse_record,
    validate_ack,
)
from .record_delivery import BrokerAck, RecordDeliveryStore, record_topic


def _now(value):
    value = time.time() if value is None else value
    if not math.isfinite(value):
        raise ValueError("finite time is required")
    return value


class MonitoringDeliveryStore(RecordDeliveryStore):
    def __init__(self, path):
        super().__init__(path)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("PRAGMA user_version").fetchone()[0] == 3:
                for statement in (
                    """CREATE TABLE monitoring_origin (
                        singleton INTEGER PRIMARY KEY CHECK(singleton=1), origin_id TEXT NOT NULL)""",
                    """CREATE TABLE monitoring_targets (
                        destination TEXT PRIMARY KEY REFERENCES record_delivery_targets(destination),
                        consumer_id TEXT NOT NULL, reader_endpoint TEXT NOT NULL)""",
                    """CREATE TABLE monitoring_delivery (
                        job_id TEXT PRIMARY KEY REFERENCES delivery_jobs(job_id),
                        wire_payload BLOB NOT NULL, payload_sha256 TEXT NOT NULL,
                        state TEXT NOT NULL CHECK(state IN
                            ('pending','awaiting_ack','uncertain','stored','quarantined','blocked')),
                        attempts INTEGER NOT NULL DEFAULT 0,
                        next_attempt_at REAL NOT NULL DEFAULT 0, lease_token TEXT, lease_until REAL,
                        ack_json TEXT, acked_at REAL, last_error TEXT)""",
                    """CREATE TABLE monitoring_attempts (
                        lease_token TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES monitoring_delivery(job_id),
                        started_at REAL NOT NULL, finished_at REAL, outcome TEXT NOT NULL,
                        error TEXT, response_json TEXT)""",
                    "CREATE INDEX monitoring_due ON monitoring_delivery(state,next_attempt_at)",
                ):
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO monitoring_origin VALUES (1, ?)",
                    ("origin_" + uuid4().hex,),
                )
                connection.execute("PRAGMA user_version=4")
            self.origin_id = connection.execute(
                "SELECT origin_id FROM monitoring_origin WHERE singleton=1"
            ).fetchone()[0]

    def _monitoring_bound(self, connection, binding, endpoint, reader_endpoint):
        if (
            binding.origin_id != self.origin_id
            or not isinstance(reader_endpoint, str)
            or not reader_endpoint.strip()
        ):
            raise MonitoringContractError("binding_mismatch")
        record_topic(binding.destination)
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("A stable endpoint identity is required")
        target = connection.execute(
            "SELECT t.endpoint,m.consumer_id,m.reader_endpoint FROM record_delivery_targets t "
            "LEFT JOIN monitoring_targets m USING(destination) WHERE t.destination=?",
            (binding.destination,),
        ).fetchone()
        if target is None:
            return False
        if target[0] != endpoint:
            raise ValueError(
                "Destination endpoint changed; use a new destination version"
            )
        if target[1] is None:
            return False
        if (target[1], target[2]) != (binding.consumer_id, reader_endpoint):
            raise MonitoringContractError("binding_mismatch")
        return True

    def _bind_monitoring(self, connection, binding, endpoint, reader_endpoint):
        if self._monitoring_bound(connection, binding, endpoint, reader_endpoint):
            return
        self._bind_target(connection, binding.destination, endpoint)
        if connection.execute(
            "SELECT 1 FROM delivery_jobs WHERE destination=? AND (attempts>0 OR state!='pending') LIMIT 1",
            (binding.destination,),
        ).fetchone():
            raise ValueError(
                "Previously attempted destination requires a new destination version"
            )
        connection.execute(
            "INSERT INTO monitoring_targets VALUES (?, ?, ?)",
            (binding.destination, binding.consumer_id, reader_endpoint),
        )

    def enqueue_monitoring(
        self, namespace, binding, endpoint, reader_endpoint, limit=100
    ):
        if (
            not isinstance(namespace, str)
            or not namespace.strip()
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise ValueError("namespace and limit 1..1000 are required")
        with self._connect() as connection:
            query = (
                "SELECT c.*, j.job_id FROM canonical_records c LEFT JOIN delivery_jobs j "
                "ON j.source_namespace=c.source_namespace AND j.source_record_id=c.source_record_id AND j.destination=? "
                "WHERE c.source_namespace=? AND NOT EXISTS "
                "(SELECT 1 FROM monitoring_delivery m WHERE m.job_id=j.job_id) ORDER BY c.rowid LIMIT ?"
            )
            if self._monitoring_bound(connection, binding, endpoint, reader_endpoint):
                if (
                    connection.execute(
                        query, (binding.destination, namespace, 1)
                    ).fetchone()
                    is None
                ):
                    return 0
            # Recheck under the writer lock: another worker may have enqueued it.
            connection.execute("BEGIN IMMEDIATE")
            self._bind_monitoring(connection, binding, endpoint, reader_endpoint)
            rows = connection.execute(
                query,
                (binding.destination, namespace, limit),
            ).fetchall()
            for row in rows:
                job_id = row["job_id"] or "job_" + uuid4().hex
                if row["job_id"] is None:
                    connection.execute(
                        "INSERT INTO delivery_jobs(job_id,source_namespace,source_record_id,destination,state) "
                        "VALUES (?, ?, ?, ?, 'pending')",
                        (
                            job_id,
                            namespace,
                            row["source_record_id"],
                            binding.destination,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO record_delivery_state(job_id) VALUES (?)",
                        (job_id,),
                    )
                state, error = "pending", None
                wire = row["payload_json"].encode("utf-8")
                try:
                    record = json.loads(row["payload_json"])
                    id_field = (
                        "observation_id"
                        if record.get("message_type") == "sensor_observation"
                        else "event_id"
                    )
                    wire = json.dumps(
                        {
                            "schema_version": "1.0",
                            "delivery_id": job_id,
                            "source_namespace": namespace,
                            id_field: row["source_record_id"],
                            "record": record,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ).encode("utf-8")
                    parse_record(wire)
                except (ValueError, TypeError, AttributeError, RecursionError):
                    state, error = "quarantined", "invalid"
                connection.execute(
                    "INSERT INTO monitoring_delivery(job_id,wire_payload,payload_sha256,state,last_error) VALUES (?, ?, ?, ?, ?)",
                    (job_id, wire, hashlib.sha256(wire).hexdigest(), state, error),
                )
            return len(rows)

    def claim_monitoring(
        self, binding, endpoint, reader_endpoint, *, lease_seconds, now=None
    ):
        now = _now(now)
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("positive finite lease is required")
        with self._connect() as connection:
            query = (
                "SELECT m.*, j.destination,c.first_ingest_id,r.received_at FROM monitoring_delivery m "
                "JOIN delivery_jobs j USING(job_id) JOIN canonical_records c "
                "ON c.source_namespace=j.source_namespace AND c.source_record_id=j.source_record_id "
                "JOIN ingest_receipts r ON r.ingest_id=c.first_ingest_id "
                "WHERE j.destination=? AND m.state IN ('pending','awaiting_ack','uncertain') "
                "AND m.next_attempt_at<=? AND (m.lease_token IS NULL OR m.lease_until<=?) "
                "ORDER BY m.next_attempt_at, m.rowid LIMIT 1"
            )
            parameters = (binding.destination, now, now)
            if self._monitoring_bound(connection, binding, endpoint, reader_endpoint):
                if connection.execute(query, parameters).fetchone() is None:
                    return None
            # A candidate is advisory until checked again under the writer lock.
            connection.execute("BEGIN IMMEDIATE")
            self._bind_monitoring(connection, binding, endpoint, reader_endpoint)
            row = connection.execute(query, parameters).fetchone()
            if row is None:
                return None
            job = dict(row)
            if job["lease_token"]:
                connection.execute(
                    "UPDATE monitoring_attempts SET outcome='lease_expired',finished_at=? "
                    "WHERE lease_token=? AND finished_at IS NULL",
                    (now, job["lease_token"]),
                )
                connection.execute(
                    "UPDATE record_delivery_attempts SET outcome='lease_expired',finished_at=? "
                    "WHERE lease_token=? AND outcome='sending'",
                    (now, job["lease_token"]),
                )
            token = uuid4().hex
            connection.execute(
                "UPDATE monitoring_delivery SET lease_token=?,lease_until=?,attempts=attempts+1 WHERE job_id=?",
                (token, now + lease_seconds, job["job_id"]),
            )
            connection.execute(
                "INSERT INTO monitoring_attempts(lease_token,job_id,started_at,outcome) VALUES (?, ?, ?, 'active')",
                (token, job["job_id"], now),
            )
            job.update(
                lease_token=token,
                lease_until=now + lease_seconds,
                attempts=job["attempts"] + 1,
                topic=record_topic(binding.destination),
            )
            return job

    @staticmethod
    def _owned(connection, job, now):
        return connection.execute(
            "SELECT m.*, j.destination, t.consumer_id FROM monitoring_delivery m "
            "JOIN delivery_jobs j USING(job_id) JOIN monitoring_targets t USING(destination) "
            "WHERE m.job_id=? AND m.lease_token=? AND m.lease_until>? AND m.state!='stored'",
            (job["job_id"], job["lease_token"], now),
        ).fetchone()

    @staticmethod
    def _end(connection, row, now, state, *, delay=0, error=None, response_json=None):
        connection.execute(
            "UPDATE monitoring_delivery SET state=?,next_attempt_at=?,lease_token=NULL,lease_until=NULL,last_error=? "
            "WHERE job_id=?",
            (state, now + delay, error, row["job_id"]),
        )
        connection.execute(
            "UPDATE monitoring_attempts SET finished_at=?,outcome=?,error=?,response_json=? WHERE lease_token=?",
            (now, state, error, response_json, row["lease_token"]),
        )

    def fail_monitoring(
        self, job, error, *, state="uncertain", response_json=None, now=None
    ):
        now = _now(now)
        if state not in {"uncertain", "quarantined", "blocked"}:
            raise ValueError("invalid failure state")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._owned(connection, job, now)
            if row is None:
                return False
            delay = min(60, 2 ** min(row["attempts"] - 1, 6))
            self._end(
                connection,
                row,
                now,
                state,
                delay=delay,
                error=error,
                response_json=response_json,
            )
            connection.execute(
                "UPDATE record_delivery_attempts SET outcome='uncertain',finished_at=?,error=? "
                "WHERE lease_token=? AND outcome='sending'",
                (now, error, job["lease_token"]),
            )
            connection.execute(
                "UPDATE delivery_jobs SET state='uncertain',last_error=? WHERE job_id=? "
                "AND state='sending' AND (SELECT acked_at FROM record_delivery_state WHERE job_id=?) IS NULL",
                (error, job["job_id"], job["job_id"]),
            )
            return True

    def begin_publish(self, job, *, now=None):
        now = _now(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._owned(connection, job, now)
            if (
                row is None
                or connection.execute(
                    "SELECT 1 FROM record_delivery_attempts WHERE lease_token=?",
                    (job["lease_token"],),
                ).fetchone()
            ):
                return False
            connection.execute(
                "UPDATE delivery_jobs SET attempts=attempts+1,last_error=NULL,state=CASE WHEN "
                "(SELECT acked_at FROM record_delivery_state WHERE job_id=?) IS NULL THEN 'sending' ELSE state END WHERE job_id=?",
                (job["job_id"], job["job_id"]),
            )
            attempt = connection.execute(
                "SELECT attempts FROM delivery_jobs WHERE job_id=?", (job["job_id"],)
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO record_delivery_attempts(lease_token,job_id,attempt,started_at,outcome) "
                "VALUES (?, ?, ?, ?, 'sending')",
                (job["lease_token"], job["job_id"], attempt, now),
            )
            return True

    def finish_publish(self, job, ack, *, ack_wait=10, now=None):
        now = _now(now)
        if not isinstance(ack, BrokerAck) or ack.job_id != job["job_id"]:
            raise MonitoringContractError("broker_ack_mismatch")
        if not math.isfinite(ack_wait) or ack_wait <= 0:
            raise ValueError("positive finite ACK wait is required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._owned(connection, job, now)
            if row is None:
                return False
            updated = connection.execute(
                "UPDATE record_delivery_attempts SET outcome='broker_acked',finished_at=? "
                "WHERE lease_token=? AND outcome='sending'",
                (now, job["lease_token"]),
            ).rowcount
            if not updated:
                return False
            connection.execute(
                "UPDATE record_delivery_state SET ack_kind='broker_puback',acked_at=COALESCE(acked_at,?) WHERE job_id=?",
                (now, job["job_id"]),
            )
            connection.execute(
                "UPDATE delivery_jobs SET state='submitted',last_error=NULL WHERE job_id=?",
                (job["job_id"],),
            )
            self._end(connection, row, now, "awaiting_ack", delay=ack_wait)
            return True

    def complete_monitoring(self, job, ack, *, now=None):
        now = _now(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._owned(connection, job, now)
            if row is None:
                return False
            binding = MonitoringBinding(
                self.origin_id, row["consumer_id"], row["destination"]
            )
            record = parse_record(row["wire_payload"])
            try:
                if record.payload_sha256 != row["payload_sha256"]:
                    raise MonitoringContractError("payload_hash_mismatch")
                validated = validate_ack(ack, binding, record)
            except MonitoringContractError as error:
                try:
                    raw_ack = json.dumps(ack, ensure_ascii=True, default=repr)
                except (TypeError, ValueError, RecursionError):
                    raw_ack = json.dumps({"invalid_ack_type": type(ack).__name__})
                self._end(
                    connection,
                    row,
                    now,
                    "uncertain",
                    delay=min(60, 2 ** min(row["attempts"] - 1, 6)),
                    error=error.code,
                    response_json=raw_ack,
                )
                return False
            ack_json = json.dumps(
                validated, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            connection.execute(
                "UPDATE monitoring_delivery SET ack_json=?,acked_at=? WHERE job_id=?",
                (ack_json, now, job["job_id"]),
            )
            self._end(connection, row, now, "stored", response_json=ack_json)
            return True

    def get_monitoring(self, job_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM monitoring_delivery WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)
