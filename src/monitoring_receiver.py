"""Opt-in local monitoring consumer; commit records before storage ACK.

This store has its own database. It never opens device/alarm execution paths.
Authentication and network transport belong to the caller's adapter.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .monitoring_contract import (
    MonitoringBinding,
    MonitoringContractError,
    parse_record,
    require_string,
)

APPLICATION_ID = 1129598257


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class MonitoringReceiverStore:
    def __init__(self, path: str | Path, *, consumer_id: str):
        self.consumer_id = require_string(consumer_id, "binding_mismatch")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            application = connection.execute("PRAGMA application_id").fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if application == 0 and version == 0:
                if connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchone():
                    raise ValueError("Existing database is not a monitoring receiver")
                for statement in (
                    """CREATE TABLE monitoring_identity (
                        singleton INTEGER PRIMARY KEY CHECK(singleton=1), consumer_id TEXT NOT NULL)""",
                    """CREATE TABLE monitoring_bindings (
                        destination TEXT PRIMARY KEY NOT NULL, origin_id TEXT NOT NULL)""",
                    """CREATE TABLE monitoring_records (
                        receiver_record_id TEXT PRIMARY KEY NOT NULL,
                        origin_id TEXT NOT NULL, source_namespace TEXT NOT NULL,
                        record_type TEXT NOT NULL, record_id TEXT NOT NULL,
                        record_json TEXT NOT NULL, stored_at TEXT NOT NULL,
                        UNIQUE(origin_id, source_namespace, record_type, record_id))""",
                    """CREATE TABLE monitoring_receipts (
                        destination TEXT NOT NULL REFERENCES monitoring_bindings(destination),
                        delivery_id TEXT NOT NULL,
                        receiver_record_id TEXT NOT NULL REFERENCES monitoring_records(receiver_record_id),
                        raw_payload BLOB NOT NULL, payload_sha256 TEXT NOT NULL,
                        received_at TEXT NOT NULL, PRIMARY KEY(destination, delivery_id))""",
                    """CREATE TABLE monitoring_rejected (
                        rejection_id TEXT PRIMARY KEY NOT NULL,
                        origin_id TEXT NOT NULL, consumer_id TEXT NOT NULL, destination TEXT NOT NULL,
                        received_at TEXT NOT NULL, raw_payload BLOB NOT NULL, error_code TEXT NOT NULL)""",
                    "CREATE INDEX monitoring_receipt_record ON monitoring_receipts(receiver_record_id, destination)",
                ):
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO monitoring_identity VALUES (1, ?)", (self.consumer_id,)
                )
                connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                connection.execute("PRAGMA user_version=1")
            elif (application, version) != (APPLICATION_ID, 1):
                raise ValueError("Unsupported monitoring receiver database/version")
            self._check_consumer(connection, self.consumer_id)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=0.25)
        try:
            connection.row_factory = sqlite3.Row
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise ValueError("Monitoring receiver requires DELETE journal mode")
            connection.execute("PRAGMA synchronous=EXTRA")
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _check_consumer(self, connection, consumer_id):
        row = connection.execute(
            "SELECT consumer_id FROM monitoring_identity WHERE singleton=1"
        ).fetchone()
        if row is None or row[0] != consumer_id or consumer_id != self.consumer_id:
            raise MonitoringContractError("binding_mismatch")

    def _check_binding(self, connection, binding):
        self._check_consumer(connection, binding.consumer_id)
        row = connection.execute(
            "SELECT origin_id FROM monitoring_bindings WHERE destination=?",
            (binding.destination,),
        ).fetchone()
        if row is None or row[0] != binding.origin_id:
            raise MonitoringContractError("binding_mismatch")

    def register_binding(self, binding: MonitoringBinding):
        """Explicit local registration; inbound payloads cannot register routes."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_consumer(connection, binding.consumer_id)
            connection.execute(
                "INSERT OR IGNORE INTO monitoring_bindings VALUES (?, ?)",
                (binding.destination, binding.origin_id),
            )
            self._check_binding(connection, binding)

    @staticmethod
    def _ack(binding, row, status):
        return {
            "ack_schema_version": "1.0",
            "ack_kind": "consumer_committed",
            **binding.as_dict(),
            **{
                key: row[key]
                for key in (
                    "delivery_id",
                    "source_namespace",
                    "record_type",
                    "record_id",
                    "payload_sha256",
                    "receiver_record_id",
                    "stored_at",
                )
            },
            "status": status,
            "ack_emitted_at": _utc_now(),
        }

    @staticmethod
    def _receipt(connection, binding, delivery_id):
        return connection.execute(
            "SELECT d.*, r.source_namespace, r.record_type, r.record_id, r.stored_at "
            "FROM monitoring_receipts d JOIN monitoring_records r USING(receiver_record_id) "
            "WHERE d.destination=? AND d.delivery_id=? AND r.origin_id=?",
            (binding.destination, delivery_id, binding.origin_id),
        ).fetchone()

    def accept(self, binding: MonitoringBinding, wire: bytes) -> dict:
        if not isinstance(wire, bytes):
            raise TypeError("Monitoring payload must be bytes")
        received_at = _utc_now()
        try:
            record = parse_record(wire)
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._check_binding(connection, binding)
                receipt = self._receipt(connection, binding, record.delivery_id)
                existing = connection.execute(
                    "SELECT * FROM monitoring_records WHERE origin_id=? AND source_namespace=? "
                    "AND record_type=? AND record_id=?",
                    (
                        binding.origin_id,
                        record.source_namespace,
                        record.record_type,
                        record.record_id,
                    ),
                ).fetchone()
                if receipt and receipt["payload_sha256"] != record.payload_sha256:
                    raise MonitoringContractError("delivery_conflict")
                if existing and existing["record_json"] != record.record_json:
                    raise MonitoringContractError("record_conflict")
                status = "duplicate" if existing else "stored"
                receiver_id = (
                    existing["receiver_record_id"] if existing else "rec_" + uuid4().hex
                )
                if existing is None:
                    connection.execute(
                        "INSERT INTO monitoring_records VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            receiver_id,
                            binding.origin_id,
                            record.source_namespace,
                            record.record_type,
                            record.record_id,
                            record.record_json,
                            _utc_now(),
                        ),
                    )
                if receipt is None:
                    connection.execute(
                        "INSERT INTO monitoring_receipts VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            binding.destination,
                            record.delivery_id,
                            receiver_id,
                            wire,
                            record.payload_sha256,
                            received_at,
                        ),
                    )
                row = self._receipt(connection, binding, record.delivery_id)
        except MonitoringContractError as error:
            # The rejected bytes are retained in a separate successful transaction.
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO monitoring_rejected VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        "rej_" + uuid4().hex,
                        binding.origin_id,
                        binding.consumer_id,
                        binding.destination,
                        received_at,
                        wire,
                        error.code,
                    ),
                )
            raise
        # Returning from the transaction is essential: commit may itself fail.
        return self._ack(binding, row, status)

    def get_delivery_status(
        self, binding: MonitoringBinding, delivery_id: str
    ) -> dict | None:
        require_string(delivery_id)
        with self._connect() as connection:
            self._check_binding(connection, binding)
            row = self._receipt(connection, binding, delivery_id)
        return None if row is None else self._ack(binding, row, "duplicate")

    def get_record(
        self, binding: MonitoringBinding, receiver_record_id: str
    ) -> dict | None:
        require_string(receiver_record_id)
        with self._connect() as connection:
            self._check_binding(connection, binding)
            row = connection.execute(
                "SELECT r.* FROM monitoring_records r WHERE r.receiver_record_id=? AND r.origin_id=? "
                "AND EXISTS (SELECT 1 FROM monitoring_receipts d "
                "WHERE d.receiver_record_id=r.receiver_record_id AND d.destination=?)",
                (receiver_record_id, binding.origin_id, binding.destination),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["record"] = json.loads(result.pop("record_json"))
        return result
