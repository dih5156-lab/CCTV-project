"""Durable, scoped manifests in a separate DB; source and consumer are read-only."""

import json
import math
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .monitoring_contract import MonitoringContractError, parse_record, require_string
from .monitoring_freshness import SensorFreshness

APPLICATION_ID = 1129598258


def utc_text(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def check_source(connection, binding):
    row = connection.execute(
        "SELECT o.origin_id,t.consumer_id FROM monitoring_origin o "
        "JOIN monitoring_targets t ON t.destination=? WHERE o.singleton=1",
        (binding.destination,),
    ).fetchone()
    if row is None or tuple(row) != (binding.origin_id, binding.consumer_id):
        raise MonitoringContractError("binding_mismatch")


def source_rows(connection, binding, namespace=None):
    return connection.execute(
        "SELECT m.*,j.source_namespace,j.source_record_id,j.state AS broker_status, "
        "j.attempts AS publish_attempts,c.first_ingest_id,r.received_at,r.device_id,b.acked_at AS broker_acked_at "
        "FROM monitoring_delivery m JOIN delivery_jobs j USING(job_id) "
        "JOIN canonical_records c ON c.source_namespace=j.source_namespace AND c.source_record_id=j.source_record_id "
        "JOIN ingest_receipts r ON r.ingest_id=c.first_ingest_id "
        "JOIN record_delivery_state b USING(job_id) WHERE j.destination=? "
        "AND (? IS NULL OR j.source_namespace=?) ORDER BY m.rowid",
        (binding.destination, namespace, namespace),
    )


def source_item(row):
    item = {
        "delivery_id": row["job_id"],
        "source_namespace": row["source_namespace"],
        "record_id": row["source_record_id"],
        "record_type": None,
        "payload_sha256": row["payload_sha256"],
        "ingest_id": row["first_ingest_id"],
        "received_at": row["received_at"],
        "device_id": row["device_id"],
        "broker_status": row["broker_status"],
        "monitoring_status": row["state"],
        "record": None,
        "validation_error": None,
    }
    try:
        record = parse_record(row["wire_payload"])
        if (
            record.delivery_id,
            record.source_namespace,
            record.record_id,
            record.payload_sha256,
        ) != (
            item["delivery_id"],
            item["source_namespace"],
            item["record_id"],
            item["payload_sha256"],
        ):
            raise MonitoringContractError("payload_hash_mismatch")
        item.update(
            record_type=record.record_type, record=json.loads(record.record_json)
        )
    except MonitoringContractError as error:
        item["validation_error"] = error.code
    return item


def pending_counts(connection, binding, namespace):
    pending = connection.execute(
        "SELECT count(*) FROM ingest_receipts r LEFT JOIN event_ingest_state e USING(ingest_id) "
        "LEFT JOIN sensor_ingest_state s USING(ingest_id) "
        "WHERE (? IS NULL OR r.source_namespace=?) AND "
        "(e.stage='pending' OR (e.ingest_id IS NULL AND (s.stage IN ('pending','ready','complete') OR s.ingest_id IS NULL)))",
        (namespace, namespace),
    ).fetchone()[0]
    creation = connection.execute(
        "SELECT count(*) FROM canonical_records c WHERE (? IS NULL OR c.source_namespace=?) "
        "AND NOT EXISTS (SELECT 1 FROM delivery_jobs j JOIN monitoring_delivery m USING(job_id) "
        "WHERE j.destination=? AND j.source_namespace=c.source_namespace AND j.source_record_id=c.source_record_id)",
        (namespace, namespace, binding.destination),
    ).fetchone()[0]
    rejected = connection.execute(
        "SELECT count(*) FROM ingest_receipts r LEFT JOIN event_ingest_state e USING(ingest_id) "
        "LEFT JOIN sensor_ingest_state s USING(ingest_id) WHERE (? IS NULL OR r.source_namespace=?) "
        "AND (e.stage IN ('invalid','conflict') OR (e.ingest_id IS NULL AND s.stage='quarantined'))",
        (namespace, namespace),
    ).fetchone()[0]
    return {
        "pending_normalization": pending,
        "pending_delivery_creation": creation,
        "normalization_rejected": rejected,
    }


class MonitoringSnapshots:
    def __init__(self, path, *, clock=time.time, freshness=None):
        if freshness is not None and not isinstance(freshness, SensorFreshness):
            raise ValueError("freshness must be a SensorFreshness policy")
        self.freshness = freshness if freshness is not None else SensorFreshness()
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            identity = (
                connection.execute("PRAGMA application_id").fetchone()[0],
                connection.execute("PRAGMA user_version").fetchone()[0],
            )
            if identity == (0, 0):
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table'"
                ).fetchone():
                    raise ValueError("Existing DB is not a monitoring snapshot store")
                connection.execute(
                    "CREATE TABLE snapshots(snapshot_id TEXT PRIMARY KEY, binding_json TEXT NOT NULL, metadata_json TEXT NOT NULL, expires_at REAL NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE snapshot_items(snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id), ordinal INTEGER NOT NULL, delivery_id TEXT NOT NULL, cursor TEXT UNIQUE NOT NULL, item_json TEXT NOT NULL, PRIMARY KEY(snapshot_id,ordinal), UNIQUE(snapshot_id,delivery_id))"
                )
                connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                connection.execute("PRAGMA user_version=1")
            elif identity != (APPLICATION_ID, 1):
                raise ValueError("Unsupported monitoring snapshot DB/version")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=0.25)
        try:
            connection.row_factory = sqlite3.Row
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise ValueError("Snapshot store requires DELETE journal mode")
            connection.execute("PRAGMA synchronous=EXTRA")
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _create(self, store, binding, side, namespace, ttl, max_items):
        if self.path.resolve() == Path(store.path).resolve():
            raise ValueError("Snapshot DB must be separate")
        if namespace is not None:
            require_string(namespace)
        now = self.clock()
        if not math.isfinite(now) or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("finite time and positive TTL required")
        if type(max_items) is not int or not 1 <= max_items <= 100000:
            raise ValueError("max_items must be 1..100000")
        metadata = {
            "snapshot_id": "snap_" + uuid4().hex,
            "side": side,
            "count": 0,
            "created_at": utc_text(now),
            "expires_at": utc_text(now + ttl),
            "applied_filters": {"source_namespace": namespace},
            "pending_normalization": None,
            "pending_delivery_creation": None,
            "normalization_rejected": None,
        }
        with self._connect() as output, store._connect() as source:
            output.execute("BEGIN IMMEDIATE")
            source.execute("PRAGMA query_only=ON")
            source.execute("BEGIN")
            if side == "source":
                check_source(source, binding)
                metadata.update(pending_counts(source, binding, namespace))
                items = (
                    source_item(row) for row in source_rows(source, binding, namespace)
                )
            else:
                store._check_binding(source, binding)
                items = self._consumer_items(source, binding, namespace)
            output.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?)",
                (
                    metadata["snapshot_id"],
                    json.dumps(binding.as_dict(), sort_keys=True),
                    "{}",
                    now + ttl,
                ),
            )
            for count, item in enumerate(items, 1):
                if count > max_items:
                    raise MonitoringContractError("snapshot_capacity")
                output.execute(
                    "INSERT INTO snapshot_items VALUES (?, ?, ?, ?, ?)",
                    (
                        metadata["snapshot_id"],
                        count,
                        item["delivery_id"],
                        uuid4().hex,
                        json.dumps(item, ensure_ascii=False, allow_nan=False),
                    ),
                )
                metadata["count"] = count
            output.execute(
                "UPDATE snapshots SET metadata_json=? WHERE snapshot_id=?",
                (json.dumps(metadata), metadata["snapshot_id"]),
            )
        return metadata

    @staticmethod
    def _consumer_items(connection, binding, namespace):
        rows = connection.execute(
            "SELECT d.*,r.source_namespace,r.record_type,r.record_id,r.record_json,r.stored_at "
            "FROM monitoring_receipts d JOIN monitoring_records r USING(receiver_record_id) "
            "WHERE d.destination=? AND r.origin_id=? AND (? IS NULL OR r.source_namespace=?) ORDER BY d.rowid",
            (binding.destination, binding.origin_id, namespace, namespace),
        )
        for row in rows:
            item = {
                key: row[key]
                for key in (
                    "delivery_id",
                    "source_namespace",
                    "record_type",
                    "record_id",
                    "payload_sha256",
                    "receiver_record_id",
                    "stored_at",
                    "received_at",
                )
            }
            item.update(record=json.loads(row["record_json"]), validation_error=None)
            try:
                parsed = parse_record(row["raw_payload"])
                if (
                    parsed.delivery_id,
                    parsed.source_namespace,
                    parsed.record_type,
                    parsed.record_id,
                    parsed.payload_sha256,
                    parsed.record_json,
                ) != (
                    row["delivery_id"],
                    row["source_namespace"],
                    row["record_type"],
                    row["record_id"],
                    row["payload_sha256"],
                    row["record_json"],
                ):
                    raise MonitoringContractError("payload_hash_mismatch")
            except MonitoringContractError as error:
                item["validation_error"] = error.code
            yield item

    def create_source(
        self, store, binding, *, namespace=None, ttl=300, max_items=100000
    ):
        return self._create(store, binding, "source", namespace, ttl, max_items)

    def create_consumer(
        self, store, binding, *, namespace=None, ttl=300, max_items=100000
    ):
        return self._create(store, binding, "consumer", namespace, ttl, max_items)

    def _metadata(self, connection, binding, snapshot_id):
        row = connection.execute(
            "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise MonitoringContractError("cursor_invalid")
        if json.loads(row["binding_json"]) != binding.as_dict():
            raise MonitoringContractError("binding_mismatch")
        now = self.clock()
        if not math.isfinite(now) or now >= row["expires_at"]:
            raise MonitoringContractError("snapshot_expired")
        return json.loads(row["metadata_json"])

    def list_snapshot(self, binding, snapshot_id, *, cursor=None, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        with self._connect() as connection:
            metadata = self._metadata(connection, binding, snapshot_id)
            ordinal = 0
            if cursor is not None:
                row = connection.execute(
                    "SELECT ordinal FROM snapshot_items WHERE snapshot_id=? AND cursor=?",
                    (snapshot_id, cursor),
                ).fetchone()
                if row is None:
                    raise MonitoringContractError("cursor_invalid")
                ordinal = row[0]
            rows = connection.execute(
                "SELECT cursor,item_json FROM snapshot_items WHERE snapshot_id=? AND ordinal>? ORDER BY ordinal LIMIT ?",
                (snapshot_id, ordinal, limit + 1),
            ).fetchall()
            return {
                **metadata,
                "items": self._present_items(
                    [json.loads(row["item_json"]) for row in rows[:limit]]
                ),
                "next_cursor": rows[limit - 1]["cursor"] if len(rows) > limit else None,
            }

    def get_records(self, binding, snapshot_id, keys):
        if not isinstance(keys, list) or not 1 <= len(keys) <= 100:
            raise ValueError("1..100 business keys required")
        fields = ("source_namespace", "record_type", "record_id")
        wanted = {
            tuple(require_string(key.get(field)) for field in fields) for key in keys
        }
        found, items = set(), []
        with self._connect() as connection:
            self._metadata(connection, binding, snapshot_id)
            for row in connection.execute(
                "SELECT item_json FROM snapshot_items WHERE snapshot_id=? ORDER BY ordinal",
                (snapshot_id,),
            ):
                item = json.loads(row[0])
                key = tuple(item[field] for field in fields)
                if key in wanted:
                    items.append(item)
                    found.add(key)
        return {
            "snapshot_id": snapshot_id,
            "items": self._present_items(items),
            "missing": [dict(zip(fields, key)) for key in sorted(wanted - found)],
        }

    def _present_items(self, items):
        now = self.clock()
        for item in items:
            item["freshness"] = self.freshness.describe(item, now=now)
        return items
