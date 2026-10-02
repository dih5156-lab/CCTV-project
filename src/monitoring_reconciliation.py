"""Read-only comparison of frozen delivery sets; never repairs or deletes data."""

import json
import sqlite3

from .monitoring_contract import MonitoringContractError
from .monitoring_snapshot import check_source, source_item


def _same(left, right):
    return all(
        left.get(key) == right.get(key)
        for key in (
            "delivery_id",
            "source_namespace",
            "record_type",
            "record_id",
            "payload_sha256",
        )
    )


def current_delivery(store, binding, delivery_id):
    with store._connect() as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        check_source(connection, binding)
        row = connection.execute(
            "SELECT m.*,j.source_namespace,j.source_record_id,j.state AS broker_status, "
            "c.first_ingest_id,r.received_at,r.device_id FROM monitoring_delivery m "
            "JOIN delivery_jobs j USING(job_id) JOIN canonical_records c "
            "ON c.source_namespace=j.source_namespace AND c.source_record_id=j.source_record_id "
            "JOIN ingest_receipts r ON r.ingest_id=c.first_ingest_id WHERE j.destination=? AND j.job_id=?",
            (binding.destination, delivery_id),
        ).fetchone()
        return None if row is None else source_item(row)


def reconcile(
    snapshots, binding, source_id, consumer_id, *, lookup_current, detail_limit=100
):
    if type(detail_limit) is not int or not 0 <= detail_limit <= 1000:
        raise ValueError("detail_limit must be 0..1000")
    result = {
        "source_snapshot_id": source_id,
        "consumer_snapshot_id": consumer_id,
        "complete": False,
        "source_set_matches": False,
        "all_inputs_accounted": False,
        "sets_equal": False,
        "error": None,
        "counts": {
            key: 0
            for key in (
                "matched",
                "missing",
                "conflict",
                "unexpected",
                "out_of_snapshot",
                "unverifiable",
            )
        },
        "details": [],
        "details_truncated": 0,
    }

    def note(kind, delivery_id):
        result["counts"][kind] += 1
        if kind != "matched":
            if len(result["details"]) < detail_limit:
                result["details"].append({"delivery_id": delivery_id, "result": kind})
            else:
                result["details_truncated"] += 1

    try:
        with snapshots._connect() as connection:
            source = snapshots._metadata(connection, binding, source_id)
            consumer = snapshots._metadata(connection, binding, consumer_id)
            if (
                source["side"] != "source"
                or consumer["side"] != "consumer"
                or source["applied_filters"] != consumer["applied_filters"]
            ):
                raise MonitoringContractError("binding_mismatch")
            result.update(
                {
                    key: source[key]
                    for key in (
                        "pending_normalization",
                        "pending_delivery_creation",
                        "normalization_rejected",
                    )
                }
            )
            for row in connection.execute(
                "SELECT s.item_json,c.item_json FROM snapshot_items s LEFT JOIN snapshot_items c "
                "ON c.snapshot_id=? AND c.delivery_id=s.delivery_id WHERE s.snapshot_id=? ORDER BY s.ordinal",
                (consumer_id, source_id),
            ):
                left, right = json.loads(row[0]), json.loads(row[1]) if row[1] else None
                if left["validation_error"] or (right and right["validation_error"]):
                    kind = "unverifiable"
                elif right is None:
                    kind = "missing"
                else:
                    kind = "matched" if _same(left, right) else "conflict"
                note(kind, left["delivery_id"])
            for row in connection.execute(
                "SELECT c.item_json FROM snapshot_items c WHERE c.snapshot_id=? AND NOT EXISTS "
                "(SELECT 1 FROM snapshot_items s WHERE s.snapshot_id=? AND s.delivery_id=c.delivery_id) ORDER BY c.ordinal",
                (consumer_id, source_id),
            ):
                item = json.loads(row[0])
                try:
                    current = lookup_current(binding, item["delivery_id"])
                    if item["validation_error"] or (
                        current and current["validation_error"]
                    ):
                        kind = "unverifiable"
                    elif current is None:
                        kind = "unexpected"
                    else:
                        kind = "out_of_snapshot" if _same(current, item) else "conflict"
                except (OSError, sqlite3.Error, MonitoringContractError):
                    kind = "unverifiable"
                note(kind, item["delivery_id"])
            # A long comparison must not report success after either TTL expires.
            snapshots._metadata(connection, binding, source_id)
            snapshots._metadata(connection, binding, consumer_id)
        result["complete"] = result["counts"]["unverifiable"] == 0
        counts = result["counts"]
        result["source_set_matches"] = result["complete"] and not (
            counts["missing"] or counts["conflict"]
        )
        result["sets_equal"] = result["source_set_matches"] and not (
            counts["unexpected"] or counts["out_of_snapshot"]
        )
        result["all_inputs_accounted"] = result["sets_equal"] and not (
            source["pending_normalization"]
            or source["pending_delivery_creation"]
            or source["normalization_rejected"]
        )
    except (OSError, sqlite3.Error, MonitoringContractError) as error:
        result["error"] = (
            error.code
            if isinstance(error, MonitoringContractError)
            else type(error).__name__
        )
        result["counts"]["unverifiable"] += 1
    return result
