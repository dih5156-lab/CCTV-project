"""ID/hash comparison distinguishes missing data from concurrent new deliveries."""

import importlib
import json
import sqlite3

import pytest

from src.event_ingest_store import EventIngestStore
from src.monitoring_receiver import MonitoringReceiverStore
from src.monitoring_worker import MonitoringDeliveryWorker
from src.sensor_record import SensorRecordStore, SensorRecordWorker
from tests.test_monitoring_delivery import BROKER, READER, prepared, seed
from tests.test_monitoring_snapshot import snapshots
from tests.test_monitoring_worker import Broker, Reader
from tests.test_sensor_record import prepared as sensor_prepared


def setup(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    receiver = MonitoringReceiverStore(
        tmp_path / "consumer.db", consumer_id=binding.consumer_id
    )
    receiver.register_binding(binding)
    with sqlite3.connect(store.path) as connection:
        wire = connection.execute(
            "SELECT wire_payload FROM monitoring_delivery"
        ).fetchone()[0]
    return store, binding, receiver, wire, snapshots(tmp_path)


def compare(audit, store, binding, source, consumer, **kwargs):
    module = importlib.import_module("src.monitoring_reconciliation")
    return module.reconcile(
        audit,
        binding,
        source["snapshot_id"],
        consumer["snapshot_id"],
        lookup_current=lambda b, d: module.current_delivery(store, b, d),
        **kwargs,
    )


def test_equal_counts_with_wrong_id_are_missing_and_unexpected(tmp_path):
    store, binding, receiver, wire, audit = setup(tmp_path)
    envelope = json.loads(wire)
    envelope["delivery_id"] = "unknown-delivery"
    receiver.accept(binding, json.dumps(envelope).encode())
    source, consumer = audit.create_source(store, binding), audit.create_consumer(
        receiver, binding
    )
    result = compare(audit, store, binding, source, consumer)
    assert result["complete"]
    assert not result["source_set_matches"]
    assert result["counts"]["missing"] == result["counts"]["unexpected"] == 1


def test_changed_body_is_conflict_and_normal_late_delivery_is_out_of_snapshot(tmp_path):
    store, binding, receiver, wire, audit = setup(tmp_path)
    source = audit.create_source(store, binding)
    envelope = json.loads(wire)
    envelope["record"]["value"] = 99
    receiver.accept(binding, json.dumps(envelope).encode())
    seed(store.path, "late")
    store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    with sqlite3.connect(store.path) as connection:
        late = connection.execute(
            "SELECT wire_payload FROM monitoring_delivery ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
    receiver.accept(binding, late)
    consumer = audit.create_consumer(receiver, binding)
    result = compare(audit, store, binding, source, consumer)
    assert result["counts"]["conflict"] == result["counts"]["out_of_snapshot"] == 1
    assert result["counts"]["unexpected"] == 0
    assert not result["sets_equal"]


def test_complete_matching_data_and_record_lookup(tmp_path):
    store, binding, receiver, wire, audit = setup(tmp_path)
    receiver.accept(binding, wire)
    source, consumer = audit.create_source(store, binding), audit.create_consumer(
        receiver, binding
    )
    result = compare(audit, store, binding, source, consumer)
    assert result["all_inputs_accounted"] and result["counts"]["matched"] == 1
    key = {
        "source_namespace": "video_detection",
        "record_type": "event",
        "record_id": "evt-1",
    }
    records = audit.get_records(
        binding, source["snapshot_id"], [key, {**key, "record_id": "absent"}]
    )
    assert records["items"][0]["record"]["value"] == 0
    assert records["missing"] == [{**key, "record_id": "absent"}]


@pytest.mark.parametrize("failure", ["expired", "lookup", "database"])
def test_incomplete_check_never_reports_success(tmp_path, failure):
    store, binding, receiver, wire, audit = setup(tmp_path)
    source = audit.create_source(store, binding)
    receiver.accept(binding, wire)
    consumer = audit.create_consumer(receiver, binding)
    module = importlib.import_module("src.monitoring_reconciliation")
    if failure == "expired":
        audit.clock = lambda: 400
    elif failure == "database":
        with audit._connect() as connection:
            connection.execute("DROP TABLE snapshot_items")
    else:
        # A new consumer ID absent from the original source manifest needs a live lookup.
        seed(store.path, "late")
        store.enqueue_monitoring("video_detection", binding, BROKER, READER)
        with sqlite3.connect(store.path) as connection:
            receiver.accept(
                binding,
                connection.execute(
                    "SELECT wire_payload FROM monitoring_delivery ORDER BY rowid DESC LIMIT 1"
                ).fetchone()[0],
            )
        consumer = audit.create_consumer(receiver, binding)

    def failed_lookup(*args):
        raise TimeoutError()

    result = module.reconcile(
        audit,
        binding,
        source["snapshot_id"],
        consumer["snapshot_id"],
        lookup_current=failed_lookup,
    )
    assert not result["complete"]
    assert not result["source_set_matches"] and not result["all_inputs_accounted"]
    assert result["counts"]["unverifiable"]


def test_sensor_video_worker_recovery_then_snapshots_match(tmp_path):
    store, binding, receiver, wire, audit = setup(tmp_path)
    receipt = sensor_prepared(store.path)
    SensorRecordWorker(SensorRecordStore(store.path)).drain_once()
    clock = [100.0]
    broker, reader = Broker(receiver, binding), Reader(receiver, clock)
    workers = [
        MonitoringDeliveryWorker(
            store,
            broker,
            reader,
            namespace=namespace,
            binding=binding,
            clock=lambda: clock[0],
        )
        for namespace in ("video_detection", "tlv_mqtt")
    ]
    for worker in workers:
        worker.drain_once()
    clock[0] = 110
    workers[0].drain_once()
    source, consumer = audit.create_source(store, binding), audit.create_consumer(
        receiver, binding
    )
    result = compare(audit, store, binding, source, consumer)
    assert result["all_inputs_accounted"]
    assert result["counts"]["matched"] == 2
    sensor = next(
        item
        for item in audit.list_snapshot(binding, source["snapshot_id"])["items"]
        if item["record_type"] == "sensor_observation"
    )
    assert sensor["ingest_id"] == receipt
    assert sensor["record"]["measurements"]["temperature"]["value"] == 0
    assert sensor["record"]["measurements"]["humidity"]["quality"] == "missing"
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                "SELECT state FROM delivery_jobs WHERE destination='mqtt:cctv/ai/events'"
            ).fetchone()[0]
            == "held"
        )


def test_rejected_input_is_not_hidden_by_matching_delivery_sets(tmp_path):
    store, binding, receiver, wire, audit = setup(tmp_path)
    ingest = EventIngestStore(store.path)
    receipt = ingest.accept_event_json(
        "video_detection", {"bad": True}, destination=None
    )
    ingest.reject_event(receipt, [{"code": "invalid"}], converter_version="test")
    receiver.accept(binding, wire)
    source, consumer = audit.create_source(store, binding), audit.create_consumer(
        receiver, binding
    )
    result = compare(audit, store, binding, source, consumer)
    assert result["sets_equal"]
    assert result["normalization_rejected"] == 1
    assert not result["all_inputs_accounted"]


def test_corrupt_source_hash_is_unverifiable_not_matched(tmp_path):
    store, binding, receiver, wire, audit = setup(tmp_path)
    receiver.accept(binding, wire)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE monitoring_delivery SET payload_sha256=?", ("0" * 64,)
        )
    source, consumer = audit.create_source(store, binding), audit.create_consumer(
        receiver, binding
    )
    result = compare(audit, store, binding, source, consumer)
    assert not result["complete"]
    assert result["counts"]["unverifiable"] == 1


def test_normal_late_record_is_not_unexpected_and_requires_new_round(tmp_path):
    store, binding, receiver, wire, audit = setup(tmp_path)
    receiver.accept(binding, wire)
    source = audit.create_source(store, binding)
    seed(store.path, "late")
    store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    with sqlite3.connect(store.path) as connection:
        receiver.accept(
            binding,
            connection.execute(
                "SELECT wire_payload FROM monitoring_delivery ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[0],
        )
    consumer = audit.create_consumer(receiver, binding)
    result = compare(audit, store, binding, source, consumer)
    assert result["source_set_matches"] and not result["sets_equal"]
    assert result["counts"]["out_of_snapshot"] == 1
    assert result["counts"]["unexpected"] == 0
    assert compare(
        audit, store, binding, audit.create_source(store, binding), consumer
    )["all_inputs_accounted"]
