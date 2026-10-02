"""Real SQLite capacity failures in temporary source and consumer databases."""

import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.monitoring_contract import MonitoringBinding, parse_record
from src.monitoring_delivery import MonitoringDeliveryStore
from src.monitoring_receiver import MonitoringReceiverStore
from src.monitoring_worker import MonitoringDeliveryWorker
from src.protocols.monitoring_local import LocalMqttReceiver
from tests.test_monitoring_delivery import BROKER, DESTINATION, READER
from tests.test_monitoring_receiver import BINDING, counts, payload, receiver
from tests.test_monitoring_worker import Broker, Reader

LARGE_VALUE = "x" * 65536
FULL_ERROR = "database or disk is full"


@contextmanager
def capped_store(store, monkeypatch, record_property):
    """Disallow page growth on this store only; never fill the host filesystem."""
    original_connect = store._connect
    with original_connect() as connection:
        page_count = connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    record_property("capacity_bytes", page_count * page_size)

    @contextmanager
    def connect():
        with original_connect() as connection:
            actual = connection.execute(
                f"PRAGMA max_page_count={page_count}"
            ).fetchone()[0]
            assert actual == page_count
            yield connection

    with monkeypatch.context() as scoped:
        scoped.setattr(store, "_connect", connect)
        yield
    assert store.path.stat().st_size <= page_count * page_size


def database_rows(path, tables):
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in tables
        }


@pytest.mark.parametrize("source", ["sensor", "video"])
def test_full_raw_intake_requires_explicit_resubmit(
    tmp_path, monkeypatch, record_property, source
):
    store = MonitoringDeliveryStore(tmp_path / "source.db")

    def accept(target, value):
        if source == "sensor":
            return target.accept_mqtt("tlv_mqtt", "sensors/demo", value.encode())
        return target.accept_event_json(
            "video_detection", {"event_id": "evt-1", "value": value}, destination=None
        )

    existing = accept(store, "keep")
    tables = ("ingest_receipts", "sensor_ingest_state", "event_ingest_state")
    before = database_rows(store.path, tables)
    with capped_store(store, monkeypatch, record_property):
        with pytest.raises(sqlite3.OperationalError, match=FULL_ERROR):
            accept(store, LARGE_VALUE)
        assert database_rows(store.path, tables) == before

    reopened = MonitoringDeliveryStore(store.path)
    retried = accept(reopened, LARGE_VALUE)
    assert retried != existing
    after = database_rows(store.path, tables)
    assert len(after["ingest_receipts"]) == 2
    state_table = "sensor_ingest_state" if source == "sensor" else "event_ingest_state"
    assert len(after[state_table]) == 2
    assert before["ingest_receipts"][0] in after["ingest_receipts"]


def test_full_normalization_preserves_committed_raw_for_retry(
    tmp_path, monkeypatch, record_property
):
    store = MonitoringDeliveryStore(tmp_path / "source.db")
    event = {"event_id": "evt-large", "value": LARGE_VALUE}
    receipt = store.accept_event_json("video_detection", event, destination=None)
    tables = (
        "ingest_receipts",
        "event_ingest_state",
        "canonical_records",
        "ingest_revisions",
        "delivery_jobs",
    )
    before = database_rows(store.path, tables)
    with capped_store(store, monkeypatch, record_property):
        with pytest.raises(sqlite3.OperationalError, match=FULL_ERROR):
            store.record_event(receipt, event, converter_version="capacity-test")
        assert database_rows(store.path, tables) == before
        assert store.get_event_receipt(receipt)["stage"] == "pending"

    reopened = MonitoringDeliveryStore(store.path)
    assert reopened.record_event(
        receipt, event, converter_version="capacity-test", recovery=True
    ) == ("accepted", None)
    after = database_rows(store.path, tables)
    assert after["ingest_receipts"] == before["ingest_receipts"]
    assert len(after["canonical_records"]) == len(after["ingest_revisions"]) == 1


def test_full_enqueue_rolls_back_new_job_and_preserves_existing_wire(
    tmp_path, monkeypatch, record_property
):
    store = MonitoringDeliveryStore(tmp_path / "source.db")
    binding = MonitoringBinding(store.origin_id, "capacity-consumer", DESTINATION)
    for record_id, value in (("evt-small", 0), ("evt-large", LARGE_VALUE)):
        event = {"event_id": record_id, "value": value}
        receipt = store.accept_event_json("video_detection", event, destination=None)
        store.record_event(receipt, event, converter_version="capacity-test")
        if record_id == "evt-small":
            assert (
                store.enqueue_monitoring("video_detection", binding, BROKER, READER)
                == 1
            )
    tables = (
        "ingest_receipts",
        "canonical_records",
        "delivery_jobs",
        "record_delivery_state",
        "monitoring_delivery",
        "monitoring_targets",
    )
    before = database_rows(store.path, tables)
    with capped_store(store, monkeypatch, record_property):
        with pytest.raises(sqlite3.OperationalError, match=FULL_ERROR):
            store.enqueue_monitoring("video_detection", binding, BROKER, READER)
        assert database_rows(store.path, tables) == before

    reopened = MonitoringDeliveryStore(store.path)
    assert reopened.enqueue_monitoring("video_detection", binding, BROKER, READER) == 1
    after = database_rows(store.path, tables)
    assert after["canonical_records"] == before["canonical_records"]
    assert len(after["delivery_jobs"]) == len(after["monitoring_delivery"]) == 2
    assert before["monitoring_delivery"][0] in after["monitoring_delivery"]
    with reopened._connect() as connection:
        wires = connection.execute(
            "SELECT wire_payload FROM monitoring_delivery"
        ).fetchall()
    assert len({parse_record(row[0]).delivery_id for row in wires}) == 2


def test_full_receiver_withholds_ack_then_replay_stores_once(
    tmp_path, monkeypatch, record_property
):
    store = receiver(tmp_path / "consumer.db")
    old_ack = store.accept(BINDING, payload())
    wire = payload("job-large", "evt-large", LARGE_VALUE)
    adapter = LocalMqttReceiver(store, BINDING, tmp_path / "unused.sock")
    client = Mock()
    client.ack.return_value = 0
    message = SimpleNamespace(
        topic=adapter.topic, qos=1, retain=False, mid=17, payload=wire
    )
    tables = ("monitoring_records", "monitoring_receipts", "monitoring_rejected")
    before = database_rows(store.path, tables)
    with capped_store(store, monkeypatch, record_property):
        with pytest.raises(sqlite3.OperationalError, match=FULL_ERROR):
            store.accept(BINDING, wire)
        adapter._on_message(client, None, message)
        client.ack.assert_not_called()
        client.disconnect.assert_called_once()
        assert adapter.status()["storage_or_transport_failures"] == 1
        assert store.get_delivery_status(BINDING, "job-large") is None
        assert database_rows(store.path, tables) == before

    reopened = receiver(store.path)
    adapter.store = reopened
    adapter._on_message(client, None, message)
    client.ack.assert_called_once_with(17, 1)
    ack = reopened.get_delivery_status(BINDING, "job-large")
    assert ack["payload_sha256"] == parse_record(wire).payload_sha256
    duplicate = reopened.accept(BINDING, wire)
    assert duplicate["receiver_record_id"] == ack["receiver_record_id"]
    assert duplicate["stored_at"] == ack["stored_at"]
    assert counts(store.path) == (2, 2, 0)
    assert (
        reopened.get_delivery_status(BINDING, "job-1")["receiver_record_id"]
        == old_ack["receiver_record_id"]
    )
    database_rows(store.path, tables)


def test_full_source_ack_save_recovers_by_lookup_without_republish(
    tmp_path, monkeypatch, record_property
):
    store = MonitoringDeliveryStore(tmp_path / "source.db")
    binding = MonitoringBinding(store.origin_id, "capacity-consumer", DESTINATION)
    consumer = MonitoringReceiverStore(
        tmp_path / "consumer.db", consumer_id=binding.consumer_id
    )
    consumer.register_binding(binding)
    total = 24
    for index in range(total):
        event = {"event_id": f"evt-{index}", "value": index}
        receipt = store.accept_event_json("video_detection", event, destination=None)
        store.record_event(receipt, event, converter_version="capacity-test")
    assert store.enqueue_monitoring("video_detection", binding, BROKER, READER) == total
    jobs_and_acks = []
    for _ in range(total):
        job = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
        assert job is not None
        jobs_and_acks.append((job, consumer.accept(binding, job["wire_payload"])))

    completed = 0
    with capped_store(store, monkeypatch, record_property):
        for job, ack in jobs_and_acks:
            # Ordinary ACKs eventually need another page; no fabricated large ACK.
            before = database_rows(
                store.path, ("monitoring_delivery", "monitoring_attempts")
            )
            try:
                assert store.complete_monitoring(job, ack, now=101)
            except sqlite3.OperationalError as error:
                assert str(error) == FULL_ERROR
                assert database_rows(store.path, tuple(before)) == before
                failed = store.get_monitoring(job["job_id"])
                assert failed["acked_at"] is None and failed["ack_json"] is None
                assert failed["lease_token"] == job["lease_token"]
                break
            completed += 1
        else:
            pytest.fail("The fixture did not reach the real SQLite capacity limit")
    record_property("acks_saved_before_full", completed)
    record_property("receiver_committed_records", total)

    reopened = MonitoringDeliveryStore(store.path)
    clock = [140.0]
    broker = Broker(consumer, binding)
    worker = MonitoringDeliveryWorker(
        reopened,
        broker,
        Reader(consumer, clock),
        namespace="video_detection",
        binding=binding,
        clock=lambda: clock[0],
    )
    assert worker.drain_once() == total - completed
    assert broker.sent == []
    assert counts(consumer.path) == (total, total, 0)
    for job, ack in jobs_and_acks:
        recovered = reopened.get_monitoring(job["job_id"])
        assert recovered["state"] == "stored"
        assert recovered["acked_at"] is not None
        assert recovered["wire_payload"] == job["wire_payload"]
        assert recovered["payload_sha256"] == ack["payload_sha256"]
    database_rows(reopened.path, ("monitoring_delivery",))
