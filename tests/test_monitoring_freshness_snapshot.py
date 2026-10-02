"""Freshness is a read-time view over durable, unchanged sensor records."""

import json

import pytest

from src.monitoring_contract import MonitoringBinding
from src.monitoring_delivery import MonitoringDeliveryStore
from src.monitoring_freshness import SensorFreshness
from src.monitoring_receiver import MonitoringReceiverStore
from src.monitoring_snapshot import MonitoringSnapshots, utc_text
from src.sensor_record import SensorRecordStore
from tests.test_monitoring_delivery import BROKER, DESTINATION, READER
from tests.test_sensor_record import prepared as sensor_prepared


def sensor_flow(tmp_path, monkeypatch, *, arrival=120):
    receipt = sensor_prepared(
        tmp_path / "source.db", data={"created_at": 100000, "temperature_c": 0}
    )
    store = MonitoringDeliveryStore(tmp_path / "source.db")
    with store._connect() as connection:
        connection.execute(
            "UPDATE ingest_receipts SET received_at=? WHERE ingest_id=?",
            (utc_text(110), receipt),
        )
    SensorRecordStore(store.path).project_sensor(receipt)
    binding = MonitoringBinding(store.origin_id, "consumer-test", DESTINATION)
    store.enqueue_monitoring("tlv_mqtt", binding, BROKER, READER)
    job = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=arrival)
    receiver = MonitoringReceiverStore(
        tmp_path / "consumer.db", consumer_id=binding.consumer_id
    )
    receiver.register_binding(binding)
    monkeypatch.setattr("src.monitoring_receiver._utc_now", lambda: utc_text(arrival))
    ack = receiver.accept(binding, job["wire_payload"])
    assert store.complete_monitoring(job, ack, now=arrival + 1)
    return store, receiver, binding, job


@pytest.mark.parametrize("side", ["source", "consumer"])
def test_same_snapshot_ages_on_read_and_reopen_without_changing_databases(
    tmp_path, monkeypatch, side
):
    store, receiver, binding, job = sensor_flow(tmp_path, monkeypatch)
    clock = [120]
    policy = SensorFreshness(60, clock_sync="confirmed")
    audit = MonitoringSnapshots(
        tmp_path / "snapshots.db", clock=lambda: clock[0], freshness=policy
    )
    snapshot = (
        audit.create_source(store, binding)
        if side == "source"
        else audit.create_consumer(receiver, binding)
    )
    paths = (store.path, receiver.path, audit.path)
    before = [path.read_bytes() for path in paths]
    first = audit.list_snapshot(binding, snapshot["snapshot_id"])["items"][0]
    assert first["freshness"]["status"] == "fresh"
    clock[0] = 161
    reopened = MonitoringSnapshots(audit.path, clock=lambda: clock[0], freshness=policy)
    key = {
        name: first[name] for name in ("source_namespace", "record_type", "record_id")
    }
    later = reopened.get_records(binding, snapshot["snapshot_id"], [key])["items"][0]
    assert later["freshness"]["status"] == "stale"
    assert later["freshness"]["on_arrival"]["status"] == "fresh"
    assert later["record"] == first["record"]
    assert later["payload_sha256"] == job["payload_sha256"]
    assert [path.read_bytes() for path in paths] == before
    with audit._connect() as connection:
        saved = json.loads(
            connection.execute("SELECT item_json FROM snapshot_items").fetchone()[0]
        )
    assert "freshness" not in saved


def test_stored_ack_and_late_arrival_are_independent(tmp_path, monkeypatch):
    store, receiver, binding, job = sensor_flow(tmp_path, monkeypatch, arrival=200)
    audit = MonitoringSnapshots(
        tmp_path / "snapshots.db",
        clock=lambda: 210,
        freshness=SensorFreshness(60, clock_sync="confirmed"),
    )
    source = audit.create_source(store, binding)
    consumer = audit.create_consumer(receiver, binding)
    first = audit.list_snapshot(binding, source["snapshot_id"])["items"][0]
    last = audit.list_snapshot(binding, consumer["snapshot_id"])["items"][0]
    assert first["monitoring_status"] == "stored"
    assert first["freshness"]["status"] == last["freshness"]["status"] == "stale"
    assert first["freshness"]["on_arrival"]["status"] == "fresh"
    assert last["freshness"]["on_arrival"]["status"] == "stale"
    assert last["record"]["measurements"]["temperature"]["value"] == 0
    assert last["record"]["measurements"]["humidity"]["quality"] == "missing"
    assert store.get_monitoring(job["job_id"])["state"] == "stored"


def test_late_old_record_is_retained_and_marked_stale_after_newer_arrival(
    tmp_path, monkeypatch
):
    store, receiver, binding, job = sensor_flow(tmp_path, monkeypatch)
    base = json.loads(job["wire_payload"])
    for label, observed in (("new", 180), ("late-old", 90)):
        wire = json.loads(json.dumps(base))
        wire["delivery_id"] = label
        wire["observation_id"] = wire["record"]["observation_id"] = label
        wire["record"]["observed_at"] = utc_text(observed)
        monkeypatch.setattr("src.monitoring_receiver._utc_now", lambda: utc_text(200))
        receiver.accept(binding, json.dumps(wire).encode())
    audit = MonitoringSnapshots(
        tmp_path / "snapshots.db",
        clock=lambda: 210,
        freshness=SensorFreshness(60, clock_sync="confirmed"),
    )
    snapshot = audit.create_consumer(receiver, binding)
    page = audit.list_snapshot(binding, snapshot["snapshot_id"], limit=2)
    tail = audit.list_snapshot(
        binding, snapshot["snapshot_id"], cursor=page["next_cursor"], limit=2
    )
    values = {row["record_id"]: row for row in page["items"] + tail["items"]}
    assert len(values) == 3
    assert values["new"]["freshness"]["status"] == "fresh"
    assert values["late-old"]["freshness"]["status"] == "stale"
    assert values["late-old"]["freshness"]["on_arrival"]["status"] == "stale"
    assert values["new"]["record"]["observed_at"] == utc_text(180)


def test_default_policy_does_not_assume_threshold_or_clock_sync(tmp_path, monkeypatch):
    store, receiver, binding, job = sensor_flow(tmp_path, monkeypatch)
    audit = MonitoringSnapshots(tmp_path / "snapshots.db", clock=lambda: 120)
    snapshot = audit.create_consumer(receiver, binding)
    row = audit.list_snapshot(binding, snapshot["snapshot_id"])["items"][0]
    assert row["freshness"]["status"] == "unknown"
    assert row["freshness"]["age_seconds"] is None
    assert row["freshness"]["stale_after_seconds"] is None


def test_corrupt_wire_cannot_be_presented_as_fresh(tmp_path, monkeypatch):
    store, receiver, binding, job = sensor_flow(tmp_path, monkeypatch)
    with store._connect() as connection:
        connection.execute(
            "UPDATE monitoring_delivery SET payload_sha256=?", ("0" * 64,)
        )
    audit = MonitoringSnapshots(
        tmp_path / "snapshots.db",
        clock=lambda: 120,
        freshness=SensorFreshness(60, clock_sync="confirmed"),
    )
    snapshot = audit.create_source(store, binding)
    row = audit.list_snapshot(binding, snapshot["snapshot_id"])["items"][0]
    assert row["freshness"]["status"] == "unknown"
    assert row["freshness"]["reason"] == "unverifiable_record"
