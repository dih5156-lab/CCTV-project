"""Mixed backlog recovery with real SQLite locks and an isolated MQTT broker."""

import hashlib
import json
import sqlite3
import time
from contextlib import closing

import pytest

from src.core.video_event_ingest import VideoEventIngest
from src.monitoring_reconciliation import current_delivery, reconcile
from src.monitoring_service import build_local_service
from src.monitoring_snapshot import MonitoringSnapshots
from src.sensor_record import SensorRecordStore, SensorRecordWorker
from tests.test_monitoring_service import wait_until
from tests.test_monitoring_service_mqtt import make_service
from tests.test_mqtt_durable_integration import broker as broker  # noqa: F401


def retry_busy(operation, retries):
    """Test producer retries explicit failures, without counting them as accepted."""
    deadline = time.monotonic() + 10
    while True:
        try:
            return operation()
        except sqlite3.OperationalError as error:
            if str(error) != "database is locked" or time.monotonic() >= deadline:
                raise
            retries.append(str(error))
            time.sleep(0.02)


def add_pair(inputs, video, number, retries):
    reference = retry_busy(
        lambda: video.accept(
            {
                "event_id": f"outage-{number}",
                "type": "fall",
                "camera_id": "camera-1",
                "occurred_at": number,
            },
            "camera-1",
        ),
        retries,
    )
    video.defer(reference)
    video.poll_recovery()
    receipt = retry_busy(
        lambda: inputs.accept("tlv_mqtt", "sensor-1", {"raw": number}), retries
    )
    projection = json.dumps(
        {
            "sensor_data": [
                {
                    "device_id": "sensor-1",
                    "object_id": "34954",
                    "payload_tlv": {
                        "temperature_c": number,
                        "created_at": number,
                    },
                }
            ]
        }
    )
    retry_busy(lambda: inputs.prepare(receipt, projection), retries)
    retry_busy(SensorRecordWorker(inputs).drain_once, retries)


def deliveries(source):
    with closing(sqlite3.connect(source.path)) as connection:
        connection.row_factory = sqlite3.Row
        return {
            row["job_id"]: dict(row)
            for row in connection.execute("SELECT * FROM monitoring_delivery")
        }


@pytest.mark.parametrize(
    "lock_mode,restart", [("IMMEDIATE", False), ("EXCLUSIVE", True)]
)
def test_mixed_backlog_survives_receiver_lock(
    broker, tmp_path, lock_mode, restart, record_property
):
    service, source, receiver, binding, timings, _ = make_service(tmp_path, broker)
    inputs = SensorRecordStore(source.path)
    video = VideoEventIngest(inputs, backend="test", topic_prefix="cctv/ai/events")
    retries = []

    def backlog_ready():
        video.poll_recovery()
        retry_busy(SensorRecordWorker(inputs).drain_once, retries)
        return len(deliveries(source)) == 24

    service.start()
    lock = sqlite3.connect(receiver.path, timeout=1)
    try:
        lock.execute(f"BEGIN {lock_mode}")
        outage_started = time.monotonic()
        add_pair(inputs, video, 0, retries)
        if lock_mode == "IMMEDIATE":
            wait_until(
                lambda: service.receiver.status()["storage_or_transport_failures"] > 0,
                timeout=10,
            )
        else:
            wait_until(
                lambda: timings.report()["consumer_lookup"]["raised"] > 0,
                timeout=10,
            )
        for number in range(1, 12):
            add_pair(inputs, video, number, retries)
        wait_until(backlog_ready, timeout=15)
        before = deliveries(source)
        assert all(row["acked_at"] is None for row in before.values())
        assert all(row["ack_json"] is None for row in before.values())
        assert all(row["state"] != "stored" for row in before.values())
        assert service.status()["completed"] == 0
        assert (
            lock.execute("SELECT count(*) FROM monitoring_records").fetchone()[0] == 0
        )
        if restart:
            service.close()
        lock.rollback()
        recovery_started = time.monotonic()
        if restart:
            service, source, receiver, _, _, _ = build_local_service(
                source.path,
                receiver.path,
                binding.consumer_id,
                binding.destination,
                broker,
                namespaces=["tlv_mqtt", "video_detection"],
                interval=0.02,
                timeout=0.5,
                lease_seconds=4,
                ack_wait=0.02,
            )
            service.start()
        wait_until(lambda: service.status()["completed"] == 24, timeout=30)
        recovery_seconds = time.monotonic() - recovery_started
    finally:
        lock.rollback()
        lock.close()
        service.close()
    after = deliveries(source)
    assert before.keys() == after.keys()
    for delivery_id, original in before.items():
        final = after[delivery_id]
        assert final["state"] == "stored"
        assert final["wire_payload"] == original["wire_payload"]
        digest = hashlib.sha256(final["wire_payload"]).hexdigest()
        assert digest == original["payload_sha256"] == final["payload_sha256"]
        ack = json.loads(final["ack_json"])
        assert ack["delivery_id"] == delivery_id
        assert ack["payload_sha256"] == digest
        assert ack["ack_kind"] == "consumer_committed"
    audit = MonitoringSnapshots(tmp_path / "snapshots.db")
    result = reconcile(
        audit,
        binding,
        audit.create_source(source, binding)["snapshot_id"],
        audit.create_consumer(receiver, binding)["snapshot_id"],
        lookup_current=lambda b, d: current_delivery(source, b, d),
    )
    assert result["all_inputs_accounted"]
    assert result["counts"]["matched"] == 24
    with closing(sqlite3.connect(receiver.path)) as connection:
        assert dict(
            connection.execute(
                "SELECT source_namespace,count(*) FROM monitoring_records GROUP BY source_namespace"
            )
        ) == {"tlv_mqtt": 12, "video_detection": 12}
        assert (
            connection.execute("SELECT count(*) FROM monitoring_records").fetchone()[0]
            == 24
        )
        assert (
            connection.execute("SELECT count(*) FROM monitoring_receipts").fetchone()[0]
            == 24
        )
    with closing(sqlite3.connect(source.path)) as connection:
        assert (
            connection.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0]
            == 24
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM delivery_jobs WHERE destination='mqtt:cctv/ai/events' AND state='held'"
            ).fetchone()[0]
            == 12
        )
    record_property(
        "outage_result",
        json.dumps(
            {
                "lock_mode": lock_mode,
                "service_recreated": restart,
                "sensor_records": 12,
                "video_records": 12,
                "unconfirmed_before_recovery": len(before),
                "stored_after_recovery": len(after),
                "wire_bytes_and_hashes_unchanged": True,
                "false_storage_acks": 0,
                "duplicate_business_records": 0,
                "physical_jobs_held": 12,
                "source_busy_retries": len(retries),
                "outage_seconds": recovery_started - outage_started,
                "recovery_seconds": recovery_seconds,
                "reconciliation": result["counts"],
            },
            sort_keys=True,
        ),
    )


def test_video_normalization_lock_preserves_receipt_for_recovery(tmp_path):
    inputs = SensorRecordStore(tmp_path / "source.db")
    video = VideoEventIngest(inputs, backend="test", topic_prefix="cctv/ai/events")
    reference = video.accept(
        {
            "event_id": "source-lock",
            "type": "fall",
            "camera_id": "camera-1",
            "occurred_at": 0,
        },
        "camera-1",
    )
    video.defer(reference)
    original = inputs.get_event_receipt(reference.ingest_id)
    with closing(sqlite3.connect(inputs.path)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        video.poll_recovery()
        assert inputs.get_event_receipt(reference.ingest_id)["stage"] == "pending"
        assert lock.execute("SELECT count(*) FROM canonical_records").fetchone()[0] == 0
        lock.rollback()
    recovered = VideoEventIngest(
        SensorRecordStore(inputs.path), backend="test", topic_prefix="cctv/ai/events"
    )
    recovered.recover_pending()
    recovered.recover_pending()
    final = inputs.get_event_receipt(reference.ingest_id)
    assert final["stage"] == "accepted"
    assert final["raw_payload"] == original["raw_payload"]
    with closing(sqlite3.connect(inputs.path)) as connection:
        assert (
            connection.execute("SELECT count(*) FROM canonical_records").fetchone()[0]
            == 1
        )
        assert connection.execute("SELECT state FROM delivery_jobs").fetchall() == [
            ("held",)
        ]


def test_source_lock_rejects_intake_before_commit(tmp_path):
    inputs = SensorRecordStore(tmp_path / "source.db")
    with closing(sqlite3.connect(inputs.path)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            inputs.accept("tlv_mqtt", "sensor-1", {"raw": 1})
        assert lock.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0] == 0
        lock.rollback()
    receipt = inputs.accept("tlv_mqtt", "sensor-1", {"raw": 1})
    with closing(sqlite3.connect(inputs.path)) as connection:
        assert connection.execute(
            "SELECT ingest_id FROM ingest_receipts"
        ).fetchall() == [(receipt,)]
