"""Active shared-DB writers; video intake has no test-producer retry."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

from src.core.video_event_ingest import VideoEventIngest
from src.monitoring_contract import MonitoringBinding
from src.monitoring_delivery import MonitoringDeliveryStore
from src.monitoring_receiver import MonitoringReceiverStore
from src.monitoring_reconciliation import current_delivery, reconcile
from src.monitoring_service import MonitoringService
from src.monitoring_snapshot import MonitoringSnapshots
from src.monitoring_worker import MonitoringDeliveryWorker
from src.sensor_record import SensorRecordStore, SensorRecordWorker
from tests.test_monitoring_delivery import DESTINATION, seed
from tests.test_monitoring_outage_mqtt import deliveries, retry_busy
from tests.test_monitoring_service import wait_until
from tests.test_monitoring_worker import Broker, Reader


def test_video_and_sensor_intake_during_monitoring_writes(
    tmp_path, monkeypatch, record_property, caplog
):
    source = MonitoringDeliveryStore(tmp_path / "source.db")
    inputs = SensorRecordStore(source.path)
    video = VideoEventIngest(inputs, backend="test", topic_prefix="cctv/ai/events")
    sensor_worker = SensorRecordWorker(inputs)
    binding = MonitoringBinding(source.origin_id, "active-consumer", DESTINATION)
    receiver = MonitoringReceiverStore(
        tmp_path / "receiver.db", consumer_id=binding.consumer_id
    )
    receiver.register_binding(binding)
    sender, reader = Broker(receiver, binding), Reader(receiver, None)
    workers = [
        MonitoringDeliveryWorker(
            source,
            sender,
            reader,
            namespace=namespace,
            binding=binding,
            timeout=0.5,
            lease_seconds=4,
            ack_wait=0.02,
        )
        for namespace in ("tlv_mqtt", "video_detection")
    ]
    for number in range(8):
        seed(source.path, f"preload-{number}")

    producing, start = Event(), Event()
    monitor_writes, video_seconds, sensor_retries = [], [], []
    connect = source._connect

    def trace(statement):
        if producing.is_set() and statement.startswith(
            ("INSERT INTO monitoring_delivery", "UPDATE monitoring_delivery")
        ):
            monitor_writes.append(statement.split()[0])

    @contextmanager
    def monitored_connect():
        with connect() as connection:
            connection.set_trace_callback(trace)
            yield connection

    monkeypatch.setattr(source, "_connect", monitored_connect)

    def produce_video():
        assert start.wait(5)
        for number in range(20):
            before = time.monotonic()
            # An exhausted initial save fails this test; never hide it by resubmitting.
            reference = video.accept(
                {
                    "event_id": f"active-{number}",
                    "type": "fall",
                    "camera_id": "camera-1",
                    "occurred_at": number,
                },
                "camera-1",
            )
            video_seconds.append(time.monotonic() - before)
            video.defer(reference)
            time.sleep(0.005)

    def produce_sensor():
        assert start.wait(5)
        for number in range(20):
            # Explicit sensor producer redelivery, separate from video's bounded policy.
            receipt = retry_busy(
                lambda: inputs.accept("tlv_mqtt", "sensor-1", {"raw": number}),
                sensor_retries,
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
            retry_busy(lambda: inputs.prepare(receipt, projection), sensor_retries)
            retry_busy(sensor_worker.drain_once, sensor_retries)

    service = MonitoringService(workers, interval=0.02)
    started = time.monotonic()
    producing.set()
    service.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(produce_video), pool.submit(produce_sensor)]
            start.set()
            for future in pending:
                future.result(timeout=30)
        producing.clear()

        def all_stored():
            video.poll_recovery()
            return (
                sum(row["state"] == "stored" for row in deliveries(source).values())
                == 48
            )

        wait_until(all_stored, timeout=30)
    finally:
        producing.clear()
        service.close()

    final = deliveries(source)
    assert len(video_seconds) == 20 and monitor_writes
    assert len(final) == 48
    for delivery_id, row in final.items():
        ack = json.loads(row["ack_json"])
        assert row["state"] == "stored"
        assert ack["ack_kind"] == "consumer_committed"
        assert ack["delivery_id"] == delivery_id
        assert (
            ack["payload_sha256"]
            == row["payload_sha256"]
            == hashlib.sha256(row["wire_payload"]).hexdigest()
        )
    audit = MonitoringSnapshots(tmp_path / "snapshots.db")
    result = reconcile(
        audit,
        binding,
        audit.create_source(source, binding)["snapshot_id"],
        audit.create_consumer(receiver, binding)["snapshot_id"],
        lookup_current=lambda b, d: current_delivery(source, b, d),
    )
    assert result["all_inputs_accounted"] and result["counts"]["matched"] == 48
    with inputs._connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0]
            == 48
        )
        assert (
            connection.execute("SELECT count(*) FROM canonical_records").fetchone()[0]
            == 48
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM delivery_jobs WHERE state='held'"
            ).fetchone()[0]
            == 28
        )
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert deliveries(MonitoringDeliveryStore(source.path)) == final
    record_property(
        "active_contention_result",
        json.dumps(
            {
                "preloaded_video_records": 8,
                "offered_video_records": 20,
                "accepted_video_records": len(video_seconds),
                "sensor_records": 20,
                "video_test_producer_retries": 0,
                "video_busy_retry_logs": sum(
                    "Event intake busy; retrying" in record.message
                    for record in caplog.records
                ),
                "sensor_test_producer_retries": len(sensor_retries),
                "monitoring_write_statements_during_producers": len(monitor_writes),
                "stored_and_matched_records": len(final),
                "service_failed_cycles": service.status()["failed_cycles"],
                "elapsed_seconds": time.monotonic() - started,
                "video_intake_max_seconds": max(video_seconds),
                "transport": "in-process test model",
            },
            sort_keys=True,
        ),
    )
