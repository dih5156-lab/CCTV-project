"""Small mixed-input measurements; local transport model, not a capacity gate."""

import json
import sqlite3
import time
from contextlib import closing, contextmanager

import pytest

from src.core.video_event_ingest import VideoEventIngest
from src.monitoring_contract import MonitoringBinding
from src.monitoring_delivery import MonitoringDeliveryStore
from src.monitoring_receiver import MonitoringReceiverStore
from src.monitoring_service import MonitoringService
from src.monitoring_worker import MonitoringDeliveryWorker
from src.sensor_record import SensorRecordStore, SensorRecordWorker
from tests.test_monitoring_delivery import DESTINATION
from tests.test_monitoring_outage_mqtt import add_pair, retry_busy
from tests.test_monitoring_service import wait_until
from tests.test_monitoring_worker import Broker, Reader


@pytest.mark.parametrize("interval", [1.0, 0.02])
def test_mixed_input_with_periodic_delivery(tmp_path, interval, record_property):
    source = MonitoringDeliveryStore(tmp_path / "source.db")
    binding = MonitoringBinding(source.origin_id, "load-consumer", DESTINATION)
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
    inputs = SensorRecordStore(source.path)
    video = VideoEventIngest(inputs, backend="test", topic_prefix="cctv/ai/events")
    service = MonitoringService(workers, interval=interval)
    retries, submission_seconds = [], []

    def all_stored():
        video.poll_recovery()
        retry_busy(SensorRecordWorker(inputs).drain_once, retries)
        with closing(sqlite3.connect(source.path)) as connection:
            return (
                connection.execute(
                    "SELECT count(*) FROM monitoring_delivery WHERE state='stored'"
                ).fetchone()[0]
                == 24
            )

    started = time.monotonic()
    service.start()
    try:
        for number in range(12):
            before = time.monotonic()
            add_pair(inputs, video, number, retries)
            submission_seconds.append(time.monotonic() - before)
            time.sleep(0.05)
        wait_until(all_stored, timeout=30)
        elapsed = time.monotonic() - started
    finally:
        service.close()
    with closing(sqlite3.connect(source.path)) as connection:
        assert (
            connection.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0]
            == 24
        )
        assert (
            connection.execute("SELECT count(*) FROM canonical_records").fetchone()[0]
            == 24
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM delivery_jobs WHERE state='held'"
            ).fetchone()[0]
            == 12
        )
    with closing(sqlite3.connect(receiver.path)) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_records").fetchone()[0]
            == 24
        )

    statements = []
    connect = source._connect

    @contextmanager
    def traced_connect():
        with connect() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    source._connect = traced_connect
    try:
        for _ in range(20):
            for worker in workers:
                assert worker.drain_once(10) == 0
    finally:
        source._connect = connect
    write_transactions = sum(statement == "BEGIN IMMEDIATE" for statement in statements)
    record_property(
        "contention_result",
        json.dumps(
            {
                "interval_seconds": interval,
                "input_records": 24,
                "stored_records": 24,
                "source_busy_retries": len(retries),
                "service_failed_cycles": service.status()["failed_cycles"],
                "elapsed_seconds": elapsed,
                "pair_submission_max_seconds": max(submission_seconds),
                "idle_worker_calls": 40,
                "idle_write_transactions": write_transactions,
                "transport": "in-process test model",
            },
            sort_keys=True,
        ),
    )
    assert write_transactions == 0
