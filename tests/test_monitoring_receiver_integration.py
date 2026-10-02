"""Actual input records and child-process exit boundaries, without network/devices."""

import json
import multiprocessing
import os
import sqlite3
from contextlib import contextmanager

import pytest

from src.event_ingest_store import EventIngestStore
from src.ingest_store import IngestStore
from src.monitoring_contract import MonitoringBinding, parse_record, validate_ack
from src.monitoring_receiver import MonitoringReceiverStore
from src.record_delivery import RecordDeliveryStore
from src.sensor_record import SensorRecordStore, SensorRecordWorker

BINDING = MonitoringBinding(
    "source-lab", "consumer-lab", "record-mqtt:cctv/records/v1/consumer-lab"
)
ENDPOINT = "local-contract-test-no-network"


def _prepared_source(path):
    event_store = EventIngestStore(path)
    event = {"event_id": "evt-fall", "severity": "critical", "label": "낙상"}
    receipt = event_store.accept_event_json(
        "video_detection", event, destination="mqtt:cctv/ai/events"
    )
    _, alarm_job = event_store.record_event(
        receipt, event, converter_version="integration-v1", recovery=True
    )
    sensor = IngestStore(path)
    ingest_id = sensor.accept("tlv_mqtt", "sensor-1", {"temperature_c": 0})
    sensor.prepare(
        ingest_id,
        json.dumps(
            {
                "t34954": [{}],
                "sensor_data": [
                    {
                        "device_id": "sensor-1",
                        "object_id": "34954",
                        "payload_tlv": {"temperature_c": 0},
                    }
                ],
            }
        ),
    )
    SensorRecordWorker(SensorRecordStore(path)).drain_once()
    delivery = RecordDeliveryStore(path)
    for namespace in ("video_detection", "tlv_mqtt"):
        assert delivery.enqueue_pending(namespace, BINDING.destination, ENDPOINT) == 1
    return delivery, alarm_job


def test_actual_sensor_and_ai_envelopes_store_and_query_without_alarm_replay(tmp_path):
    source, alarm_job = _prepared_source(tmp_path / "source.db")
    consumer = MonitoringReceiverStore(
        tmp_path / "consumer.db", consumer_id=BINDING.consumer_id
    )
    consumer.register_binding(BINDING)
    seen = set()
    for _ in range(2):
        job = source.claim(BINDING.destination, ENDPOINT, lease_seconds=30)
        wire = job["wire_payload"].encode()
        record = parse_record(wire)
        ack = validate_ack(consumer.accept(BINDING, wire), BINDING, record)
        stored = consumer.get_record(BINDING, ack["receiver_record_id"])
        assert stored["record"] == json.loads(wire)["record"]
        seen.add(record.record_type)
        if record.record_type == "sensor_observation":
            measurements = stored["record"]["measurements"]
            assert measurements["temperature"]["value"] == 0
            assert measurements["humidity"]["value"] is None
            assert measurements["humidity"]["quality"] == "missing"
            assert stored["record"]["observed_at"] is None
        # A consumer ACK is not recorded as a broker ACK by this slice.
        with sqlite3.connect(source.path) as connection:
            assert (
                connection.execute(
                    "SELECT ack_kind FROM record_delivery_state WHERE job_id=?",
                    (job["job_id"],),
                ).fetchone()[0]
                is None
            )
    assert seen == {"event", "sensor_observation"}
    with sqlite3.connect(source.path) as connection:
        assert (
            connection.execute(
                "SELECT state FROM delivery_jobs WHERE job_id=?", (alarm_job,)
            ).fetchone()[0]
            == "held"
        )
    with sqlite3.connect(consumer.path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_records").fetchone()[0]
            == 2
        )


def _exit_during_accept(path, wire, boundary):
    store = MonitoringReceiverStore(path, consumer_id=BINDING.consumer_id)
    if boundary == "before_commit":
        original_connect = store._connect

        @contextmanager
        def exiting_connection():
            with original_connect() as connection:
                connection.create_function("test_exit", 0, lambda: os._exit(42))
                connection.execute(
                    "CREATE TEMP TRIGGER exit_before_commit AFTER INSERT ON monitoring_receipts "
                    "BEGIN SELECT test_exit(); END"
                )
                yield connection

        store._connect = exiting_connection
    else:
        store._ack = lambda *args: os._exit(42)
    store.accept(BINDING, wire)


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit_before_ack"])
def test_process_exit_then_query_or_same_id_retry(tmp_path, boundary):
    source, _ = _prepared_source(tmp_path / "source.db")
    job = source.claim(BINDING.destination, ENDPOINT, lease_seconds=30)
    wire = job["wire_payload"].encode()
    path = tmp_path / "consumer.db"
    consumer = MonitoringReceiverStore(path, consumer_id=BINDING.consumer_id)
    consumer.register_binding(BINDING)
    child = multiprocessing.get_context("fork").Process(
        target=_exit_during_accept, args=(path, wire, boundary)
    )
    child.start()
    child.join(timeout=10)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail("receiver child did not exit at the expected boundary")
    assert child.exitcode == 42
    reopened = MonitoringReceiverStore(path, consumer_id=BINDING.consumer_id)
    recovered = reopened.get_delivery_status(BINDING, job["job_id"])
    if boundary == "before_commit":
        assert recovered is None
    else:
        assert validate_ack(recovered, BINDING, parse_record(wire)) == recovered
    ack = reopened.accept(BINDING, wire)
    assert ack["status"] == ("stored" if boundary == "before_commit" else "duplicate")
    assert validate_ack(ack, BINDING, parse_record(wire)) == ack
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_records").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT count(*) FROM monitoring_receipts").fetchone()[0]
            == 1
        )
