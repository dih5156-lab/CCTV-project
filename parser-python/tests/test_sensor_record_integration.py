"""Direct parser input to common observations and the record-only delivery path."""

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.sensor_projection_worker import SensorProjectionWorker
from service.sensor_service import SensorService
from test_sensor_projection import counts
from test_sensor_projection import pg_db as pg_db
from test_tlv_mqtt_durable import receipts, sensor_service, wire_payload

from src.ingest_store import IngestStore
from src.protocols.record_mqtt import MqttRecordTransport
from src.record_delivery import RecordDeliveryStore, RecordDeliveryWorker
from src.sensor_record import SensorRecordStore, SensorRecordWorker
from tests.test_record_delivery_mqtt import broker as broker
from tests.test_record_delivery_mqtt import consumer as consumer
from tests.test_sensor_record import prepared, rows


def test_sensor_service_starts_and_closes_independent_record_worker(tmp_path):
    path = tmp_path / "common.db"
    prepared(path)
    with (
        patch("service.sensor_service.EventService"),
        patch("service.sensor_service.DeviceInfoService"),
        patch("database.sensor_projection_worker.SensorProjectionWorker"),
    ):
        service = SensorService(None, ingest_store=IngestStore(path))
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if len(rows(path, "canonical_records")) == 1:
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("SensorService did not run common record recovery")
        finally:
            service.close()
        assert not service._record_worker._thread.is_alive()


def test_actual_tlv_parser_preserves_decoded_measurements_without_replaying_outputs(
    tmp_path,
):
    path = tmp_path / "common.db"
    ingest = IngestStore(path)
    service = sensor_service(ingest)
    service.process_mqtt_message("app/dev/up", wire_payload())
    receipt = receipts(ingest)[0]
    original = list(service._edgex_forwarder.mock_calls)
    records = SensorRecordStore(path)
    SensorRecordWorker(records).drain_once()
    observation = json.loads(rows(path, "canonical_records")[0]["payload_json"])
    assert observation["source_record_id"] == receipt
    assert observation["measurements"]["temperature"]["value"] == 21.0
    assert observation["measurements"]["humidity"]["value"] == 80.0
    assert len(receipts(ingest)) == 1
    assert service._edgex_forwarder.mock_calls == original
    assert ingest.get(receipt)["stage"] == "ready"


def delivery_worker(path, broker, destination):
    return RecordDeliveryWorker(
        RecordDeliveryStore(path),
        MqttRecordTransport(broker, transport="unix"),
        namespace="tlv_mqtt",
        destination=destination,
    )


def test_pg_rejection_does_not_prevent_common_record_puback(
    broker, consumer, pg_db, tmp_path
):
    path = tmp_path / "common.db"
    ingest = IngestStore(path)
    service = sensor_service(ingest)
    service.process_mqtt_message("app/dev/up", wire_payload())
    receipt = receipts(ingest)[0]
    pg_db.execute_insert("ALTER TABLE t34954 ADD CHECK (temperature < 0)")
    SensorProjectionWorker(ingest, pg_db, service._recover_receipt).drain_once()
    assert ingest.get(receipt)["stage"] == "quarantined"
    assert counts(pg_db) == [0, 0, 0]
    SensorRecordWorker(SensorRecordStore(path)).drain_once()
    destination = "record-mqtt:cctv/records/v1/" + uuid4().hex
    worker = delivery_worker(path, broker, destination)
    assert worker.drain_once() == 1
    payload = consumer.get(timeout=10)[1]
    assert payload["observation_id"] == payload["record"]["observation_id"]
    assert "event_id" not in payload
    assert payload["record"]["source_record_id"] == receipt
    assert payload["record"]["measurements"]["temperature"]["value"] == 21.0
    assert rows(path, "record_delivery_state")[0]["ack_kind"] == "broker_puback"


@pytest.mark.parametrize("point", ["prepared", "canonical", "enqueued", "acknowledged"])
def test_process_exit_recovers_same_tlv_observation_and_pg(
    broker, consumer, pg_db, tmp_path, point
):
    path = tmp_path / "common.db"
    ingest = IngestStore(path)
    receipt = ingest.accept_mqtt("tlv_mqtt", "app/dev/up", wire_payload())
    destination = "record-mqtt:cctv/records/v1/" + uuid4().hex
    script = r"""
import os, sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root/'parser-python'))
sys.path.insert(0, str(root/'parser-python/tests'))
from test_tlv_mqtt_durable import sensor_service
from src.ingest_store import IngestStore
from src.sensor_record import SensorRecordStore, SensorRecordWorker
from src.record_delivery import RecordDeliveryStore, RecordDeliveryWorker
from src.protocols.record_mqtt import MqttRecordTransport
path, receipt, broker, destination, point = sys.argv[1:]
store = IngestStore(path)
sensor_service(store)._recover_receipt(store.get(receipt))
if point == 'prepared': os._exit(42)
SensorRecordWorker(SensorRecordStore(path)).drain_once()
if point == 'canonical': os._exit(42)
sender = MqttRecordTransport(broker, transport='unix')
delivery = RecordDeliveryStore(path)
delivery.enqueue_pending('tlv_mqtt', destination, sender.endpoint)
if point == 'enqueued': os._exit(42)
assert RecordDeliveryWorker(delivery, sender, namespace='tlv_mqtt', destination=destination).drain_once() == 1
os._exit(42)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(path), receipt, broker, destination, point],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 42, child.stderr.decode()
    records = SensorRecordStore(path)
    SensorRecordWorker(records).drain_once()
    SensorRecordWorker(records).drain_once()
    service = sensor_service(ingest)
    SensorProjectionWorker(ingest, pg_db, service._recover_receipt).drain_once()
    assert counts(pg_db) == [1, 1, 1]
    assert ingest.get(receipt)["stage"] == "complete"
    worker = delivery_worker(path, broker, destination)
    assert worker.drain_once() == (0 if point == "acknowledged" else 1)
    payload = consumer.get(timeout=15)[1]
    assert payload["record"]["source_record_id"] == receipt
    assert payload["source_namespace"] == "tlv_mqtt"
    assert len(rows(path, "canonical_records")) == 1
    assert len(rows(path, "ingest_revisions")) == 1
    assert len(rows(path, "delivery_jobs")) == 1
    assert receipts(ingest) == [receipt]
    assert not service._edgex_forwarder.mock_calls
    assert worker.drain_once() == 0
