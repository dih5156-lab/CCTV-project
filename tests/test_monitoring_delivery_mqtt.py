"""Opt-in real MQTT publish/PUBACK with a local consumer status reader."""

import json
import sqlite3
from queue import Queue
from threading import Event
from uuid import uuid4

import paho.mqtt.client as mqtt
import pytest

from src.monitoring_contract import MonitoringBinding
from src.monitoring_delivery import MonitoringDeliveryStore
from src.monitoring_receiver import MonitoringReceiverStore
from src.monitoring_worker import MonitoringDeliveryWorker
from src.protocols.record_mqtt import MqttRecordTransport
from src.sensor_record import SensorRecordStore, SensorRecordWorker
from tests.test_monitoring_delivery import seed
from tests.test_monitoring_worker import Reader
from tests.test_mqtt_durable_integration import broker as broker  # noqa: F401
from tests.test_sensor_record import prepared as sensor_prepared


@pytest.mark.parametrize("namespace", ["video_detection", "tlv_mqtt"])
def test_puback_then_late_consumer_receives_frozen_retry(broker, tmp_path, namespace):
    path = tmp_path / "source.db"
    store = MonitoringDeliveryStore(path)
    if namespace == "tlv_mqtt":
        receipt = sensor_prepared(path)
        SensorRecordWorker(SensorRecordStore(path)).drain_once()
    else:
        seed(path)
    binding = MonitoringBinding(
        store.origin_id, "consumer-1", "record-mqtt:cctv/records/v1/" + uuid4().hex
    )
    receiver = MonitoringReceiverStore(
        tmp_path / "consumer.db", consumer_id=binding.consumer_id
    )
    receiver.register_binding(binding)
    clock = [100.0]
    sender = MqttRecordTransport(broker, transport="unix")
    worker = MonitoringDeliveryWorker(
        store,
        sender,
        Reader(receiver, clock),
        namespace=namespace,
        binding=binding,
        clock=lambda: clock[0],
    )
    # No consumer subscription: PUBACK proves only broker receipt.
    assert worker.drain_once() == 0
    with sqlite3.connect(path) as connection:
        job_id, frozen = connection.execute(
            "SELECT job_id,wire_payload FROM monitoring_delivery"
        ).fetchone()
        assert (
            connection.execute("SELECT ack_kind FROM record_delivery_state").fetchone()[
                0
            ]
            == "broker_puback"
        ), store.get_monitoring(job_id)["last_error"]
    assert store.get_monitoring(job_id)["acked_at"] is None
    assert receiver.get_delivery_status(binding, job_id) is None

    ready, received = Event(), Queue()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="unix")
    client.on_connect = lambda c, u, f, r, p: c.subscribe(
        binding.destination.removeprefix("record-mqtt:"), qos=1
    )
    client.on_subscribe = lambda *args: ready.set()

    def accept(client, userdata, message):
        try:
            receiver.accept(binding, message.payload)
            received.put(message.payload)
        except Exception as error:
            received.put(error)

    client.on_message = accept
    client.connect(broker)
    client.loop_start()
    try:
        assert ready.wait(10)
        clock[0] += 10
        assert worker.drain_once() == 0
        assert received.get(timeout=10) == frozen
        clock[0] += 10
        assert worker.drain_once() == 1
        assert store.get_monitoring(job_id)["state"] == "stored"
        if namespace == "tlv_mqtt":
            record = json.loads(frozen)["record"]
            assert record["source_record_id"] == receipt
            assert record["measurements"]["temperature"]["value"] == 0
            assert record["measurements"]["humidity"]["quality"] == "missing"
        with sqlite3.connect(receiver.path) as connection:
            assert (
                connection.execute(
                    "SELECT count(*) FROM monitoring_records"
                ).fetchone()[0]
                == 1
            )
    finally:
        client.disconnect()
        client.loop_stop()
