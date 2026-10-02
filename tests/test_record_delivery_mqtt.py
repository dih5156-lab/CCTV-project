"""Opt-in isolated Mosquitto tests; no production endpoint or device outputs."""

import importlib
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Event
from uuid import uuid4

import paho.mqtt.client as mqtt
import pytest

from tests.test_mqtt_durable_integration import broker as broker  # noqa: F401
from tests.test_record_delivery import rows, seed


@pytest.fixture
def consumer(broker):
    ready, messages = Event(), Queue()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="unix")
    client.on_connect = lambda c, u, f, r, p: c.subscribe("cctv/records/v1/#", qos=1)
    client.on_subscribe = lambda *args: ready.set()
    client.on_message = lambda c, u, message: messages.put(
        (message.topic, json.loads(message.payload))
    )
    client.connect(broker)
    client.loop_start()
    try:
        assert ready.wait(10)
        yield messages
    finally:
        client.disconnect()
        client.loop_stop()


def make_worker(path, broker, destination):
    module = importlib.import_module("src.record_delivery")
    transport = importlib.import_module(
        "src.protocols.record_mqtt"
    ).MqttRecordTransport(broker, transport="unix")
    store = module.RecordDeliveryStore(path)
    worker = module.RecordDeliveryWorker(
        store, transport, namespace="video_detection", destination=destination
    )
    return store, worker


def test_real_puback_records_receipt_without_releasing_alarm(
    broker, consumer, tmp_path
):
    path = tmp_path / "common.db"
    _, alarm_job = seed(path)
    destination = "record-mqtt:cctv/records/v1/" + uuid4().hex
    store, worker = make_worker(path, broker, destination)
    assert worker.drain_once() == 1
    topic, payload = consumer.get(timeout=10)
    assert topic == destination.removeprefix("record-mqtt:")
    assert payload["source_namespace"] == "video_detection"
    assert payload["event_id"] == "evt-1"
    state = rows(path, "record_delivery_state")[0]
    assert state["job_id"] == payload["delivery_id"]
    assert state["ack_kind"] == "broker_puback"
    assert state["acked_at"] is not None
    assert (
        next(j for j in rows(path, "delivery_jobs") if j["job_id"] == alarm_job)[
            "state"
        ]
        == "held"
    )
    assert worker.drain_once() == 0


def test_disconnect_before_puback_remains_retryable(
    broker, consumer, tmp_path, monkeypatch
):
    path = tmp_path / "common.db"
    seed(path)
    destination = "record-mqtt:cctv/records/v1/" + uuid4().hex
    store, worker = make_worker(path, broker, destination)
    original = mqtt.Client._handle_pubackcomp

    def lose_ack(client, command):
        client.disconnect()
        return mqtt.MQTT_ERR_SUCCESS

    monkeypatch.setattr(mqtt.Client, "_handle_pubackcomp", lose_ack)
    worker.send_timeout = 0.5
    assert worker.drain_once(limit=1) == 0
    first = consumer.get(timeout=10)[1]
    assert rows(path, "record_delivery_state")[0]["acked_at"] is None
    monkeypatch.setattr(mqtt.Client, "_handle_pubackcomp", original)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE record_delivery_state SET next_attempt_at=0")
    assert worker.drain_once(limit=1) == 1
    second = consumer.get(timeout=10)[1]
    assert second == first
    assert [a["outcome"] for a in rows(path, "record_delivery_attempts")] == [
        "uncertain",
        "broker_acked",
    ]


@pytest.mark.parametrize("point", ["enqueue", "claim", "puback", "complete"])
def test_real_process_exit_preserves_delivery_and_id(broker, consumer, tmp_path, point):
    path = tmp_path / "common.db"
    seed(path)
    destination = "record-mqtt:cctv/records/v1/" + uuid4().hex
    script = r"""
import os, sys
from src.record_delivery import RecordDeliveryStore, RecordDeliveryWorker
from src.protocols.record_mqtt import MqttRecordTransport
path, broker, destination, point = sys.argv[1:]
store = RecordDeliveryStore(path)
sender = MqttRecordTransport(broker, transport='unix')
owner, method = {
    'enqueue': (store, 'enqueue_pending'), 'claim': (store, 'claim'),
    'puback': (sender, 'send'), 'complete': (store, 'finish'),
}[point]
original = getattr(owner, method)
def exit_after(*args, **kwargs):
    result = original(*args, **kwargs)
    os._exit(42)
setattr(owner, method, exit_after)
RecordDeliveryWorker(store, sender, namespace='video_detection', destination=destination,
                     send_timeout=0.5, lease_seconds=3).drain_once(limit=1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), broker, destination, point],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 42, result.stderr.decode()
    first = consumer.get(timeout=10)[1] if point in {"puback", "complete"} else None
    state = rows(path, "record_delivery_state")[0]
    if state["lease_until"]:
        time.sleep(max(0, state["lease_until"] - time.time()) + 0.05)
    store, worker = make_worker(path, broker, destination)
    assert worker.drain_once(limit=1) == (0 if point == "complete" else 1)
    recovered = consumer.get(timeout=10)[1] if point != "complete" else first
    assert recovered["delivery_id"] == state["job_id"]
    if first:
        assert recovered == first
    assert len(rows(path, "canonical_records")) == 1
    assert len(rows(path, "delivery_jobs")) == 2
    assert rows(path, "record_delivery_state")[0]["ack_kind"] == "broker_puback"
    assert worker.drain_once(limit=1) == 0
