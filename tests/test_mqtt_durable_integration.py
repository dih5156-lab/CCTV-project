"""Opt-in real broker tests; use only a disposable Unix-socket Mosquitto."""

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from uuid import uuid4

import paho.mqtt.client as mqtt
import pytest

from src.config import AppConfig
from src.event_ingest_store import EventIngestStore
from src.services.external_ingest import ExternalIngestService

PAYLOAD = b'{"camera_id":"test-cam","type":"telemetry","timestamp":"2026-09-18T00:00:00Z","message_id":"test-msg"}'


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for isolated MQTT test condition")


def count(store, table):
    with sqlite3.connect(store.path) as connection:
        return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


@pytest.fixture
def broker(monkeypatch):
    socket_path = os.environ.get("CCTV_TEST_MQTT_SOCKET")
    if not socket_path:
        pytest.skip("CCTV_TEST_MQTT_SOCKET requires a disposable local broker")
    path = Path(socket_path)
    assert str(path).startswith("/tmp/cctv-p22-mqtt-") and path.is_socket()
    for key in ("MQTT_USER", "MQTT_PASSWORD", "AIOT_INGEST_DB", "MQTT_EVENT_OUTBOX_DB"):
        monkeypatch.delenv(key, raising=False)
    return socket_path


@pytest.fixture
def publisher(broker):
    connected = Event()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="unix")
    client.on_connect = (
        lambda client, userdata, flags, code, properties: connected.set()
    )
    client.connect(broker)
    client.loop_start()
    try:
        assert connected.wait(5)
        yield client
    finally:
        client.disconnect()
        client.loop_stop()


def publish(client, topic, payload=PAYLOAD):
    result = client.publish(topic, payload, qos=1)
    result.wait_for_publish(timeout=5)
    assert result.is_published()


def make_service(broker, path, client_id, topic):
    config = AppConfig()
    config.external_ingest.mqtt_broker = broker
    config.external_ingest.mqtt_client_id = client_id
    config.external_ingest.topics = (topic,)
    config.external_ingest.republish_enabled = False
    service = ExternalIngestService(
        config.external_ingest, config.mqtt, ingest_store=EventIngestStore(path)
    )
    service._subscriber.transport = "unix"
    acked = Queue()
    delivered = []
    original_build = service._subscriber._build_client

    def build():
        client = original_build()
        original_ack, original_message = client.ack, client.on_message

        def acknowledge(mid, qos):
            result = original_ack(mid, qos)
            acked.put((mid, qos, result))
            return result

        def receive(client, userdata, message):
            delivered.append((message.mid, message.qos, message.dup))
            original_message(client, userdata, message)

        client.ack, client.on_message = acknowledge, receive
        return client

    service._subscriber._build_client = build
    return service, acked, delivered


def flush_ack(service, acked, topic):
    _, qos, result = acked.get(timeout=5)
    assert qos == 1 and result == 0
    # This outbound PUBLISH follows PUBACK on the same TCP/Unix byte stream.
    # Its broker response confirms the earlier PUBACK reached the broker.
    publish(service._subscriber._client, topic + "/barrier", b"barrier")


def test_saved_and_quarantined_messages_ack_and_do_not_redeliver(
    broker, publisher, tmp_path
):
    client_id = "ack-" + uuid4().hex
    topic = "test/" + client_id
    path = tmp_path / "common.db"
    service, acked, _ = make_service(broker, path, client_id, topic)
    try:
        service.start()
        publish(publisher, topic)
        flush_ack(service, acked, topic)
        publish(publisher, topic, b"\xff{broken")
        flush_ack(service, acked, topic)
        assert count(service._ingest_store, "ingest_receipts") == 2
        assert count(service._ingest_store, "canonical_records") == 1
        assert service.get_stats()["quarantined_count"] == 1
    finally:
        service.stop()
    resumed, _, delivered = make_service(broker, path, client_id, topic)
    try:
        resumed.start()
        assert resumed._subscriber.session_present is True
        publish(resumed._subscriber._client, topic + "/barrier", b"barrier")
        assert delivered == []
        assert count(resumed._ingest_store, "ingest_receipts") == 2
    finally:
        resumed.stop()


def test_sqlite_failure_no_ack_then_same_session_redelivers(
    broker, publisher, tmp_path, monkeypatch
):
    client_id = "disk-" + uuid4().hex
    topic = "test/" + client_id
    service, acked, delivered = make_service(
        broker, tmp_path / "common.db", client_id, topic
    )
    store = service._ingest_store
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON ingest_receipts "
            "BEGIN SELECT RAISE(ABORT, 'injected disk failure'); END"
        )
    monkeypatch.setattr(
        "src.services.external_ingest.signal.signal", lambda *args: None
    )
    failures = []

    def run_service():
        try:
            service.run_forever()
        except Exception as exc:
            failures.append(exc)

    owner = Thread(target=run_service, daemon=True)
    owner.start()
    try:
        wait_until(lambda: service._running)
        publish(publisher, topic)
        wait_until(lambda: service._subscriber._restart_requested)
        assert acked.empty()
        assert count(store, "ingest_receipts") == 0
        with sqlite3.connect(store.path) as connection:
            connection.execute("DROP TRIGGER fail_receipt")
        wait_until(lambda: len(delivered) == 2)
        flush_ack(service, acked, topic)
        assert service._subscriber.session_present is True
        assert len(delivered) == 2
        assert delivered[0][0] == delivered[1][0]
        assert delivered[1][2] is True
        assert count(store, "ingest_receipts") == 1
        assert count(store, "canonical_records") == 1
    finally:
        service._running = False
        owner.join(timeout=8)
        if owner.is_alive():
            service.stop()
        assert not owner.is_alive()
        assert failures == []


def test_persistent_session_queues_input_while_service_is_offline(
    broker, publisher, tmp_path
):
    client_id = "offline-" + uuid4().hex
    topic = "test/" + client_id
    path = tmp_path / "common.db"
    service, _, _ = make_service(broker, path, client_id, topic)
    try:
        service.start()
    finally:
        service.stop()
    publish(publisher, topic)
    resumed, acked, _ = make_service(broker, path, client_id, topic)
    try:
        resumed.start()
        flush_ack(resumed, acked, topic)
        assert resumed._subscriber.session_present is True
        assert count(resumed._ingest_store, "ingest_receipts") == 1
        assert count(resumed._ingest_store, "canonical_records") == 1
    finally:
        resumed.stop()


CRASH_SCRIPT = r"""
import os, sys, time
from pathlib import Path
from threading import Event
from src.config import AppConfig
from src.event_ingest_store import EventIngestStore
from src.services.external_ingest import ExternalIngestService
broker, path, client_id, topic, checkpoint, ready = sys.argv[1:]
config = AppConfig()
config.external_ingest.mqtt_broker = broker
config.external_ingest.mqtt_client_id = client_id
config.external_ingest.topics = (topic,)
config.external_ingest.republish_enabled = False
store = EventIngestStore(path)
service = ExternalIngestService(config.external_ingest, config.mqtt, ingest_store=store)
service._subscriber.transport = "unix"
if checkpoint in ("before_T1", "after_T1"):
    original = store.accept_message
    def accept(*args, **kwargs):
        if checkpoint == "after_T1":
            original(*args, **kwargs)
        os._exit(42)
    store.accept_message = accept
elif checkpoint == "before_ACK":
    original = service._subscriber.message_handler
    def handle(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(42)
    service._subscriber.message_handler = handle
service.start()
ack_sent = Event()
original_ack = service._subscriber._client.ack
def acknowledge(mid, qos):
    result = original_ack(mid, qos)
    ack_sent.set()
    return result
service._subscriber._client.ack = acknowledge
Path(ready).write_text("ready")
if checkpoint == "after_ACK":
    if not ack_sent.wait(10):
        raise AssertionError("ACK timeout")
    result = service._subscriber._client.publish(topic + "/barrier", b"barrier", qos=1)
    result.wait_for_publish(timeout=5)
    if not result.is_published():
        raise AssertionError("Broker barrier timeout")
    os._exit(42)
time.sleep(15)
raise AssertionError("Crash boundary not reached")
"""


@pytest.mark.parametrize(
    "checkpoint,receipts,redelivery",
    [
        ("before_T1", 1, True),
        ("after_T1", 2, True),
        ("before_ACK", 2, True),
        ("after_ACK", 1, False),
    ],
)
def test_real_process_exit_and_broker_redelivery(
    broker, publisher, tmp_path, checkpoint, receipts, redelivery
):
    client_id = "crash-" + uuid4().hex
    topic = "test/" + client_id
    path, ready = tmp_path / "common.db", tmp_path / "ready"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            CRASH_SCRIPT,
            broker,
            str(path),
            client_id,
            topic,
            checkpoint,
            str(ready),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_until(lambda: ready.exists() or process.poll() is not None)
        if not ready.exists():
            _, error = process.communicate(timeout=5)
            pytest.fail(error)
        publish(publisher, topic)
        _, error = process.communicate(timeout=12)
        assert process.returncode == 42, error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    resumed, acked, delivered = make_service(broker, path, client_id, topic)
    try:
        resumed.start()
        assert resumed._subscriber.session_present is True
        if redelivery:
            flush_ack(resumed, acked, topic)
            assert len(delivered) == 1 and delivered[0][2] is True
        else:
            publish(resumed._subscriber._client, topic + "/barrier", b"barrier")
            assert delivered == []
        store = resumed._ingest_store
        assert count(store, "ingest_receipts") == receipts
        assert count(store, "canonical_records") == 1
        assert count(store, "ingest_revisions") == receipts
        with sqlite3.connect(path) as connection:
            event = json.loads(
                connection.execute(
                    "SELECT payload_json FROM canonical_records"
                ).fetchone()[0]
            )
        assert event["raw"]["raw_payload"]["message_id"] == "test-msg"
    finally:
        resumed.stop()
