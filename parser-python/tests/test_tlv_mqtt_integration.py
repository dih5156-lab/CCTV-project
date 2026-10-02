"""Disposable broker/PG only; real client exits and real transaction boundaries."""

import sqlite3
import subprocess
import sys
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "parser-python"))

from database.sensor_projection_worker import SensorProjectionWorker  # noqa: E402
from mqtt.manager import Manager  # noqa: E402
from test_mqtt_durable_integration import (  # noqa: E402, F401
    broker as broker,
)
from test_mqtt_durable_integration import (
    publish,
    wait_until,
)
from test_mqtt_durable_integration import (
    publisher as publisher,
)
from test_sensor_projection import counts  # noqa: E402, F401
from test_sensor_projection import pg_db as pg_db
from test_tlv_mqtt_durable import receipts, sensor_service, wire_payload  # noqa: E402

from src.ingest_store import IngestStore  # noqa: E402
from src.protocols.mqtt_subscriber import MqttTopicSubscriber  # noqa: E402


def make_manager(broker, store, client_id, topic):
    service = sensor_service(store)
    manager = Manager(None, durable=True, client_id_prefix=client_id)
    manager.set_processor(service)
    acked, delivered = Queue(), []

    def subscriber_factory(**options):
        options.update(transport="unix", topics=(topic,))
        subscriber = MqttTopicSubscriber(**options)
        original_build = subscriber._build_client

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

        subscriber._build_client = build
        return subscriber

    config = SimpleNamespace(host=broker, port=1883, username="", password="")
    with patch(
        "src.protocols.mqtt_subscriber.MqttTopicSubscriber",
        side_effect=subscriber_factory,
    ):
        manager.connect_client("lab", config)
    wait_until(lambda: manager.get_client_status("lab"), timeout=15)
    return manager, service, manager._subscribers["lab"], acked, delivered


def flush_ack(subscriber, acked, topic):
    _, qos, result = acked.get(timeout=20)
    assert (qos, result) == (1, 0)
    publish(subscriber._client, topic + "/barrier", b"barrier")


def drain(service, store, db):
    SensorProjectionWorker(store, db, service._recover_receipt).drain_once()


def test_normal_and_invalid_bytes_ack_with_pg_projection(
    broker, publisher, pg_db, tmp_path
):
    client_id = "tlv-" + uuid4().hex
    topic = client_id + "/dev/up"
    store = IngestStore(tmp_path / "common.db")
    manager, service, subscriber, acked, _ = make_manager(
        broker, store, client_id, topic
    )
    try:
        publish(publisher, topic, wire_payload())
        flush_ack(subscriber, acked, topic)
        publish(publisher, topic, b"\xff{broken")
        flush_ack(subscriber, acked, topic)
        drain(service, store, pg_db)
        assert [store.get(key)["stage"] for key in receipts(store)] == [
            "complete",
            "quarantined",
        ]
        assert counts(pg_db) == [1, 1, 1]
    finally:
        manager.disconnect_all()
    manager, _, subscriber, _, delivered = make_manager(broker, store, client_id, topic)
    try:
        assert subscriber.session_present
        publish(subscriber._client, topic + "/barrier", b"barrier")
        assert delivered == []
    finally:
        manager.disconnect_all()


def test_failed_t1_has_no_ack_and_manager_poll_redelivers(
    broker, publisher, pg_db, tmp_path
):
    client_id = "disk-" + uuid4().hex
    topic = client_id + "/dev/up"
    store = IngestStore(tmp_path / "common.db")
    with store._connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON ingest_receipts BEGIN SELECT RAISE(ABORT, 'test disk failure'); END"
        )
    manager, service, subscriber, acked, delivered = make_manager(
        broker, store, client_id, topic
    )
    try:
        publish(publisher, topic, wire_payload())
        wait_until(lambda: subscriber._restart_requested)
        assert acked.empty() and receipts(store) == []
        with store._connect() as connection:
            connection.execute("DROP TRIGGER fail_receipt")
        wait_until(lambda: (manager.poll(), not acked.empty())[1])
        flush_ack(subscriber, acked, topic)
        assert subscriber.session_present
        assert len(delivered) == 2 and delivered[1] == (delivered[0][0], 1, True)
        drain(service, store, pg_db)
        assert len(receipts(store)) == 1 and counts(pg_db) == [1, 1, 1]
    finally:
        manager.disconnect_all()


def test_failure_after_t1_ack_then_recover_same_receipt(
    broker, publisher, pg_db, tmp_path
):
    client_id = "prepare-" + uuid4().hex
    topic = client_id + "/dev/up"
    store = IngestStore(tmp_path / "common.db")
    manager, service, subscriber, acked, _ = make_manager(
        broker, store, client_id, topic
    )
    try:
        with patch.object(
            store,
            "prepare",
            side_effect=sqlite3.OperationalError("test prepare failure"),
        ):
            publish(publisher, topic, wire_payload())
            flush_ack(subscriber, acked, topic)
        assert len(receipts(store)) == 1 and counts(pg_db) == [0, 0, 0]
        with store._connect() as connection:
            connection.execute("UPDATE sensor_ingest_state SET next_attempt_at=0")
        drain(service, store, pg_db)
        assert len(receipts(store)) == 1 and counts(pg_db) == [1, 1, 1]
        assert not service._edgex_forwarder.mock_calls
    finally:
        manager.disconnect_all()


def test_pg_commit_before_sqlite_completion_is_idempotent(pg_db, tmp_path):
    path = tmp_path / "common.db"
    store = IngestStore(path)
    key = store.accept_mqtt("tlv_mqtt", "app/dev/up", wire_payload())
    service = sensor_service(store)
    with patch.object(
        store, "complete", side_effect=sqlite3.OperationalError("test response lost")
    ):
        drain(service, store, pg_db)
    assert store.get(key)["stage"] == "ready" and counts(pg_db) == [1, 1, 1]
    resumed = IngestStore(path)
    with resumed._connect() as connection:
        connection.execute("UPDATE sensor_ingest_state SET next_attempt_at=0")
    drain(sensor_service(resumed), resumed, pg_db)
    assert receipts(resumed) == [key] and resumed.get(key)["stage"] == "complete"
    assert counts(pg_db) == [1, 1, 1]
    assert not service._edgex_forwarder.mock_calls


def test_offline_session_keeps_input(broker, publisher, pg_db, tmp_path):
    client_id = "offline-" + uuid4().hex
    topic = client_id + "/dev/up"
    store = IngestStore(tmp_path / "common.db")
    manager, _, _, _, _ = make_manager(broker, store, client_id, topic)
    manager.disconnect_all()
    publish(publisher, topic, wire_payload())
    manager, service, subscriber, acked, _ = make_manager(
        broker, store, client_id, topic
    )
    try:
        flush_ack(subscriber, acked, topic)
        assert subscriber.session_present
        drain(service, store, pg_db)
        assert counts(pg_db) == [1, 1, 1]
    finally:
        manager.disconnect_all()


CRASH_SCRIPT = r"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, "parser-python/tests")
from test_tlv_mqtt_integration import make_manager, flush_ack
from src.ingest_store import IngestStore
broker, path, client_id, topic, checkpoint, ready = sys.argv[1:]
store = IngestStore(path)
if checkpoint in ("before_T1", "after_T1"):
    original = store.accept_mqtt
    def accept(*args, **kwargs):
        if checkpoint == "after_T1":
            original(*args, **kwargs)
        os._exit(42)
    store.accept_mqtt = accept
manager, service, subscriber, acked, delivered = make_manager(broker, store, client_id, topic)
if checkpoint == "before_ACK":
    original = subscriber.message_handler
    def handle(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(42)
    subscriber.message_handler = handle
Path(ready).write_text("ready")
if checkpoint == "after_ACK":
    flush_ack(subscriber, acked, topic)
    os._exit(42)
time.sleep(15)
raise AssertionError("Crash boundary not reached")
"""


@pytest.mark.parametrize(
    "checkpoint,expected_receipts,redelivery",
    [
        ("before_T1", 1, True),
        ("after_T1", 2, True),
        ("before_ACK", 2, True),
        ("after_ACK", 1, False),
    ],
)
def test_process_exit_preserves_receipts_and_pg_recovery(
    broker, publisher, pg_db, tmp_path, checkpoint, expected_receipts, redelivery
):
    client_id = "exit-" + uuid4().hex
    topic = client_id + "/dev/up"
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
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_until(lambda: ready.exists() or process.poll() is not None, timeout=30)
        if not ready.exists():
            _, error = process.communicate(timeout=5)
            pytest.fail(error)
        publish(publisher, topic, wire_payload())
        _, error = process.communicate(timeout=30)
        assert process.returncode == 42, error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    store = IngestStore(path)
    recovering = sensor_service(store)
    drain(recovering, store, pg_db)
    assert not recovering._edgex_forwarder.mock_calls
    manager, service, subscriber, acked, delivered = make_manager(
        broker, store, client_id, topic
    )
    try:
        assert subscriber.session_present
        if redelivery:
            flush_ack(subscriber, acked, topic)
            assert len(delivered) == 1 and delivered[0][2]
        else:
            publish(subscriber._client, topic + "/barrier", b"barrier")
            assert delivered == []
        drain(service, store, pg_db)
        assert len(receipts(store)) == expected_receipts
        # No producer ID contract: preserve separate broker deliveries, even if values match.
        assert counts(pg_db) == [expected_receipts] * 3
        assert all(store.get(key)["stage"] == "complete" for key in receipts(store))
    finally:
        manager.disconnect_all()
