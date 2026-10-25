"""격리 Unix socket 브로커에서 Action 영속 접수와 재전달 경계를 확인한다."""

import sqlite3
from threading import Event, Thread
from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from src.services.action_bridge import ActionBridge
from tests.test_mqtt_durable_integration import broker as broker
from tests.test_mqtt_durable_integration import publish, wait_until
from tests.test_mqtt_durable_integration import publisher as publisher


def receipt_count(bridge):
    with sqlite3.connect(bridge._mqtt_intake.store.path) as connection:
        return connection.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0]


@pytest.fixture
def running_bridge(broker, tmp_path, monkeypatch):
    monkeypatch.setattr("src.services.action_bridge.signal.signal", lambda *args: None)
    owners = []

    def start(client_id, topic, *, handler=None, before_start=None):
        bridge = ActionBridge(
            mqtt_broker=broker, subscribe_topics={topic},
            db_path=str(tmp_path / "actions.db"),
            mqtt_intake_db=str(tmp_path / "input.db"), mqtt_client_id=client_id,
            mqtt_capture_only=handler is None, edgex_command_mode="edgex",
        )
        bridge._forwarder = Mock()
        if handler is not None:
            bridge._process_mqtt_payload = handler
        acked, delivered = [], []

        def subscriber(**kwargs):
            instance = MqttTopicSubscriber(**kwargs, transport="unix")
            original_build = instance._build_client

            def build():
                client = original_build()
                acknowledge, receive = client.ack, client.on_message

                def ack(mid, qos):
                    count = receipt_count(bridge)
                    result = acknowledge(mid, qos)
                    acked.append((mid, qos, count, result))
                    return result

                def message(client, userdata, msg):
                    delivered.append((msg.mid, msg.qos, msg.dup))
                    receive(client, userdata, msg)

                client.ack, client.on_message = ack, message
                return client

            instance._build_client = build
            return instance

        monkeypatch.setattr("src.services.action_bridge.MqttTopicSubscriber", subscriber)
        if before_start is not None:
            before_start(bridge)
        failures = []

        def run():
            try:
                bridge.start()
            except Exception as exc:
                failures.append(exc)

        owner = Thread(target=run, daemon=True)
        owners.append((bridge, owner, failures))
        owner.start()
        wait_until(lambda: bridge._running or bool(failures))
        assert not failures
        return bridge, acked, delivered

    yield start
    for bridge, owner, failures in owners:
        bridge._running = False
        owner.join(timeout=8)
        assert not owner.is_alive()
        assert not failures


def stop(bridge):
    bridge._running = False
    wait_until(lambda: bridge._mqtt_intake._lease is None)


def test_offline_qos1_is_saved_after_reconnect_without_output(running_bridge, publisher):
    client_id = "action-offline-" + uuid4().hex
    topic = "test/" + client_id
    bridge, _, _ = running_bridge(client_id, topic)
    bridge._forwarder.start.assert_not_called()
    stop(bridge)
    publish(publisher, topic, b"offline-1")
    publish(publisher, topic, b"\xffinvalid-json")
    resumed, acked, _ = running_bridge(client_id, topic)
    wait_until(lambda: len(acked) == 2)
    assert resumed._mqtt_client.session_present is True
    assert receipt_count(resumed) == 2
    assert all(qos == 1 and count >= 1 and result == 0 for _, qos, count, result in acked)
    assert len(resumed._mqtt_intake.store.pending(namespace="action_mqtt")) == 2
    resumed._forwarder.start.assert_not_called()


def test_sqlite_failure_redelivers_same_packet_without_false_ack(running_bridge, publisher):
    client_id = "action-failed-" + uuid4().hex
    topic = "test/" + client_id

    def reject(bridge):
        with sqlite3.connect(bridge._mqtt_intake.store.path) as connection:
            connection.execute("CREATE TRIGGER reject_input BEFORE INSERT ON ingest_receipts "
                               "BEGIN SELECT RAISE(ABORT, 'disk full'); END")

    bridge, acked, delivered = running_bridge(client_id, topic, before_start=reject)
    publish(publisher, topic, b"must-survive")
    wait_until(lambda: bridge._mqtt_client._restart_requested)
    assert acked == [] and receipt_count(bridge) == 0
    with sqlite3.connect(bridge._mqtt_intake.store.path) as connection:
        connection.execute("DROP TRIGGER reject_input")
    wait_until(lambda: len(acked) == 1)
    assert delivered[0][0] == delivered[-1][0]
    assert delivered[-1][2] is True
    assert bridge._mqtt_client.session_present is True
    assert receipt_count(bridge) == 1
    assert acked[0][2:] == (1, 0)


def test_blocked_execution_does_not_block_raw_commit_or_ack(running_bridge, publisher):
    entered, release = Event(), Event()

    def execute(*args):
        entered.set()
        assert release.wait(6)

    client_id = "action-slow-" + uuid4().hex
    topic = "test/" + client_id
    bridge, acked, _ = running_bridge(client_id, topic, handler=execute)
    try:
        publish(publisher, topic, b"first")
        assert entered.wait(3)
        publish(publisher, topic, b"second")
        wait_until(lambda: len(acked) == 2)
        assert receipt_count(bridge) == 2
        assert not release.is_set()
    finally:
        release.set()


def test_qos0_offline_is_not_claimed_as_preserved(running_bridge, publisher):
    client_id = "action-qos0-" + uuid4().hex
    topic = "test/" + client_id
    bridge, _, _ = running_bridge(client_id, topic)
    stop(bridge)
    result = publisher.publish(topic, b"qos0", qos=0)
    result.wait_for_publish(timeout=3)
    publish(publisher, topic + "/barrier", b"barrier")
    resumed, acked, delivered = running_bridge(client_id, topic)
    publish(resumed._mqtt_client, topic + "/barrier", b"barrier")
    assert receipt_count(resumed) == 0
    assert acked == [] and delivered == []
