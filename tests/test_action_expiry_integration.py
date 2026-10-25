"""격리 브로커에서 기한 저장 후 ACK와 오프라인 경보의 재개 조건을 확인한다."""

import json
import time
from uuid import uuid4

import pytest

from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from tests.test_action_expiry import TOPIC, make_bridge
from tests.test_mqtt_durable_integration import broker as broker
from tests.test_mqtt_durable_integration import publish, wait_until
from tests.test_mqtt_durable_integration import publisher as publisher


@pytest.mark.parametrize("age,expected", [(1, "complete"), (60, "quarantined")])
def test_offline_event_keeps_source_deadline_and_commits_before_ack(
    broker, publisher, tmp_path, age, expected
):
    client_id = "expiry-offline-" + uuid4().hex
    initial = make_bridge(tmp_path, capture=True)
    initial._mqtt_intake.start()
    subscriber = MqttTopicSubscriber(
        broker=broker,
        topics=[TOPIC],
        client_id=client_id,
        durable=True,
        transport="unix",
        message_handler=initial._mqtt_intake.accept,
    )
    try:
        assert subscriber.connect()
        publish(subscriber, TOPIC + "/barrier", b"barrier")
    finally:
        subscriber.disconnect()
        initial._mqtt_intake.stop()
    occurred = time.time() - age
    raw = json.dumps(
        {
            "event_id": "event-1",
            "camera_id": "cam1",
            "type": "fall_detected",
            "severity": "critical",
            "occurred_at": occurred,
        }
    ).encode()
    publish(publisher, TOPIC, raw)
    resumed = make_bridge(tmp_path, capture=True)
    # 브로커 접수와 ACK를 먼저 확인한 다음, 저장된 입력의 실행을 별도로 재개한다.
    resumed._mqtt_intake.start()
    received = []
    subscriber = MqttTopicSubscriber(
        broker=broker,
        topics=[TOPIC],
        client_id=client_id,
        durable=True,
        transport="unix",
        message_handler=resumed._mqtt_intake.accept,
    )
    client = subscriber._build_client()
    original_ack = client.ack

    def ack(mid, qos):
        with resumed._mqtt_intake.store._connect() as connection:
            row = connection.execute(
                "SELECT ingest_id, projection_json FROM sensor_ingest_state"
            ).fetchone()
            saved = json.loads(
                connection.execute(
                    "SELECT plan_json FROM action_execution_windows"
                ).fetchone()[0]
            )
            assert saved == json.loads(row["projection_json"])
            assert saved["deadlines"]["signboard"] == occurred + 10
        result = original_ack(mid, qos)
        received.append((row["ingest_id"], result))
        return result

    client.ack = ack
    subscriber._client = client
    try:
        assert subscriber.connect()
        assert subscriber.session_present
        wait_until(lambda: len(received) == 1)
        key, result = received[0]
        assert result == 0
        resumed._mqtt_intake._process(key)
        receipt = resumed._mqtt_intake.store.get(key)
        assert receipt["stage"] == expected
        if expected == "complete":
            resumed._executor._publish_edgex_command.assert_called_once()
        else:
            assert receipt["last_error"] == "expired:expired"
            resumed._executor._publish_edgex_command.assert_not_called()
    finally:
        subscriber.disconnect()
        resumed._mqtt_intake.stop()
