"""격리 MQTT 경로에서 실제 접수·결과 수신과 물리 큐 점유 해제를 확인한다."""

import json
import time
from uuid import uuid4

from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from tests.test_action_device_queue import emitted, make_queue, rows
from tests.test_action_expiry import TOPIC
from tests.test_mqtt_durable_integration import broker as broker
from tests.test_mqtt_durable_integration import publish, wait_until
from tests.test_mqtt_durable_integration import publisher as publisher


def test_real_mqtt_ack_only_holds_queue_until_completion_evidence(
    broker, publisher, tmp_path
):
    bridge = make_queue(tmp_path)
    result_topic = "edgex/results/cctv/signboard"
    subscriber = MqttTopicSubscriber(
        broker=broker,
        topics=[TOPIC, result_topic],
        client_id="physical-queue-" + uuid4().hex,
        durable=True,
        transport="unix",
        message_handler=bridge._mqtt_intake.accept,
    )
    bridge._mqtt_intake.start()
    try:
        assert subscriber.connect()
        for event_id in ["one", "two"]:
            publish(
                publisher,
                TOPIC,
                json.dumps(
                    {
                        "event_id": event_id,
                        "camera_id": "cam1",
                        "type": "fall_detected",
                        "severity": "critical",
                        "occurred_at": time.time() - 1,
                    }
                ).encode(),
            )
        wait_until(lambda: len(rows(bridge)) == 2 and emitted(bridge) == ["one"])
        row = rows(bridge)[0]
        result = {
            "request_id": row["request_id"],
            "event_id": row["event_id"],
            "device_id": row["resource_key"],
            "status": "acknowledged",
        }
        publish(publisher, result_topic, json.dumps(result).encode())
        wait_until(lambda: rows(bridge)[0]["state"] == "acknowledged")
        assert emitted(bridge) == ["one"]
        result.update(execution_completed=True, resource_idle=True)
        publish(publisher, result_topic, json.dumps(result).encode())
        wait_until(lambda: emitted(bridge) == ["one", "two"])
        assert rows(bridge)[0]["state"] == "complete"
    finally:
        subscriber.disconnect()
        bridge._mqtt_intake.stop()
