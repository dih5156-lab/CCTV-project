"""격리 브로커에서 전광판 완료 발행/유실 후 조회와 다음 경보를 검증한다."""

import json
from uuid import uuid4

import pytest

from src.edgex.command_contract import handle_mqtt_request
from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from tests.test_action_completion import idle, make_service
from tests.test_action_device_queue import enqueue, make_queue, queue, rows
from tests.test_mqtt_durable_integration import broker as broker
from tests.test_mqtt_durable_integration import publish, wait_until
from tests.test_mqtt_durable_integration import publisher as publisher


@pytest.mark.parametrize("lose_completion", [False, True])
def test_real_mqtt_device_completion_and_status_reconciliation(broker, publisher, tmp_path, lose_completion):
    bridge = make_queue(tmp_path)
    service, device, _ = make_service(tmp_path)
    device.config.display_time = 0  # 실물 시간 정책과 무관한 모의 장치의 즉시 유휴 전환
    result_topic = "edgex/results/cctv/jetson-01/signboard"
    requests = []
    def send(topic, payload):
        publish(publisher, topic, json.dumps(payload).encode())
        return True
    def receive(topic, raw):
        request = json.loads(raw)
        requests.append(request)
        result = handle_mqtt_request(service, request, "signboard")
        send(result_topic, result)
    service.on_completion = lambda result: False if lose_completion else send(result_topic, result)
    queue(bridge).publish = send
    device_sub = MqttTopicSubscriber(
        broker=broker, topics=["edgex/commands/cctv/jetson-01/signboard/#"],
        client_id="completion-device-" + uuid4().hex, transport="unix", message_handler=receive,
    )
    action_sub = MqttTopicSubscriber(
        broker=broker, topics=[result_topic], client_id="completion-action-" + uuid4().hex,
        transport="unix", durable=True, message_handler=bridge._mqtt_intake.accept,
    )
    try:
        assert device_sub.connect() and action_sub.connect()
        enqueue(bridge, "one")
        enqueue(bridge, "two")
        bridge._mqtt_intake.start()
        wait_until(lambda: rows(bridge)[0]["state"] == "acknowledged")
        assert [r["event_id"] for r in requests if r["action"] == "display"] == ["one"]
        idle(device)
        if lose_completion:
            assert service.get_request_status(rows(bridge)[0]["request_id"])["execution_completed"] is True
            bridge._mqtt_intake.stop()
            bridge._mqtt_intake.start()
        wait_until(lambda: len([r for r in requests if r["action"] == "display"]) == 2)
        assert rows(bridge)[0]["state"] == "complete"
        assert [r["event_id"] for r in requests if r["action"] == "display"] == ["one", "two"]
        if lose_completion:
            assert any(r["action"] == "request_status" and r["event_id"] == "one" for r in requests)
    finally:
        device_sub.disconnect()
        action_sub.disconnect()
        bridge._mqtt_intake.stop()
