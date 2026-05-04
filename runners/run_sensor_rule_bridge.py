"""run_sensor_rule_bridge.py - 센서 규칙 브리지 실행 진입점"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

_RUNNER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _RUNNER_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runners._shared import ensure_project_root, setup_runner_logging

ensure_project_root()

from src.protocols._mqtt_factory import create_mqtt_client
from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from src.services.sensor_rule_bridge import (
    SensorRuleBridgeService,
    build_rule_topic,
)

logger = logging.getLogger("run-sensor-rule-bridge")


class _RuleTopicPublisher:
    """센서 운영 이벤트를 지정 토픽으로 발행한다."""

    def __init__(
        self,
        *,
        broker: str,
        port: int,
        topic_prefix: str,
    ) -> None:
        self.broker = broker
        self.port = int(port)
        self.topic_prefix = topic_prefix.rstrip("/")
        self._client = create_mqtt_client("sensor-rule-bridge-pub")
        self._connected = False

    def _on_connect(self, client, userdata, flags, rc, *args) -> None:
        self._connected = rc == 0
        if self._connected:
            logger.info("센서 규칙 MQTT 발행 연결 성공: %s:%s", self.broker, self.port)
        else:
            logger.error("센서 규칙 MQTT 발행 연결 실패 (rc=%s)", rc)

    def _on_disconnect(self, client, userdata, rc, *args) -> None:
        self._connected = False
        if rc != 0:
            logger.warning("센서 규칙 MQTT 발행 연결 해제 (rc=%s)", rc)

    def connect(self) -> bool:
        """발행용 MQTT 클라이언트를 연결한다."""
        if self._connected:
            return True

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()
        except Exception as exc:
            logger.error("센서 규칙 MQTT 발행 연결 오류: %s", exc)
            self._connected = False
            return False

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._connected:
                return True
            time.sleep(0.05)
        return False

    def publish(self, event_payload: dict) -> bool:
        """센서 운영 이벤트를 규칙 토픽으로 발행한다."""
        if not self.connect():
            return False

        topic = build_rule_topic(
            str(event_payload.get("type") or "unknown"),
            topic_prefix=self.topic_prefix,
        )
        body = json.dumps(event_payload, ensure_ascii=False)
        result = self._client.publish(topic, body, qos=0, retain=False)
        if result.rc == 0:
            logger.info(
                "센서 규칙 이벤트 발행 성공: topic=%s camera_id=%s type=%s",
                topic,
                event_payload.get("camera_id"),
                event_payload.get("type"),
            )
            return True

        logger.error("센서 규칙 이벤트 발행 실패 (rc=%s): %s", result.rc, topic)
        return False

    def close(self) -> None:
        """발행용 MQTT 연결을 종료한다."""
        try:
            self._client.loop_stop()
            self._client.disconnect()
        finally:
            self._connected = False


def main() -> None:
    """CLI 진입점."""
    setup_runner_logging()

    parser = argparse.ArgumentParser(description="센서 측정값 -> 규칙 이벤트 MQTT 브리지")
    parser.add_argument("--mqtt-broker", default="localhost", help="MQTT 브로커 호스트")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT 브로커 포트")
    parser.add_argument(
        "--subscribe-topic",
        default="aiot/sensors/#",
        help="구독할 센서 측정값 토픽",
    )
    parser.add_argument(
        "--publish-topic-prefix",
        default="aiot/rules/sensor",
        help="운영 이벤트 발행 토픽 prefix",
    )
    parser.add_argument(
        "--max-pending-events",
        type=int,
        default=500,
        help="MQTT 발행 실패 시 메모리에 보관할 최대 센서 규칙 이벤트 수",
    )
    args = parser.parse_args()

    if args.mqtt_port <= 0:
        parser.error("--mqtt-port는 양수여야 합니다")
    if args.max_pending_events <= 0:
        parser.error("--max-pending-events는 양수여야 합니다")

    service = SensorRuleBridgeService()
    publisher = _RuleTopicPublisher(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic_prefix=args.publish_topic_prefix,
    )
    pending_events = deque(maxlen=args.max_pending_events)

    def _flush_pending_events() -> None:
        while pending_events:
            event_payload = pending_events[0]
            if not publisher.publish(event_payload):
                return
            pending_events.popleft()

    def _handle_message(topic: str, payload: bytes) -> None:
        sensor_message = service.parse_message(payload)
        if not sensor_message:
            return

        _flush_pending_events()
        for event_payload in service.process_sensor_message(sensor_message):
            if not publisher.publish(event_payload):
                was_full = len(pending_events) >= pending_events.maxlen
                pending_events.append(event_payload)
                logger.warning(
                    "센서 규칙 이벤트 발행 보류: topic=%s device_id=%s pending=%d dropped_oldest=%s",
                    topic,
                    sensor_message.get("device_id"),
                    len(pending_events),
                    was_full,
                )

    subscriber = MqttTopicSubscriber(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topics=(args.subscribe_topic,),
        message_handler=_handle_message,
        client_id_prefix="sensor-rule-bridge-sub",
    )

    if not subscriber.connect():
        raise RuntimeError("센서 규칙 브리지 MQTT 연결 실패")

    stop_requested = False

    def _handle_signal(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "센서 규칙 브리지 시작: subscribe=%s publish_prefix=%s",
        args.subscribe_topic,
        args.publish_topic_prefix,
    )

    try:
        while not stop_requested:
            _flush_pending_events()
            time.sleep(0.5)
    finally:
        subscriber.disconnect()
        publisher.close()


if __name__ == "__main__":
    main()
