"""센서 측정값 MQTT를 운영 이벤트 MQTT로 변환한다."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..core.sensor_detection import SensorEventDetector
from ..devices.sensor_device import SensorReading

logger = logging.getLogger(__name__)


def build_sensor_bridge_inputs(
    sensor_message: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """파싱된 센서 메시지를 SensorReading 입력 형식으로 맞춘다."""
    raw_data = sensor_message.get("data") or {}
    if not isinstance(raw_data, Mapping):
        raw_data = {}

    table_name = str(
        sensor_message.get("table")
        or sensor_message.get("tableName")
        or raw_data.get("tableName")
        or "unknown"
    )
    received_at = sensor_message.get("received_at") or sensor_message.get("timestamp") or 0

    uplink_message = {
        "app_eui": sensor_message.get("app_eui"),
        "dev_eui": sensor_message.get("dev_eui"),
        "device_id": sensor_message.get("device_id"),
        "timestamp": received_at,
        "rx_metadata": [{
            "time": received_at,
        }],
    }
    decoded_payload = {
        "tableName": table_name,
        "data": dict(raw_data),
    }
    return uplink_message, decoded_payload


def build_rule_topic(event_type: str, topic_prefix: str = "aiot/rules/sensor") -> str:
    """운영 이벤트 타입을 센서 규칙 토픽으로 변환한다."""
    normalized = str(event_type or "unknown").strip().lower()
    if normalized.endswith("_alert"):
        normalized = normalized[:-6]
    return f"{topic_prefix.rstrip('/')}/{normalized}"


class SensorRuleBridgeService:
    """센서 측정값 메시지를 운영 이벤트 목록으로 변환한다."""

    def __init__(
        self,
        detector: Optional[SensorEventDetector] = None,
    ) -> None:
        self.detector = detector or SensorEventDetector()

    def process_sensor_message(
        self,
        sensor_message: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """센서 측정값 메시지를 운영 이벤트 payload 목록으로 변환한다."""
        uplink_message, decoded_payload = build_sensor_bridge_inputs(sensor_message)
        reading = SensorReading.from_decoded(uplink_message, decoded_payload)
        return [
            event.to_payload()
            for event in self.detector.detect_events(reading)
        ]

    @staticmethod
    def parse_message(payload: bytes) -> Optional[Dict[str, Any]]:
        """MQTT payload bytes를 JSON으로 파싱한다."""
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("센서 측정값 JSON 파싱 실패: %s", exc)
            return None
        if not isinstance(data, dict):
            logger.warning("센서 측정값 형식이 dict가 아닙니다: %r", data)
            return None
        return data
