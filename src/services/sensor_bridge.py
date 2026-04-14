"""센서 업링크를 표준 운영 이벤트로 변환해 MQTT로 발행하는 브리지."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from ..core.sensor_detection import SensorAlertEvent, SensorEventDetector
from ..devices.sensor_device import SensorReading
from ..protocols.mqtt_publisher import MqttEventPublisher
from ..protocols.tlv_decoder import GoTLVDecoderClient

logger = logging.getLogger(__name__)


class SensorBridgeService:
    """Go TLV 디코드 결과를 Python 규칙 엔진으로 판정하고 내부 MQTT로 발행한다."""

    def __init__(
        self,
        *,
        publisher: MqttEventPublisher,
        detector: Optional[SensorEventDetector] = None,
        decoder_client: Optional[GoTLVDecoderClient] = None,
    ) -> None:
        self.publisher = publisher
        self.detector = detector or SensorEventDetector()
        self.decoder_client = decoder_client

    def process_uplink(self, uplink_message: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if self.decoder_client is None:
            raise RuntimeError("decoder_client 없이 raw uplink를 처리할 수 없습니다")

        decoded_payload = self.decoder_client.decode_uplink(uplink_message)
        if not decoded_payload:
            return []
        return self.process_decoded_uplink(uplink_message, decoded_payload)

    def process_decoded_uplink(
        self,
        uplink_message: Mapping[str, Any],
        decoded_payload: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        reading = SensorReading.from_decoded(uplink_message, decoded_payload)
        detected_events = self.detector.detect_events(reading)

        published_events: List[Dict[str, Any]] = []
        for event in detected_events:
            payload = event.to_payload()
            if self.publisher.publish_event(payload):
                published_events.append(payload)
                logger.info(
                    "센서 이벤트 발행 성공: camera_id=%s type=%s severity=%s",
                    payload["camera_id"],
                    payload["type"],
                    payload["severity"],
                )
            else:
                logger.warning(
                    "센서 이벤트 발행 실패: camera_id=%s type=%s",
                    payload["camera_id"],
                    payload["type"],
                )

        if not published_events:
            logger.debug(
                "SensorBridge 발행 이벤트 없음: dev_eui=%s table=%s",
                reading.dev_eui,
                reading.table_name,
            )

        return published_events
