"""
mqtt/classifier.py
==================
Go 원본: aiot-tlv-parser/pkg/mqtt/classifier.go

MQTT 메시지 분류 및 파싱 모듈입니다.
토픽과 페이로드를 분석하여 적절한 센서 데이터 처리기로 라우팅합니다.

토픽 형식:
  target 타입: v3/{appID}/devices/eui-{devEUI}/up
  da 타입    : {appEUI}/{devEUI}/up
"""

import json
import logging
from dataclasses import dataclass

from mqtt.interfaces import SensorDataProcessor

logger = logging.getLogger(__name__)


@dataclass
class MessageData:
    """
    MQTT 메시지 내부 구조
    Go: type MessageData struct { Payload string; RxMetadata []struct{...} }
    """
    payload: str = ""
    rx_metadata: list = None

    def __post_init__(self):
        if self.rx_metadata is None:
            self.rx_metadata = []


@dataclass
class TopicInfo:
    """
    MQTT 토픽에서 파싱된 정보
    Go: type TopicInfo struct { AppID string; DevEUI string }
    """
    app_id: str = ""
    dev_eui: str = ""


class Classifier:
    """
    MQTT 메시지 분류기
    Go: type Classifier struct { allowedDevices []string; sensorProcessor SensorDataProcessor }

    - 허용된 디바이스만 처리
    - 토픽 형식에 따라 appID / devEUI 추출
    - 페이로드 JSON 파싱 후 SensorDataProcessor 에 전달
    """

    def __init__(self, allowed_devices: list, sensor_processor: SensorDataProcessor):
        """
        Go: func NewClassifier(allowedDevices []string, sensorProcessor SensorDataProcessor) *Classifier
        """
        self._allowed_devices = allowed_devices
        self._sensor_processor = sensor_processor

    def classify_message(self, topic: str, message: bytes) -> None:
        """
        수신된 MQTT 메시지 처리 진입점
        Go: func (c *Classifier) ClassifyMessage(topic string, message []byte)

        처리 흐름:
          1. 토픽 마지막 세그먼트가 "up" 인지 확인
          2. 토픽 형식(target/da) 판별
          3. appID, devEUI 추출
          4. 허용된 디바이스인지 확인
          5. JSON 파싱 후 SensorDataProcessor 호출
        """
        topic_arr = topic.split("/")

        # 'up' 메시지만 처리
        # Go: if topicArr[len(topicArr)-1] != "up" { return }
        if not topic_arr or topic_arr[-1] != "up":
            return

        parser_type = self._get_parser_type(topic)
        topic_info = self._parse_topic(topic_arr, parser_type)

        # 허용된 디바이스 확인
        if not self._is_device_allowed(topic_info.dev_eui):
            logger.debug(f"entry fail: {topic_info.dev_eui}")
            return

        # JSON 파싱
        try:
            raw = json.loads(message)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"error: payload is not exist or parsing error in {topic_info.dev_eui} message: {e}")
            return

        msg_data = MessageData(
            payload=raw.get("payload", ""),
            rx_metadata=raw.get("rx_metadata", []),
        )

        if not msg_data.payload:
            logger.info(f"appId:{topic_info.app_id} devEui:{topic_info.dev_eui} : has not payload")
            return

        if not msg_data.rx_metadata:
            logger.info(f"appId:{topic_info.app_id} devEui:{topic_info.dev_eui} : has not rx_metadata")
            return

        # received_at 추출
        received_at = 0
        if msg_data.rx_metadata:
            received_at = int(msg_data.rx_metadata[0].get("time", 0))

        # 센서 데이터 처리기 호출
        if topic_info.app_id and topic_info.dev_eui:
            meta = msg_data.rx_metadata[0]
            self._sensor_processor.process_sensor_data(
                app_id=topic_info.app_id,
                dev_eui=topic_info.dev_eui,
                payload=msg_data.payload,
                channel=meta.get("channel", 0),
                frequency=meta.get("frequency", 0),
                received_at=received_at,
            )

    def _get_parser_type(self, topic: str) -> str:
        """
        토픽에서 파서 타입 결정
        Go: func (c *Classifier) getParserType(topic string) string

        "/devices/" 포함 → "target"
        그 외             → "da"
        """
        if "/devices/" in topic:
            return "target"
        return "da"

    def _parse_topic(self, topic_arr: list, parser_type: str) -> TopicInfo:
        """
        토픽 배열에서 AppID, DevEUI 추출
        Go: func (c *Classifier) parseTopic(topicArr []string, parserType string) TopicInfo

        target 형식: v3/{appID}/devices/eui-{devEUI}/up
          → topicArr[1] = appID
          → topicArr[3] = "eui-{devEUI}" → "eui-" 제거 후 대문자화

        da 형식: {appEUI}/{devEUI}/up
          → topicArr[0] = appEUI
          → topicArr[1] = devEUI (대문자화)
        """
        if parser_type == "target":
            if len(topic_arr) >= 4:
                app_id = topic_arr[1]
                dev_eui = topic_arr[3].replace("eui-", "").upper()
                return TopicInfo(app_id=app_id, dev_eui=dev_eui)
        elif parser_type == "da":
            if len(topic_arr) >= 2:
                app_eui = topic_arr[0]
                dev_eui = topic_arr[1].upper()
                return TopicInfo(app_id=app_eui, dev_eui=dev_eui)
        return TopicInfo()

    def _is_device_allowed(self, dev_eui: str) -> bool:
        """
        디바이스 허용 여부 확인
        Go: func (c *Classifier) isDeviceAllowed(devEUI string) bool
        """
        return dev_eui in self._allowed_devices
