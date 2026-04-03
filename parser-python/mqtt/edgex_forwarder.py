"""
mqtt/edgex_forwarder.py
=======================
파싱된 센서 데이터를 EdgeX MQTT 브로커로 발행하고 Core Data에 직접 push하는 포워더입니다.

BaseMqttPublisher 를 상속해 연결 관리(재연결 백오프)를 재사용합니다.

발행 토픽 형식:
  aiot/sensors/{dev_eui}/{table_name}

EdgeX Core Data 전송:
  POST http://{core_data_host}/api/v3/event/{profileName}/{deviceName}/{sourceName}
  (등록된 device-rest 디바이스 기준, 디바이스명: aiot-{device_id})

환경변수:
  EDGEX_MQTT_HOST      : EdgeX MQTT 브로커 호스트 (기본: edgex-mqtt-broker)
  EDGEX_MQTT_PORT      : 포트 (기본: 1883)
  ALERT_API_URL        : sensor-readings 수신 URL (기본: http://cctv-alert-api:8000/api/sensor-readings)
  EDGEX_CORE_DATA_URL  : Core Data URL (기본: http://edgex-core-data:59880)
"""

import json
import logging
import os
import time
import uuid
import urllib.request
import urllib.error
from typing import Optional

from mqtt.base_publisher import BaseMqttPublisher

logger = logging.getLogger(__name__)

# 테이블 이름 → EdgeX 프로필 이름 매핑 (edgex/register_aiot_devices.py 와 일치)
_TABLE_PROFILE_MAP = {
    "t34950": "aiot-t34950-river",
    "t34955": "aiot-t34955-inclinometer",
    "t34957": "aiot-t34957-tilt-temp",
    "t34958": "aiot-t34958-imu",
}

# 정수(Int32)로 처리할 리소스 이름
_INT32_FIELDS = {"event_code"}


class EdgeXForwarder(BaseMqttPublisher):
    """TLV 파싱 결과를 EdgeX MQTT 발행 + Core Data HTTP push + Alert API HTTP POST 합니다.

    MQTT 토픽  : aiot/sensors/{dev_eui}/{table_name}
    Core Data : POST {core_data_url}/api/v3/event/{profile}/{device}/{source}
    Alert API : POST {alert_api_url}
    """

    def __init__(
        self,
        host: str,
        port: int = 1883,
        alert_api_url: Optional[str] = None,
        core_data_url: Optional[str] = None,
    ):
        super().__init__(
            broker=host,
            port=port,
            client_id_prefix="aiot-parser-forwarder",
            qos=0,
        )
        self._alert_api_url = alert_api_url or os.environ.get(
            "ALERT_API_URL", "http://cctv-alert-api:8000/api/sensor-readings"
        )
        self._core_data_url = (core_data_url or os.environ.get(
            "EDGEX_CORE_DATA_URL", "http://edgex-core-data:59880"
        )).rstrip("/")

    def publish(
        self,
        dev_eui: str,
        app_eui: str,
        device_id: str,
        table_name: str,
        data: dict,
        received_at: int,
    ) -> bool:
        """파싱된 센서 데이터를 aiot/sensors/{dev_eui}/{table_name} 으로 발행하고
        EdgeX Core Data 및 Alert API에 전송합니다.

        Returns:
            발행 성공 여부
        """
        if not self._ensure_connected():
            return False

        topic = f"aiot/sensors/{dev_eui}/{table_name}"
        sensor_payload = {
            "dev_eui":     dev_eui,
            "app_eui":     app_eui,
            "device_id":   device_id,
            "table":       table_name,
            "data":        {k: v for k, v in data.items() if k != "tableName"},
            "received_at": received_at,
        }
        payload_json = json.dumps(sensor_payload, default=str)

        # 1. MQTT 발행 (aiot/sensors/{dev_eui}/{table_name})
        mqtt_ok = False
        try:
            result = self._client.publish(topic, payload_json, qos=self.qos)
            if result.rc == 0:
                logger.debug("[EdgeXForwarder] MQTT 발행: %s", topic)
                mqtt_ok = True
            else:
                logger.warning("[EdgeXForwarder] MQTT 발행 실패 rc=%s: %s", result.rc, topic)
        except Exception as e:
            logger.error("[EdgeXForwarder] MQTT 발행 오류: %s", e)

        # 2. EdgeX Core Data 직접 push (EdgeX Event 포맷)
        self._post_to_core_data(device_id, table_name, sensor_payload["data"], received_at)

        # 3. HTTP POST → cctv-alert-api /api/sensor-readings (센서 로그 백업)
        self._post_to_alert_api(payload_json)

        return mqtt_ok

    # ──────────────────────────────────────────────────────────────
    # Core Data push
    # ──────────────────────────────────────────────────────────────

    def _build_edgex_event(
        self, device_id: str, table_name: str, data: dict, received_at: int
    ) -> Optional[dict]:
        """EdgeX v3 Event 딕셔너리를 생성합니다. 매핑된 프로필이 없으면 None 반환."""
        profile_name = _TABLE_PROFILE_MAP.get(table_name)
        if not profile_name:
            logger.debug("[EdgeXForwarder] 알 수 없는 테이블 '%s' — Core Data push 스킵", table_name)
            return None

        device_name = f"aiot-{device_id}"
        # received_at 은 밀리초 단위 — 나노초로 변환
        origin_ns = int(received_at) * 1_000_000

        readings = []
        for resource_name, raw_value in data.items():
            if resource_name == "tableName":
                continue
            try:
                if resource_name in _INT32_FIELDS:
                    value_type = "Int32"
                    value_str = str(int(raw_value))
                else:
                    value_type = "Float64"
                    value_str = str(float(raw_value))
            except (TypeError, ValueError):
                value_str = str(raw_value)
                value_type = "String"

            readings.append({
                "id":           str(uuid.uuid4()),
                "origin":       origin_ns,
                "deviceName":   device_name,
                "resourceName": resource_name,
                "profileName":  profile_name,
                "valueType":    value_type,
                "value":        value_str,
            })

        if not readings:
            return None

        return {
            "apiVersion": "v3",
            "event": {
                "apiVersion": "v3",
                "id":          str(uuid.uuid4()),
                "deviceName":  device_name,
                "profileName": profile_name,
                "sourceName":  table_name,
                "origin":      origin_ns,
                "readings":    readings,
            },
        }

    def _post_to_core_data(
        self, device_id: str, table_name: str, data: dict, received_at: int
    ) -> None:
        """EdgeX Core Data API에 Event를 POST합니다."""
        profile_name = _TABLE_PROFILE_MAP.get(table_name)
        if not profile_name:
            return

        event_body = self._build_edgex_event(device_id, table_name, data, received_at)
        if not event_body:
            return

        device_name = f"aiot-{device_id}"
        # EdgeX v3: POST /api/v3/event/{serviceName}/{profileName}/{deviceName}/{sourceName}
        url = f"{self._core_data_url}/api/v3/event/device-rest/{profile_name}/{device_name}/{table_name}"
        try:
            body = json.dumps(event_body).encode("utf-8")
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status not in (200, 201, 207):
                    logger.warning("[EdgeXForwarder] Core Data POST 응답: %s %s", resp.status, url)
                else:
                    logger.debug("[EdgeXForwarder] Core Data push 완료: %s", device_name)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")[:200]
            logger.warning("[EdgeXForwarder] Core Data HTTP 오류 %s: %s | %s", e.code, url, err_body)
        except urllib.error.URLError as e:
            logger.warning("[EdgeXForwarder] Core Data 연결 오류: %s", e)

    # ──────────────────────────────────────────────────────────────
    # Alert API
    # ──────────────────────────────────────────────────────────────

    def _post_to_alert_api(self, payload_json: str) -> None:
        if not self._alert_api_url:
            return
        try:
            data = payload_json.encode("utf-8")
            req = urllib.request.Request(
                self._alert_api_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status not in (200, 202):
                    logger.warning("[EdgeXForwarder] HTTP POST 응답 상태: %s", resp.status)
                else:
                    logger.debug("[EdgeXForwarder] HTTP POST 완료: %s", self._alert_api_url)
        except urllib.error.URLError as e:
            logger.warning("[EdgeXForwarder] HTTP POST 실패: %s", e)
        except Exception as e:
            logger.error("[EdgeXForwarder] HTTP POST 오류: %s", e)


def create_from_env() -> Optional["EdgeXForwarder"]:
    """환경변수에서 EdgeXForwarder 를 생성합니다.

    EDGEX_MQTT_HOST 가 없으면 None 반환 (EdgeX 발행 비활성화).
    """
    host = os.environ.get("EDGEX_MQTT_HOST", "").strip()
    if not host:
        logger.info("[EdgeXForwarder] EDGEX_MQTT_HOST 미설정 - EdgeX 발행 비활성화")
        return None

    port = int(os.environ.get("EDGEX_MQTT_PORT", "1883"))
    alert_api_url = os.environ.get(
        "ALERT_API_URL", "http://cctv-alert-api:8000/api/sensor-readings"
    )
    core_data_url = os.environ.get(
        "EDGEX_CORE_DATA_URL", "http://edgex-core-data:59880"
    )
    logger.info("[EdgeXForwarder] 대상 브로커: %s:%s", host, port)
    logger.info("[EdgeXForwarder] Alert API URL: %s", alert_api_url)
    logger.info("[EdgeXForwarder] Core Data URL: %s", core_data_url)
    return EdgeXForwarder(host=host, port=port, alert_api_url=alert_api_url, core_data_url=core_data_url)
