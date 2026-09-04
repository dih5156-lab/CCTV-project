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
import threading
import urllib.error
import urllib.request
import uuid
from typing import Optional

from database.edgex_outbox import EdgeXOutbox

from mqtt.base_publisher import BaseMqttPublisher

logger = logging.getLogger(__name__)

# 테이블 이름 → EdgeX 프로필 이름 매핑 (edgex/register_aiot_devices.py 와 일치)
_TABLE_PROFILE_MAP = {
    "t34950": "aiot-t34950-river",
    "t34955": "aiot-t34955-inclinometer",
    "t34957": "aiot-t34957-tilt-temp",
    "t34958": "aiot-t34958-imu",
}

_TABLE_RESOURCE_MAP = {
    "t34950": {
        "water_level_m": "water_level",
        "flow_velocity_mps": "flow_velocity",
        "rain_fall_mm": "rain_fall",
        "reporting_period_s": "reporting_period",
    },
    "t34955": {
        "angle_x_deg": "angle_x",
        "angle_y_deg": "angle_y",
        "reporting_period_s": "reporting_period",
        "reporting_angle_threshold_deg": "reporting_angle_threshold",
    },
    "t34957": {
        "temperature_c": "temperature",
        "angle_x_deg": "angle_x",
        "angle_y_deg": "angle_y",
        "event_code": "event_code",
    },
    "t34958": {
        "acc_x_g": "acc_x",
        "acc_y_g": "acc_y",
        "acc_z_g": "acc_z",
        "gyro_x_dps": "gyro_x",
        "gyro_y_dps": "gyro_y",
        "gyro_z_dps": "gyro_z",
        "event_code": "event_code",
    },
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
        outbox_db_path: Optional[str] = None,
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

        # ── 아웃박스(로컬 SQLite) 초기화 ──
        self._outbox = EdgeXOutbox(outbox_db_path)

        # ── 재전송 워커 ──
        self._retry_stop = threading.Event()
        _retry_interval = int(os.environ.get("EDGEX_OUTBOX_RETRY_INTERVAL", "30"))
        self._retry_interval = max(5, _retry_interval)
        self._retry_thread = threading.Thread(
            target=self._retry_worker,
            daemon=True,
            name="edgex-outbox-retry",
        )
        self._retry_thread.start()

    def publish(
        self,
        dev_eui: str,
        app_eui: str,
        device_id: str,
        table_name: str,
        data: dict,
        received_at: int,
        uplink_metadata: Optional[dict] = None,
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
        if uplink_metadata:
            sensor_payload["uplink"] = uplink_metadata
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

        resource_map = _TABLE_RESOURCE_MAP[table_name]
        readings = []
        for raw_resource_name, raw_value in data.items():
            if raw_resource_name == "tableName":
                continue
            resource_name = resource_map.get(raw_resource_name)
            if resource_name is None:
                logger.debug(
                    "[EdgeXForwarder] 프로파일에 없는 리소스 제외: %s.%s",
                    table_name,
                    raw_resource_name,
                )
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
        """EdgeX Core Data API에 Event를 POST합니다.

        전송 전 아웃박스에 'pending' 으로 저장하고, 성공 시 'sent' 로 업데이트합니다.
        실패한 항목은 백그라운드 워커가 주기적으로 재시도합니다.
        """
        profile_name = _TABLE_PROFILE_MAP.get(table_name)
        if not profile_name:
            return

        event_body = self._build_edgex_event(device_id, table_name, data, received_at)
        if not event_body:
            return

        device_name = f"aiot-{device_id}"
        url = (
            f"{self._core_data_url}/api/v3/event/device-rest"
            f"/{profile_name}/{device_name}/{table_name}"
        )

        # 아웃박스에 먼저 저장 (pending)
        row_id = self._outbox.save_pending(device_id, table_name, url, event_body)

        # Core Data 전송 시도
        if self._do_post_edgex_event(url, event_body):
            self._outbox.mark_sent(row_id)
        # 실패 시 pending 유지 → retry_worker 가 재시도

    def _do_post_edgex_event(self, url: str, event_body: dict) -> bool:
        """EdgeX Core Data POST 실행. 성공 시 True 반환."""
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
                    return False
                else:
                    logger.debug("[EdgeXForwarder] Core Data push 완료: %s", url)
                    return True
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")[:200]
            logger.warning("[EdgeXForwarder] Core Data HTTP 오류 %s: %s | %s", e.code, url, err_body)
        except urllib.error.URLError as e:
            logger.warning("[EdgeXForwarder] Core Data 연결 오류: %s", e)
        return False

    # ──────────────────────────────────────────────────────────────
    # 아웃박스 재전송 워커
    # ──────────────────────────────────────────────────────────────

    def _retry_worker(self) -> None:
        """백그라운드에서 pending 아웃박스 항목을 주기적으로 재전송합니다."""
        logger.info("[Outbox] 재전송 워커 시작 (간격: %ds)", self._retry_interval)
        while not self._retry_stop.wait(self._retry_interval):
            try:
                self._retry_pending_once()
            except Exception as e:
                logger.error("[Outbox] 재전송 워커 오류: %s", e, exc_info=True)

    def _retry_pending_once(self) -> None:
        """pending 항목 한 번 순회하여 재전송 시도."""
        # TTL/최대재시도 초과 항목 정리
        expired = self._outbox.expire_old_failed()
        if expired:
            logger.info("[Outbox] TTL/재시도 초과로 만료 처리: %d건", expired)

        rows = self._outbox.get_pending()
        if not rows:
            return

        sent_count = 0
        for row in rows:
            row_id    = row["id"]
            url       = row["core_data_url"]
            event     = row["edgex_event"]
            retry_cnt = row["retry_count"]

            self._outbox.increment_retry(row_id)
            if self._do_post_edgex_event(url, event):
                self._outbox.mark_sent(row_id)
                sent_count += 1
            else:
                logger.debug(
                    "[Outbox] 재전송 실패 id=%s, retry=%d", row_id, retry_cnt + 1
                )

        if sent_count:
            logger.info("[Outbox] 재전송 성공: %d건 / %d건", sent_count, len(rows))

    def stop(self) -> None:
        """재전송 워커 및 아웃박스 종료."""
        self._retry_stop.set()
        if self._retry_thread and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=3)
        self._outbox.close()

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
