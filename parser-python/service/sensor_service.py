"""
service/sensor_service.py
=========================
Go 원본: aiot-tlv-parser/pkg/service/sensor_service.go

MQTT로 수신된 센서 페이로드를 처리하는 핵심 서비스입니다.
Base64 디코딩 → TLV 파싱 → 테이블별 데이터 구조 생성 → DB 큐 추가 → 이벤트 처리

Go의 타입 어설션(type assertion) → Python의 isinstance() + dict.get() 로 변환됩니다.
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from database.connection import DB
from database.processor import DataProcessor
from database.models import (
    DefaultSensorData, Notification, SensorData,
    T3, T34950, T34952, T34954, T34955, T34956, T34957, T34958,
)
from tlv.parser import Parser
from service.device_info_service import DeviceInfoService
from service.event_service import EventService
from mqtt.interfaces import SensorDataProcessor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 메시지 데이터 구조
# Go: type MessageData struct { ... }
# ──────────────────────────────────────────────

@dataclass
class GatewayInfo:
    """
    Go: type GatewayInfo struct { GwEUI, ChannelPlan string; Latitude, Longitude float64; Altitude int }
    """
    gw_eui: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: int = 0
    channel_plan: str = ""


@dataclass
class RxMetadata:
    """
    Go: type RxMetadata struct { Channel, Frequency, RSSI int; SNR float64; Time, Timestamp int64; ... }
    """
    gateway_info: GatewayInfo = field(default_factory=GatewayInfo)
    modulation: str = ""
    data_rate: str = ""
    coding_rate: str = ""
    timestamp: int = 0
    time: int = 0
    gps_time: int = 0
    channel: int = 0
    frequency: int = 0
    rssi: int = 0
    snr: float = 0.0
    gw_recv_time: str = ""


@dataclass
class MessageData:
    """
    MQTT JSON 메시지 전체 구조
    Go: type MessageData struct { MessageID, AppEUI, DevEUI, Payload string; RxMetadata []RxMetadata; ... }
    """
    message_id: str = ""
    app_eui: str = ""
    dev_eui: str = ""
    payload: str = ""
    is_confirmed: bool = False
    f_port: int = 0
    f_cnt_up: int = 0
    rx_metadata: List[RxMetadata] = field(default_factory=list)


# ──────────────────────────────────────────────
# SensorService
# Go: type SensorService struct { db *database.DB; dataProcessor *database.DataProcessor; ... }
# ──────────────────────────────────────────────

class SensorService(SensorDataProcessor):
    """
    센서 데이터 처리 서비스 (MQTT SensorDataProcessor 구현체)
    Go: type SensorService struct { ... }
    """

    def __init__(self, db: DB, edgex_forwarder=None):
        """
        Go: func NewSensorService(db *database.DB) *SensorService

        Args:
            edgex_forwarder: EdgeXForwarder 인스턴스 (없으면 EdgeX 발행 비활성화)
        """
        from datetime import timedelta
        self._db = db
        self._data_processor = DataProcessor(db, threshold=30, interval=timedelta(seconds=1))
        self._tlv_parser = Parser()
        self.device_info = DeviceInfoService()
        self.event_service = EventService(db)
        self._edgex_forwarder = edgex_forwarder

    def process_sensor_data(
        self,
        app_id: str,
        dev_eui: str,
        payload: str,
        channel: int,
        frequency: int,
        received_at: int,
    ) -> None:
        """
        MQTT 수신 데이터 처리 메인 로직
        Go: func (s *SensorService) ProcessSensorData(appID, devEUI, message string, ...) error

        처리 흐름:
          1. Base64 디코딩
          2. DeviceID 조회
          3. TLV 파싱 (오프셋 8)
          4. 테이블별 데이터 구조 생성 및 큐 추가
          5. SensorData (통합 원시 레코드) 큐 추가
          6. 이벤트 여부 확인 → 알림 큐 추가
        """
        # 1. Base64 디코딩
        # Go: base64.StdEncoding.DecodeString(message)
        try:
            decoded_payload = base64.b64decode(payload)
        except Exception as e:
            logger.error(f"Failed to decode base64 payload for devEUI {dev_eui}: {e}")
            return

        # 2. DeviceID 조회
        device_id = self.device_info.get_device_id(dev_eui)
        if not device_id:
            logger.error(f"Device ID not found for devEUI: {dev_eui}")
            raise ValueError(f"device ID not found for devEUI: {dev_eui}")

        # 3. TLV 파싱 (Go: s.tlvParser.DecodeLwM2MTLV(decodedPayload, 8))
        try:
            parsed_tlv = self._tlv_parser.decode_lwm2m_tlv(decoded_payload, 8)
        except Exception as e:
            logger.error(f"Failed to parse TLV for devEUI {dev_eui}: {e}")
            return

        if not parsed_tlv:
            logger.error(f"No TLV data parsed for devEUI: {dev_eui}")
            raise ValueError(f"no TLV data parsed for devEUI: {dev_eui}")

        payload_hex = decoded_payload.hex()
        received_dt = _unix_milli_to_utc(received_at)
        created_dt = _get_created_at(parsed_tlv.data)

        # 파싱 성공 로그
        data_fields = {k: v for k, v in parsed_tlv.data.items() if k != "tableName"}
        logger.info(f"[{parsed_tlv.table_name}] devEUI={dev_eui} deviceID={device_id} data={data_fields}")

        # 공통 센서 데이터 (Go: DefaultSensorData)
        base = DefaultSensorData(
            app_eui=app_id,
            dev_eui=dev_eui,
            device_id=device_id,
            created_at=created_dt,
            payload=payload_hex,
            channel=channel,
            frequency=frequency,
            received_at=received_dt,
        )

        # 4. 테이블별 데이터 구조 생성 및 큐 추가
        # Go: switch parsedTLV.TableName { case "t3": ... case "t34950": ... }
        table = parsed_tlv.table_name
        d = parsed_tlv.data

        if table == "t3":
            self._data_processor.add_data(T3(
                app_eui=app_id, dev_eui=dev_eui, device_id=device_id,
                payload=payload_hex, channel=channel, frequency=frequency,
                received_at=received_dt,
                manufacturer=_get_str(d, "manufacturer"),
                model_number=_get_str(d, "model_number"),
                firmware_version=_get_str(d, "firmware_version"),
                reboot=_get_bool(d, "reboot"),
                factory_reset=_get_bool(d, "factory_reset"),
                battery_level=_get_int(d, "battery_level_pct"),
                error_code=_get_int(d, "error_code"),
                reset_error_code=_get_int(d, "reset_error_code"),
                supported_binding_and_modes=_get_str(d, "supported_binding_and_modes"),
                hardware_version=_get_str(d, "hardware_version"),
                battery_status=_get_int(d, "battery_status"),
            ))
        elif table == "t34950":
            self._data_processor.add_data(T34950(
                sensor_data=base,
                water_level=_get_float(d, "water_level_m"),
                flow_velocity=_get_float(d, "flow_velocity_mps"),
                rain_fall=_get_float(d, "rain_fall_mm"),
                reporting_period=_get_int(d, "reporting_period_s"),
            ))
        elif table == "t34952":
            self._data_processor.add_data(T34952(
                sensor_data=base,
                flood_level=_get_float(d, "flood_level_m"),
                reporting_period=_get_int(d, "reporting_period_s"),
            ))
        elif table == "t34954":
            self._data_processor.add_data(T34954(
                sensor_data=base,
                temperature=_get_float(d, "temperature_c"),
                humidity=_get_float(d, "humidity_pct"),
                reporting_period=_get_int(d, "reporting_period_s"),
            ))
        elif table == "t34955":
            self._data_processor.add_data(T34955(
                sensor_data=base,
                angle_x=_get_float(d, "angle_x_deg"),
                angle_y=_get_float(d, "angle_y_deg"),
                reporting_angle_threshold=_get_float(d, "reporting_angle_threshold_deg"),
                relative_angle_value_reset=_get_float(d, "relative_angle_value_reset"),
                reporting_period=_get_int(d, "reporting_period_s"),
            ))
        elif table == "t34956":
            self._data_processor.add_data(T34956(
                sensor_data=base,
                fire_alarm=_get_bool(d, "fire_alarm"),
                reporting_period=_get_int(d, "reporting_period_s"),
            ))
        elif table == "t34957":
            self._data_processor.add_data(T34957(
                sensor_data=base,
                temperature=_get_float(d, "temperature_c"),
                angle_x=_get_float(d, "angle_x_deg"),
                angle_y=_get_float(d, "angle_y_deg"),
                event_code=_get_bool(d, "event_code"),
            ))
        elif table == "t34958":
            self._data_processor.add_data(T34958(
                sensor_data=base,
                acc_x=_get_float(d, "acc_x_g"),
                acc_y=_get_float(d, "acc_y_g"),
                acc_z=_get_float(d, "acc_z_g"),
                gyro_x=_get_float(d, "gyro_x_dps"),
                gyro_y=_get_float(d, "gyro_y_dps"),
                gyro_z=_get_float(d, "gyro_z_dps"),
                angle_x=_get_float(d, "angle_x_deg"),
                angle_y=_get_float(d, "angle_y_deg"),
                event_code=_get_bool(d, "event_code"),
            ))
        else:
            logger.warning(f"Unknown table name: {table}")

        # 5. 통합 원시 센서 데이터 레코드 큐 추가
        # Go: sensorData := database.SensorData{...}
        sensor_data = SensorData(
            sensor_data=DefaultSensorData(
                app_eui=app_id,
                dev_eui=dev_eui,
                device_id=device_id,
                created_at=created_dt,
                received_at=received_dt,
                payload=payload_hex,
                channel=channel,
                frequency=frequency,
            ),
            object_id=table.lstrip("t"),  # "t34950" → "34950"
            payload_tlv=d,
            is_event=_get_bool(d, "event_code"),
        )
        self._data_processor.add_data(sensor_data)

        # 5-b. EdgeX MQTT 발행 (EDGEX_MQTT_HOST 설정 시)
        if self._edgex_forwarder is not None:
            self._edgex_forwarder.publish(
                dev_eui=dev_eui,
                app_eui=app_id,
                device_id=device_id,
                table_name=table,
                data=d,
                received_at=received_at,
            )

        # 6. 이벤트 처리 (Go: if isEvent(parsedTLV.Data) { ... })
        if _is_event(d):
            user_ids, found = self.event_service.get_user_ids_by_app_eui(app_id)
            if found and user_ids:
                for user_id in user_ids:
                    self.event_service.add_notification_to_queue(Notification(
                        user_id=user_id,
                        app_eui=app_id,
                        dev_eui=dev_eui,
                        device_id=device_id,
                        object_id=table.lstrip("t"),
                    ))
            else:
                logger.warning(f"No user IDs found for appID: {app_id}")

    def close(self) -> None:
        """
        서비스 리소스 정리
        Go: func (s *SensorService) Close()
        """
        self._data_processor.close()
        self.event_service.close()


# ──────────────────────────────────────────────
# 헬퍼 함수들 (TLV 딕셔너리에서 타입별 값 추출)
# Go: getFloat64FromTLV, getStringFromTLV, getIntFromTLV, getBoolFromTLV
# ──────────────────────────────────────────────

def _get_float(data: dict, key: str) -> float:
    """
    Go: func getFloat64FromTLV(data map[string]interface{}, key string) float64
    """
    val = data.get(key)
    if isinstance(val, float):
        return val
    if isinstance(val, (int, bool)):
        return float(val)
    return 0.0


def _get_str(data: dict, key: str) -> str:
    """
    Go: func getStringFromTLV(data map[string]interface{}, key string) string
    """
    val = data.get(key)
    if isinstance(val, str):
        return val
    return ""


def _get_int(data: dict, key: str) -> int:
    """
    Go: func getIntFromTLV(data map[string]interface{}, key string) int
    """
    val = data.get(key)
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, float):
        return int(val)
    return 0


def _get_bool(data: dict, key: str) -> bool:
    """
    Go: func getBoolFromTLV(data map[string]interface{}, key string) bool
    """
    val = data.get(key)
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val == 1
    if isinstance(val, float):
        return val == 1.0
    return False


def _must_marshal_json(v: Any) -> bytes:
    """
    JSON 직렬화 (실패 시 빈 객체 반환)
    Go: func mustMarshalJSON(v interface{}) []byte
    """
    try:
        return json.dumps(v).encode("utf-8")
    except Exception as e:
        logger.error(f"Failed to marshal JSON: {e}")
        return b"{}"


def _get_created_at(data: dict) -> datetime:
    """
    TLV 데이터에서 created_at 타임스탬프 추출
    Go: func getCreatedAt(data map[string]interface{}) time.Time
    """
    created_at = data.get("created_at")
    if isinstance(created_at, int):
        # Go: time.UnixMilli(timestamp).UTC()
        return _unix_milli_to_utc(created_at)
    return datetime.now(timezone.utc)


def _unix_milli_to_utc(ms: int) -> datetime:
    """Unix 밀리초 → UTC datetime 변환"""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _is_event(data: dict) -> bool:
    """
    이벤트 여부 확인
    Go: func isEvent(data map[string]interface{}) bool
    """
    event_code = data.get("event_code")
    if isinstance(event_code, bool):
        return event_code
    if isinstance(event_code, int):
        return event_code == 1
    return False
