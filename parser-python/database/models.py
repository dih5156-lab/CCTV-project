"""
models.py
=========
Go 원본: aiot-tlv-parser/pkg/database/models.go

데이터베이스 테이블에 대응하는 데이터 모델 정의 모듈입니다.
Go의 struct(태그 포함) → Python dataclass 로 변환되었습니다.

센서 테이블 구조:
  t3      : 디바이스 장치 정보 (배터리, 펌웨어 등)
  t34950  : 하천 수위 / 유속 / 강수량
  t34952  : 침수 감지 수위
  t34954  : 온도 / 습도
  t34955  : 경사계 (각도 X, Y)
  t34956  : 화재 경보
  t34957  : 복합 요약1 (온도 + 경사)
  t34958  : 복합 요약2 (가속도 + 자이로 + 경사)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from time_utils import now_kst

# ──────────────────────────────────────────────
# 공통 기본 센서 데이터
# Go: type DefaultSensorData struct { AppEUI, DevEUI, DeviceID string; CreatedAt, ReceivedAt time.Time; ... }
# ──────────────────────────────────────────────

@dataclass
class DefaultSensorData:
    """
    모든 센서 테이블이 공유하는 기본 필드 (Go의 임베딩 struct 대응)
    Go: type DefaultSensorData struct { ... }

    필드 설명:
    - app_eui     : 애플리케이션 EUI (LoRaWAN 앱 식별자)
    - dev_eui     : 디바이스 EUI (LoRaWAN 디바이스 식별자)
    - device_id   : 내부 디바이스 ID
    - created_at  : 센서 데이터 생성 시각 (TLV에서 추출)
    - payload     : 원본 HEX 페이로드 문자열
    - channel     : LoRa 채널 번호
    - frequency   : LoRa 주파수 (Hz)
    - received_at : 게이트웨이 수신 시각
    """
    app_eui: str = ""
    dev_eui: str = ""
    device_id: str = ""
    created_at: datetime = field(default_factory=now_kst)
    payload: str = ""
    channel: int = 0
    frequency: int = 0
    received_at: datetime = field(default_factory=now_kst)


# ──────────────────────────────────────────────
# T3 : 디바이스 장치 정보
# Go: type T3 struct { ... Manufacturer, ModelNumber, FirmwareVersion string; ... }
# ──────────────────────────────────────────────

@dataclass
class T3:
    """
    LwM2M Object 3 - 디바이스 정보 테이블
    Go: type T3 struct { ...DefaultSensorData 필드들...; Manufacturer string; ... }

    TLV ID 매핑:
    - ID=0  → manufacturer (제조사)
    - ID=1  → model_number (모델명)
    - ID=3  → firmware_version (펌웨어 버전)
    - ID=4  → reboot (재부팅 여부)
    - ID=5  → factory_reset (공장 초기화 여부)
    - ID=9  → battery_level (배터리 잔량 %)
    - ID=11 → error_code (오류 코드)
    - ID=12 → reset_error_code (오류 코드 초기화)
    - ID=16 → supported_binding_and_modes (지원 바인딩 모드)
    - ID=18 → hardware_version (하드웨어 버전)
    - ID=20 → battery_status (배터리 상태)
    """
    app_eui: str = ""
    dev_eui: str = ""
    device_id: str = ""
    payload: str = ""
    channel: int = 0
    frequency: int = 0
    received_at: datetime = field(default_factory=now_kst)

    manufacturer: str = ""
    model_number: str = ""
    firmware_version: str = ""
    reboot: bool = False
    factory_reset: bool = False
    battery_level: int = 0
    error_code: int = 0
    reset_error_code: int = 0
    supported_binding_and_modes: str = ""
    hardware_version: str = ""
    battery_status: int = 0


# ──────────────────────────────────────────────
# T34950 : 하천 모니터링
# Go: type T34950 struct { DefaultSensorData; WaterLevel, FlowVelocity, RainFall float64; ReportingPeriod int }
# ──────────────────────────────────────────────

@dataclass
class T34950:
    """
    LwM2M Object 34950 - 하천 모니터링 데이터
    TLV ID 매핑:
    - ID=0     → water_level (수위, m)
    - ID=1     → flow_velocity (유속, m/s)
    - ID=2     → rain_fall (강수량, mm)
    - ID=26241 → reporting_period (보고 주기, 초) [원본: ms ÷ 1000]
    """
    sensor_data: DefaultSensorData = field(default_factory=DefaultSensorData)
    water_level: float = 0.0
    flow_velocity: float = 0.0
    rain_fall: float = 0.0
    reporting_period: int = 0


@dataclass
class T34952:
    """
    LwM2M Object 34952 - 침수 감지 데이터
    TLV ID 매핑:
    - ID=0     → flood_level (침수 수위, m)
    - ID=26241 → reporting_period (보고 주기, 초)
    """
    sensor_data: DefaultSensorData = field(default_factory=DefaultSensorData)
    flood_level: float = 0.0
    reporting_period: int = 0


@dataclass
class T34954:
    """
    LwM2M Object 34954 - 온습도 데이터
    TLV ID 매핑 (V0):
    - ID=0     → temperature (온도, ℃)
    - ID=1     → humidity (습도, %)
    - ID=26241 → reporting_period (보고 주기, 초)

    TLV ID 매핑 (V1 구버전):
    - ID=1 → temperature
    - ID=2 → reporting_period
    - ID=4 → created_at (Unix ms → *1000)
    """
    sensor_data: DefaultSensorData = field(default_factory=DefaultSensorData)
    temperature: float = 0.0
    humidity: float = 0.0
    reporting_period: int = 0


@dataclass
class T34955:
    """
    LwM2M Object 34955 - 경사계 데이터
    TLV ID 매핑 (V0):
    - ID=0     → angle_x (X축 각도, °)
    - ID=1     → angle_y (Y축 각도, °)
    - ID=2     → reporting_angle_threshold (각도 임계값, °)
    - ID=3     → relative_angle_value_reset (상대 각도 초기화)
    - ID=26241 → reporting_period
    """
    sensor_data: DefaultSensorData = field(default_factory=DefaultSensorData)
    angle_x: float = 0.0
    angle_y: float = 0.0
    reporting_angle_threshold: float = 0.0
    relative_angle_value_reset: float = 0.0
    reporting_period: int = 0


@dataclass
class T34956:
    """
    LwM2M Object 34956 - 화재 경보 데이터
    TLV ID 매핑:
    - ID=0     → fire_alarm (화재 감지 여부)
    - ID=26241 → reporting_period
    """
    sensor_data: DefaultSensorData = field(default_factory=DefaultSensorData)
    fire_alarm: bool = False
    reporting_period: int = 0


@dataclass
class T34957:
    """
    LwM2M Object 34957 - 복합 요약1 (온도 + 경사)
    TLV ID 매핑 (V0):
    - ID=0 → temperature
    - ID=1 → angle_x
    - ID=2 → angle_y
    - ID=3 → event_code (이벤트 발생 여부)

    V0 특수 로직: angle_x, angle_y 모두 존재하면 event_code = 1
    """
    sensor_data: DefaultSensorData = field(default_factory=DefaultSensorData)
    temperature: float = 0.0
    angle_x: float = 0.0
    angle_y: float = 0.0
    event_code: bool = False


@dataclass
class T34958:
    """
    LwM2M Object 34958 - 복합 요약2 (가속도 + 자이로 + 경사)
    TLV ID 매핑 (V0):
    - ID=0 → acc_x   (X축 가속도)
    - ID=1 → acc_y   (Y축 가속도)
    - ID=2 → acc_z   (Z축 가속도)
    - ID=3 → gyro_x  (X축 자이로)
    - ID=4 → gyro_y  (Y축 자이로)
    - ID=5 → gyro_z  (Z축 자이로)
    - ID=6 → angle_x (X축 각도)
    - ID=7 → angle_y (Y축 각도)
    - ID=8 → event_code
    """
    sensor_data: DefaultSensorData = field(default_factory=DefaultSensorData)
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    angle_x: float = 0.0
    angle_y: float = 0.0
    event_code: bool = False


# ──────────────────────────────────────────────
# SensorData : 통합 센서 원시 데이터 레코드
# Go: type SensorData struct { DefaultSensorData; ObjectID string; PayloadTLV json.RawMessage; IsEvent bool }
# ──────────────────────────────────────────────

@dataclass
class SensorData:
    """
    sensor_data 테이블 레코드 - 모든 TLV 데이터의 공통 원시 저장소
    Go: type SensorData struct { DefaultSensorData; ObjectID string; PayloadTLV json.RawMessage; IsEvent bool }

    - object_id  : 테이블 번호 문자열 (예: "34950")
    - payload_tlv: 파싱된 TLV 데이터의 JSON (dict)
    - is_event   : 이벤트 여부 (event_code 필드 기반)
    """
    sensor_data: DefaultSensorData = field(default_factory=DefaultSensorData)
    object_id: str = ""
    payload_tlv: Dict[str, Any] = field(default_factory=dict)
    is_event: bool = False


@dataclass
class QueryResult:
    """
    SELECT 쿼리 결과
    Go: type QueryResult struct { Rows []map[string]interface{} }
    """
    rows: list = field(default_factory=list)


@dataclass
class Notification:
    """
    알림 테이블 레코드
    Go: type Notification struct { UserID, AppEUI, DevEUI, DeviceID, ObjectID string }
    """
    user_id: str = ""
    app_eui: str = ""
    dev_eui: str = ""
    device_id: str = ""
    object_id: str = ""
