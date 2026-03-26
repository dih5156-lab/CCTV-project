"""중앙화된 애플리케이션 설정 모듈"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union


# 프로젝트 루트 디렉토리 (src/config에서 2단계 상위)
PROJECT_ROOT = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)


def _identity(value: str) -> str:
    return value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EnvOverride:
    key: str
    path: tuple[str, ...]
    parser: Callable[[str], object] = _identity


MODEL_CANDIDATES: dict[str, tuple[str, tuple[Union[str, Path], ...]]] = {
    "helmet_model": (
        "helmet",
        (
            # TensorRT (Jetson 우선)
            PROJECT_ROOT / "models/helmet_model_ver0.6.engine",
            PROJECT_ROOT / "models/helmet_model_ver0.6.pt",
            PROJECT_ROOT / "helmet_model_ver0.6.pt",
        ),
    ),
    "pose_model": (
        "pose",
        (
            # TensorRT (Jetson 우선)
            PROJECT_ROOT / "models/yolov8n-pose.engine",
            PROJECT_ROOT / "models/yolov8n-pose.pt",
            PROJECT_ROOT / "yolov8n-pose.pt",
            "yolov8n-pose.pt",
        ),
    ),
    "person_model": (
        "person",
        (
            # TensorRT (Jetson 우선)
            PROJECT_ROOT / "models/yolov8n.engine",
            PROJECT_ROOT / "models/yolov8s.engine",
            PROJECT_ROOT / "models/yolov8n.pt",
            PROJECT_ROOT / "models/yolov8s.pt",
            PROJECT_ROOT / "yolov8n.pt",
            PROJECT_ROOT / "yolov8s.pt",
            "yolov8n.pt",
            "yolov8s.pt",
        ),
    ),
}


ENV_OVERRIDES: tuple[EnvOverride, ...] = (
    EnvOverride("HELMET_MODEL_PATH", ("models", "helmet_model")),
    EnvOverride("PERSON_MODEL_PATH", ("models", "person_model")),
    EnvOverride("POSE_MODEL_PATH", ("models", "pose_model")),
    EnvOverride("DEVICE", ("detection", "device")),
    EnvOverride("MQTT_BROKER", ("mqtt", "broker")),
    EnvOverride("MQTT_PORT", ("mqtt", "port"), parser=lambda v: int(v.strip())),
    EnvOverride("MQTT_TOPIC_PREFIX", ("mqtt", "topic_prefix")),
    EnvOverride("DISPLAY_ENABLED", ("display",), parser=_parse_bool),
    EnvOverride("ZONE_DETECTION_ENABLED", ("zone_detection",), parser=_parse_bool),
    EnvOverride("COLLECT_DATASET", ("collect_dataset",), parser=_parse_bool),
    # EdgeX
    EnvOverride("EDGEX_METADATA_URL", ("edgex", "metadata_url")),
    EnvOverride("EDGEX_DATA_URL", ("edgex", "data_url")),
    EnvOverride("EDGEX_MQTT_BROKER", ("edgex", "mqtt_broker")),
    EnvOverride("EDGEX_MQTT_PORT", ("edgex", "mqtt_port"), parser=lambda v: int(v.strip())),
    EnvOverride("EDGEX_REDIS_HOST", ("edgex", "redis_host")),
    EnvOverride("EDGEX_REDIS_PORT", ("edgex", "redis_port"), parser=lambda v: int(v.strip())),
    EnvOverride("EDGEX_SERVICE_BASE_URL", ("edgex", "service_base_url")),
    # ActionBridge
    EnvOverride("ACTION_MQTT_BROKER", ("action", "mqtt_broker")),
    EnvOverride("ACTION_MQTT_PORT", ("action", "mqtt_port"), parser=lambda v: int(v.strip())),
    EnvOverride("ACTION_REST_HOST", ("action", "rest_host")),
    EnvOverride("ACTION_REST_PORT", ("action", "rest_port"), parser=lambda v: int(v.strip())),
    # 카메라 / RTSP
    EnvOverride("RTSP_BUFFER_SIZE", ("camera", "buffer_size"), parser=lambda v: int(v.strip())),
)

@dataclass
class ModelPaths:
    """모델 파일 경로 관리"""

    helmet_model: Optional[str] = None
    person_model: Optional[str] = None
    pose_model: Optional[str] = None

    def __post_init__(self) -> None:
        """누락된 모델 경로를 자동으로 보강"""

        for attr, (label, candidates) in MODEL_CANDIDATES.items():
            if getattr(self, attr) is None:
                setattr(self, attr, self._resolve_path(label, candidates))

    @staticmethod
    def _resolve_path(label: str, candidates: Sequence[Union[str, Path]]) -> Optional[str]:
        """후보 경로 중 존재하는 첫 번째를 반환하고, 없으면 첫 문자열을 보존"""

        fallback: Optional[str] = None
        for candidate in candidates:
            if candidate is None:
                continue

            if isinstance(candidate, Path):
                candidate_path = candidate.expanduser()
                if candidate_path.exists():
                    logger.debug("모델 경로 감지 - %s: %s", label, candidate_path)
                    return str(candidate_path)
                continue

            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                logger.debug("모델 경로 감지 - %s: %s", label, candidate_path)
                return str(candidate_path)

            if fallback is None:
                fallback = candidate

        if fallback:
            logger.debug("모델 경로 Fallback 사용 - %s: %s", label, fallback)
        return fallback
    
    def validate(self) -> bool:
        """모델 파일 존재 여부 및 크기 검증
        
        pose_model이 있으면 person_model은 선택 사항으로 취급합니다.
        """

        def _validate(label: str, path: Optional[str], required: bool = True) -> bool:
            if not path:
                if required:
                    logger.warning("%s 모델 경로가 설정되지 않았습니다", label)
                else:
                    logger.debug("%s 모델 미설정 (선택 사항)", label)
                return not required

            candidate = Path(path).expanduser()
            if not candidate.exists():
                if required:
                    logger.warning("%s 모델을 찾을 수 없습니다: %s", label, candidate)
                else:
                    logger.debug("%s 모델 파일 없음 (선택 사항): %s", label, candidate)
                return not required
            if candidate.stat().st_size <= 0:
                logger.error("%s 모델 파일이 비어 있습니다: %s", label, candidate)
                return False
            return True

        # pose_model이 있으면 person_model은 선택 사항
        pose_ok = _validate("Pose", self.pose_model, required=True)
        person_required = not pose_ok
        results = [
            _validate("헬멧", self.helmet_model),
            pose_ok,
            _validate("Person", self.person_model, required=person_required),
        ]
        return all(results)


@dataclass
class EdgeXConfig:
    """EdgeX Foundry 연동 설정"""
    metadata_url: str = "http://localhost:59881"
    data_url: str = "http://localhost:59880"
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    redis_host: str = "edgex-redis"
    redis_port: int = 6379
    service_base_url: str = "http://cctv-device-service:59986"


@dataclass
class ActionBridgeConfig:
    """ActionBridge 서비스 연결 설정"""
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    rest_host: str = "0.0.0.0"
    rest_port: int = 8080


@dataclass
class MqttConfig:
    """MQTT 발행 설정 (AI 엔진 출력 채널)"""
    broker: str = "localhost"
    port: int = 1883
    topic_prefix: str = "cctv/ai/events"
    client_id_prefix: str = "cctv-ai-engine"
    qos: int = 0
    retain: bool = False


@dataclass
class CameraConfig:
    """카메라/RTSP 설정"""
    reconnect_interval: int = 5
    max_retries: int = 5
    read_timeout: int = 10
    buffer_size: int = 2  # RTSP 프레임 버퍼 크기
                          # 1: 최신 프레임 유지 (지연 최소화, 단 지터 환경에서 드롭 증가)
                          # 2~3: 네트워크 지터 완충 권장 (일반 CCTV 환경 기본값)


@dataclass
class DetectionConfig:
    """객체 감지 설정"""
    helmet_confidence: float = 0.7  # 헬멧 감지 신뢰도 (0.0~1.0)
    person_confidence: float = 0.4  # 사람 감지 신뢰도 (0.0~1.0) - 원거리 사람 감지 위한 낮은 임계값
    pose_confidence: float = 0.5  # 사람 감지 신뢰도 (0.0~1.0)
    device: str = "cpu"  # 계산 장치 (cpu 또는 cuda)
    target_fps: int = 30  # 목표 프레임율
    iou_threshold: float = 0.3  # 교차합집합(IoU) 임계값
    max_helmet_size: int = 500  # 최대 헬멧 크기
    fall_angle_threshold: float = 45.0  # 낙상 감지 각도 임계값
    fall_height_ratio: float = 0.3  # 낙상 감지 높이 비율
    
    def __post_init__(self):
        """신뢰도 및 임계값 검증"""
        # 신뢰도 검증 (0.0~1.0)
        if not 0.0 <= self.helmet_confidence <= 1.0:
            raise ValueError(f"헬멧 신뢰도는 0.0~1.0 사이여야 합니다. 입력값: {self.helmet_confidence}")
        if not 0.0 <= self.person_confidence <= 1.0:
            raise ValueError(f"사람 신뢰도는 0.0~1.0 사이여야 합니다. 입력값: {self.person_confidence}")
        if not 0.0 <= self.pose_confidence <= 1.0:
            raise ValueError(f"Pose 신뢰도는 0.0~1.0 사이여야 합니다. 입력값: {self.pose_confidence}")
        
        # IoU 임계값 검증
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"IoU 임계값은 0.0~1.0 사이여야 합니다. 입력값: {self.iou_threshold}")
        
        # FPS 및 장치 검증
        if self.target_fps <= 0:
            raise ValueError(f"목표 FPS는 양수여야 합니다. 입력값: {self.target_fps}")
        # "cpu", "cuda", "cuda:0", "cuda:1" ... 모두 허용 (Jetson 다중 GPU 지원)
        if not (self.device in ["cpu", "cuda"] or self.device.startswith("cuda:")):
            raise ValueError(f"장치는 'cpu', 'cuda', 'cuda:N' 형식이어야 합니다. 입력값: {self.device}")


@dataclass
class EventConfig:
    """이벤트 처리 설정"""
    debounce_enabled: bool = True
    debounce_seconds: float = 3.0
    queue_max_size: int = 500
    event_retention_hours: int = 24
    cleanup_interval: int = 3600
    # 낙상 지속 감지 설정
    fall_sustained_seconds: float = 10.0   # 낙상 상태가 이 시간(초) 이상 유지되어야 전송
    fall_resend_cooldown: float = 60.0     # 낙상 알림 전송 후 재전송 대기 시간(초)
    fall_gap_reset_seconds: float = 2.0    # 이 시간 이상 낙상 미감지 시 지속 타이머 초기화
    # 헬멧 미착용(head) 감지 설정
    head_resend_cooldown: float = 30.0     # 동일 객체 head 재전송 최소 간격(초)
    head_gap_reset_seconds: float = 5.0    # 이 시간 이상 미감지 시 상태 리셋 → 재등장은 즉시 전송


@dataclass
class ProcessingConfig:
    """비디오 처리 설정"""
    # 스레드 및 연결 관리
    thread_join_timeout: int = 5  # 스레드 종료 대기 시간 (초)
    camera_reconnect_delay: float = 0.1  # 카메라 재연결 지연 시간
    consecutive_failure_threshold: int = 5  # 연속 실패 임계값
    queue_warning_threshold: float = 0.8  # 큐 경고 임계값
    
    # 추론 및 성능
    fall_inference_interval: int = 7  # 낙상 추론 간격
    frame_skip: int = 8  # 프레임 스킵 (8프레임마다 1회 추론)
    
    # 누적 판정 방식 (오탐 필터링)
    cumulative_detection_enabled: bool = True  # 누적 판정 활성화 여부
    detection_history_size: int = 3  # 감지 이력 크기 (최근 3프레임 - 더 비중단적으로 반응)
    violation_threshold: int = 2  # 위반 임계값 (3개 중 2개 이상 위반 시 경고)
    
    # 추적 기반 오탐 필터링
    min_track_frames: int = 2  # 최소 추적 프레임 수 (연속 감지되어야 유효한 객체로 인정)


@dataclass
class AppConfig:
    """애플리케이션 메인 설정"""

    models: Optional[ModelPaths] = None
    mqtt: Optional[MqttConfig] = None
    camera: Optional[CameraConfig] = None
    detection: Optional[DetectionConfig] = None
    events: Optional[EventConfig] = None
    processing: Optional[ProcessingConfig] = None
    edgex: Optional[EdgeXConfig] = None
    action: Optional[ActionBridgeConfig] = None
    
    # 기능 활성화 플래그
    display: bool = False  # 화면 표시 여부
    zone_detection: bool = False  # 구역 감지 여부
    collect_dataset: bool = False  # 데이터셋 수집 여부
    
    # 파일 경로
    dataset_dir: str = str(PROJECT_ROOT / "collected_data")  # 데이터셋 디렉토리
    zones_config: str = str(PROJECT_ROOT / "zones_config.json")  # 구역 설정 파일
    
    def __post_init__(self) -> None:
        """기본 설정 객체 초기화"""

        self.models = self.models or ModelPaths()
        self.mqtt = self.mqtt or MqttConfig()
        self.camera = self.camera or CameraConfig()
        self.detection = self.detection or DetectionConfig()
        self.events = self.events or EventConfig()
        self.processing = self.processing or ProcessingConfig()
        self.edgex = self.edgex or EdgeXConfig()
        self.action = self.action or ActionBridgeConfig()
    
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "AppConfig":
        """환경 변수에서 설정값 로드"""

        config = cls()
        config.apply_env_overrides(env)
        return config

    def apply_env_overrides(self, env: Optional[Mapping[str, str]] = None) -> None:
        """ENV_OVERRIDES 정의에 따라 설정값을 덮어쓴다"""

        env_data = env or os.environ
        for override in ENV_OVERRIDES:
            raw_value = env_data.get(override.key)
            if raw_value is None:
                continue

            try:
                parsed_value = override.parser(raw_value)
            except Exception as exc:  # 사용자가 잘못된 값을 넣은 경우 무시
                logger.warning("환경 변수 파싱 실패: %s (%s)", override.key, exc)
                continue

            self._set_nested_attr(override.path, parsed_value)

    def _set_nested_attr(self, path: tuple[str, ...], value: object) -> None:
        target = self
        for attr in path[:-1]:
            target = getattr(target, attr)
        setattr(target, path[-1], value)
    
    def validate(self) -> bool:
        """모든 설정값 검증"""
        return self.models.validate()
    
    def summary(self) -> str:
        """현재 설정값 요약 정보 생성"""

        lines = [
            "============ 설정 요약 ============",
            "[모델 경로]",
            f"  헬멧 모델: {self.models.helmet_model or '설정되지 않음'}",
            f"  Person 모델: {self.models.person_model or '설정되지 않음'}",
            f"  Pose 모델: {self.models.pose_model or '설정되지 않음'}",
            "[감지 설정]",
            f"  장치: {self.detection.device}",
            f"  헬멧 신뢰도: {self.detection.helmet_confidence}",
            f"  사람 신뢰도: {self.detection.person_confidence}",
            f"  Pose 신뢰도: {self.detection.pose_confidence}",
            "[성능 설정]",
            f"  목표 FPS: {self.detection.target_fps}",
            f"  프레임 스킵: {self.processing.frame_skip}",
            "[누적 판정]",
            f"  활성화: {self.processing.cumulative_detection_enabled}",
            f"  이력 크기: {self.processing.detection_history_size}",
            f"  위반 임계값: {self.processing.violation_threshold}",
            "[기능]",
            f"  화면 표시: {self.display}",
            f"  구역 감지: {self.zone_detection}",
            f"  데이터셋 수집: {self.collect_dataset}",
            "[MQTT]",
            f"  Broker: {self.mqtt.broker}:{self.mqtt.port}",
            f"  Topic Prefix: {self.mqtt.topic_prefix}",
            f"  디바운싱: {'활성화' if self.events.debounce_enabled else '비활성화'} ({self.events.debounce_seconds}초)",
            "=================================",
        ]
        return "\n".join(lines)


default_config = AppConfig()
