"""
config.py - 중앙화된 설정 관리
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# 프로젝트 루트 디렉토리 (src/config에서 2단계 상위)
PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass
class ModelPaths:
    """모델 파일 경로 관리"""
    helmet_model: str = None
    pose_model: str = None
    
    def __post_init__(self):
        """기본 모델 경로 자동 탐지"""
        if self.helmet_model is None:
            helmet_candidates = [
                PROJECT_ROOT / "models/helmet_model_ver0.5.pt",
                PROJECT_ROOT / "helmet_model_ver0.5.pt"
            ]
            for path in helmet_candidates:
                if path.exists():
                    self.helmet_model = str(path)
                    break
        
        if self.pose_model is None:
            pose_candidates = [
                PROJECT_ROOT / "models/yolov8n-pose.pt",  # 나노 모델로 변경 (빠름)
                PROJECT_ROOT / "yolov8n-pose.pt",
                "yolov8n-pose.pt",
            ]
            for path in pose_candidates:
                if isinstance(path, Path) and path.exists():
                    self.pose_model = str(path)
                    break
                elif isinstance(path, str):
                    self.pose_model = path
                    break
    
    def validate(self) -> bool:
        """모델 파일 존재 여부 및 크기 검증"""
        valid = True
        
        # 헬멧 모델 검증
        if self.helmet_model:
            if not os.path.exists(self.helmet_model):
                print(f"경고: 헬멧 모델을 찾을 수 없습니다: {self.helmet_model}")
                valid = False
            elif os.path.getsize(self.helmet_model) == 0:
                print(f"오류: 헬멧 모델 파일이 비어있습니다: {self.helmet_model}")
                valid = False
        
        # Pose 모델 검증
        if self.pose_model:
            if not os.path.exists(self.pose_model):
                print(f"경고: Pose 모델을 찾을 수 없습니다: {self.pose_model}")
                valid = False
            elif os.path.getsize(self.pose_model) == 0:
                print(f"오류: Pose 모델 파일이 비어있습니다: {self.pose_model}")
                valid = False
        
        return valid


@dataclass
class ServerConfig:
    """서버 통신 설정"""
    url: str = "http://localhost:8000/api/events"
    timeout: int = 5
    retry_count: int = 3


@dataclass
class CameraConfig:
    """카메라/RTSP 설정"""
    reconnect_interval: int = 5
    max_retries: int = 5
    read_timeout: int = 10
    buffer_size: int = 1


@dataclass
class DetectionConfig:
    """객체 감지 설정"""
    helmet_confidence: float = 0.7  # 헬멧 감지 신뢰도 (0.0~1.0)
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
        if not 0.0 <= self.pose_confidence <= 1.0:
            raise ValueError(f"Pose 신뢰도는 0.0~1.0 사이여야 합니다. 입력값: {self.pose_confidence}")
        
        # IoU 임계값 검증
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"IoU 임계값은 0.0~1.0 사이여야 합니다. 입력값: {self.iou_threshold}")
        
        # FPS 및 장치 검증
        if self.target_fps <= 0:
            raise ValueError(f"목표 FPS는 양수여야 합니다. 입력값: {self.target_fps}")
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"장치는 'cpu' 또는 'cuda'여야 합니다. 입력값: {self.device}")


@dataclass
class EventConfig:
    """이벤트 처리 설정"""
    debounce_enabled: bool = True
    debounce_seconds: float = 3.0
    queue_max_size: int = 500
    event_retention_hours: int = 24
    cleanup_interval: int = 3600


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
    detection_history_size: int = 5  # 감지 이력 크기 (최근 5프레임)
    violation_threshold: int = 4  # 위반 임계값 (5개 중 4개 이상 위반 시 경고)


@dataclass
class AppConfig:
    """애플리케이션 메인 설정"""
    # 주요 설정 객체
    models: ModelPaths = None  # 모델 경로 설정
    server: ServerConfig = None  # 서버 통신 설정
    camera: CameraConfig = None  # 카메라 설정
    detection: DetectionConfig = None  # 감지 설정
    events: EventConfig = None  # 이벤트 처리 설정
    processing: ProcessingConfig = None  # 처리 설정
    
    # 기능 활성화 플래그
    display: bool = False  # 화면 표시 여부
    zone_detection: bool = False  # 구역 감지 여부
    collect_dataset: bool = False  # 데이터셋 수집 여부
    
    # 파일 경로
    dataset_dir: str = str(PROJECT_ROOT / "collected_data")  # 데이터셋 디렉토리
    zones_config: str = str(PROJECT_ROOT / "zones_config.json")  # 구역 설정 파일
    
    def __post_init__(self):
        """기본 설정 객체 초기화"""
        # 각 설정 모듈이 None이면 기본값으로 초기화
        if self.models is None:
            self.models = ModelPaths()
        if self.server is None:
            self.server = ServerConfig()
        if self.camera is None:
            self.camera = CameraConfig()
        if self.detection is None:
            self.detection = DetectionConfig()
        if self.events is None:
            self.events = EventConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """환경 변수에서 설정값 로드"""
        config = cls()
        
        # 모델 경로 환경 변수
        if os.getenv("HELMET_MODEL_PATH"):
            config.models.helmet_model = os.getenv("HELMET_MODEL_PATH")
        if os.getenv("POSE_MODEL_PATH"):
            config.models.pose_model = os.getenv("POSE_MODEL_PATH")
        
        # 서버 및 장치 환경 변수
        if os.getenv("SERVER_URL"):
            config.server.url = os.getenv("SERVER_URL")
        if os.getenv("DEVICE"):
            config.detection.device = os.getenv("DEVICE")
        
        return config
    
    def validate(self) -> bool:
        """모든 설정값 검증"""
        return self.models.validate()
    
    def summary(self) -> str:
        """현재 설정값 요약 정보 생성"""
        lines = [
            "============ 설정 요약 ============",
            "[모델 경로]",
            f"  헬멧 모델: {self.models.helmet_model or '설정되지 않음'}",
            f"  Pose 모델: {self.models.pose_model or '설정되지 않음'}",
            "[감지 설정]",
            f"  장치: {self.detection.device}",
            f"  헬멧 신뢰도: {self.detection.helmet_confidence}",
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
            "[서버]",
            f"  URL: {self.server.url}",
            f"  디바운싱: {'활성화' if self.events.debounce_enabled else '비활성화'} ({self.events.debounce_seconds}초)",
            "=================================",
        ]
        return "\n".join(lines)


default_config = AppConfig()
