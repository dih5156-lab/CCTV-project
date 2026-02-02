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
        """모델 파일 존재 및 크기 검증"""
        valid = True
        if self.helmet_model:
            if not os.path.exists(self.helmet_model):
                print(f"WARN: 헬멧 모델을 찾을 수 없습니다: {self.helmet_model}")
                valid = False
            elif os.path.getsize(self.helmet_model) == 0:
                print(f"ERROR: 헬멧 모델 파일이 비어있습니다: {self.helmet_model}")
                valid = False
        
        if self.pose_model:
            if not os.path.exists(self.pose_model):
                print(f"WARN: Pose 모델을 찾을 수 없습니다: {self.pose_model}")
                valid = False
            elif os.path.getsize(self.pose_model) == 0:
                print(f"ERROR: Pose 모델 파일이 비어있습니다: {self.pose_model}")
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
    helmet_confidence: float = 0.5
    pose_confidence: float = 0.5
    device: str = "cpu"
    target_fps: int = 30
    iou_threshold: float = 0.3
    max_helmet_size: int = 500
    fall_angle_threshold: float = 45.0
    fall_height_ratio: float = 0.3
    
    def __post_init__(self):
        """설정 값 검증"""
        if not 0.0 <= self.helmet_confidence <= 1.0:
            raise ValueError(f"helmet_confidence는 0.0~1.0 사이여야 합니다. 입력값: {self.helmet_confidence}")
        if not 0.0 <= self.pose_confidence <= 1.0:
            raise ValueError(f"pose_confidence는 0.0~1.0 사이여야 합니다. 입력값: {self.pose_confidence}")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold는 0.0~1.0 사이여야 합니다. 입력값: {self.iou_threshold}")
        if self.target_fps <= 0:
            raise ValueError(f"target_fps는 양수여야 합니다. 입력값: {self.target_fps}")
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"device는 'cpu' 또는 'cuda'여야 합니다. 입력값: {self.device}")


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
    thread_join_timeout: int = 5
    camera_reconnect_delay: float = 0.1
    consecutive_failure_threshold: int = 5
    queue_warning_threshold: float = 0.8
    fall_inference_interval: int = 7
    frame_skip: int = 8  # 8프레임마다 1번 추론 (기존 5에서 증가)
    
    # 누적 판정 방식 (오탐 필터링)
    cumulative_detection_enabled: bool = True  # 누적 판정 활성화
    detection_history_size: int = 5  # 최근 5번의 추론 결과 저장
    violation_threshold: int = 4  # 5개 중 4개 이상이 위반이면 경고


@dataclass
class AppConfig:
    """애플리케이션 설정"""
    models: ModelPaths = None
    server: ServerConfig = None
    camera: CameraConfig = None
    detection: DetectionConfig = None
    events: EventConfig = None
    processing: ProcessingConfig = None
    
    display: bool = False
    zone_detection: bool = False
    collect_dataset: bool = False
    
    dataset_dir: str = str(PROJECT_ROOT / "collected_data")
    zones_config: str = str(PROJECT_ROOT / "zones_config.json")
    
    def __post_init__(self):
        """기본값 초기화"""
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
        """환경 변수에서 설정 로드"""
        config = cls()
        
        if os.getenv("HELMET_MODEL_PATH"):
            config.models.helmet_model = os.getenv("HELMET_MODEL_PATH")
        if os.getenv("POSE_MODEL_PATH"):
            config.models.pose_model = os.getenv("POSE_MODEL_PATH")
        if os.getenv("SERVER_URL"):
            config.server.url = os.getenv("SERVER_URL")
        if os.getenv("DEVICE"):
            config.detection.device = os.getenv("DEVICE")
        
        return config
    
    def validate(self) -> bool:
        """설정 검증"""
        return self.models.validate()
    
    def summary(self) -> str:
        """설정 요약 생성"""
        lines = [
            "설정 요약:",
            f"  헬멧 모델: {self.models.helmet_model or '설정되지 않음'}",
            f"  Pose 모델: {self.models.pose_model or '설정되지 않음'}",
            f"  디바이스: {self.detection.device}",
            f"  헬멧 신뢰도: {self.detection.helmet_confidence}",
            f"  Pose 신뢰도: {self.detection.pose_confidence}",
            f"  목표 FPS: {self.detection.target_fps}",
            f"  프레임 스킵: {self.processing.frame_skip}",
            f"  화면 표시: {self.display}",
            f"  구역 감지: {self.zone_detection}",
            f"  데이터셋 수집: {self.collect_dataset}",
            f"  서버 URL: {self.server.url}",
            f"  디바운싱: {'활성화' if self.events.debounce_enabled else '비활성화'} ({self.events.debounce_seconds}초)",
        ]
        return "\n".join(lines)


default_config = AppConfig()
