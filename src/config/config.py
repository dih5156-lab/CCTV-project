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
    pose_model: str = "yolov8n-pose.pt"  # 사람 감지 + 관절 감지 (낙상 판단용)
    
    def __post_init__(self):
        """기본 모델 경로 자동 탐지"""
        # 헬멧 모델 자동 탐지
        if self.helmet_model is None:
            helmet_candidates = [
                PROJECT_ROOT / "runs/detect/train/weights/test.pt",
                PROJECT_ROOT / "runs/detect/helmet_model/weights/best.pt",
                PROJECT_ROOT / "helmet_model.pt",
                PROJECT_ROOT / "models/helmet_model.pt",
            ]
            for path in helmet_candidates:
                if path.exists():
                    self.helmet_model = str(path)
                    break
    
    def validate(self) -> bool:
        """모델 파일 존재 여부 확인"""
        valid = True
        if self.helmet_model and not os.path.exists(self.helmet_model):
            print(f"⚠️ 헬멧 모델 없음: {self.helmet_model}")
            valid = False
        # pose 모델은 ultralytics가 자동 다운로드하므로 검증 제외
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
    reconnect_interval: int = 5  # 재연결 대기 시간 (초)
    max_retries: int = 5  # 최대 재시도 횟수
    read_timeout: int = 10  # 읽기 타임아웃 (초)
    buffer_size: int = 1  # 프레임 버퍼 크기


@dataclass
class DetectionConfig:
    """객체 탐지 설정"""
    helmet_confidence: float = 0.45
    pose_confidence: float = 0.5  # pose 모델 신뢰도 (사람 + 관절)
    device: str = "cpu"  # "cuda" or "cpu"
    target_fps: int = 30
    
    # NMS 설정
    iou_threshold: float = 0.3  # 중복 박스 제거 임계값
    
    # 헬멧 박스 크기 제한 (픽셀)
    max_helmet_size: int = 500  # 이보다 크면 오탐지로 간주
    
    # 낙상 감지 설정 (관절 기반)
    fall_angle_threshold: float = 45.0  # 몸 각도가 이보다 작으면 낙상 (도 단위)
    fall_height_ratio: float = 0.3  # 머리 높이가 박스 높이의 30% 이하면 낙상


@dataclass
class EventConfig:
    """이벤트 처리 설정"""
    debounce_enabled: bool = True
    debounce_seconds: float = 3.0  # 동일 이벤트 재전송 간격
    queue_max_size: int = 500
    event_retention_hours: int = 24  # 이벤트 기록 보관 시간


@dataclass
class AppConfig:
    """전체 애플리케이션 설정"""
    models: ModelPaths = None
    server: ServerConfig = None
    camera: CameraConfig = None
    detection: DetectionConfig = None
    events: EventConfig = None
    
    # 기능 토글
    display: bool = False
    zone_detection: bool = False
    collect_dataset: bool = False
    
    # 디렉토리
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
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """환경변수에서 설정 로드"""
        config = cls()
        
        # 모델 경로 오버라이드
        if os.getenv("HELMET_MODEL_PATH"):
            config.models.helmet_model = os.getenv("HELMET_MODEL_PATH")
        if os.getenv("POSE_MODEL_PATH"):
            config.models.pose_model = os.getenv("POSE_MODEL_PATH")
        
        # 서버 URL
        if os.getenv("SERVER_URL"):
            config.server.url = os.getenv("SERVER_URL")
        
        # 디바이스 설정
        if os.getenv("DEVICE"):
            config.detection.device = os.getenv("DEVICE")
        
        return config
    
    def validate(self) -> bool:
        """설정 유효성 검증"""
        return self.models.validate()


# 전역 설정 인스턴스
default_config = AppConfig()
