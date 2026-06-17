"""AI 추론 결과 이벤트 데이터 모델.

EventType:     감지 이벤트 종류 열거형 (HELMET, HEAD, FALL_DETECTED 등).
DetectionEvent: YOLO 추론 결과를 담는 핵심 dataclass.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class EventType(str, Enum):
    """감지 결과 이벤트 타입 열거형"""
    HELMET = "helmet"
    HEAD = "head"
    FACE_RECOGNIZED = "face_recognized"
    FACE_UNKNOWN = "face_unknown"
    DANGER_ZONE = "danger_zone"
    INTRUSION = "intrusion"              # 위험구역 침입 (데모 이벤트)
    FALL_DETECTED = "fall_detected"
    NOT_FALL = "not_fall"
    UNSAFE_BEHAVIOR = "unsafe_behavior"
    PERSON = "person"
    OTHER = "other"
    CROWD_WARNING = "crowd_warning"      # 유동인구 임계값 초과 경고
    ZONE_OBJECT = "zone_object"          # 특정구역 객체 감지
    APPEARANCE_MATCH = "appearance_match" # 외형 조건 매칭 (색상·속성)
    SENSOR_TEMPERATURE = "sensor_temperature"  # 온도 이상

    @classmethod
    def from_string(cls, value: str) -> 'EventType':
        """문자열을 EventType으로 변환"""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.OTHER


# 엄중 이벤트 집합 — 새 타입 추가 시 이곳만 수정
_CRITICAL_EVENT_TYPES: frozenset = frozenset({
    EventType.FALL_DETECTED,
    EventType.DANGER_ZONE,
    EventType.UNSAFE_BEHAVIOR,
})


def severity_for_event_type(event_type: EventType) -> str:
    """이벤트 타입에 따른 심각도 문자열을 반환한다."""
    return "critical" if event_type in _CRITICAL_EVENT_TYPES else "normal"


@dataclass
class DetectionEvent:
    """감지 이벤트 데이터 클래스"""
    event_type: EventType
    x: int
    y: int
    width: int
    height: int
    confidence: float
    timestamp: float
    object_id: Optional[int] = None
    class_idx: Optional[int] = None
    class_name: Optional[str] = None  # YOLO model.names 에서 추출한 클래스 이름 (예: "person", "bicycle")
    keypoints: Optional[list] = None  # YOLOv8-pose 키포인트 정보 (낙상 이벤트에만 저장)
    metadata: Optional[Dict] = None

    def __post_init__(self) -> None:
        """bbox 좌표를 int로 강제 변환한다.

        YOLO 추론 결과는 float tensor로 전달될 수 있으며,
        ROI 좌표 복원 연산(ev.x += x1) 시 float가 누적되면
        OpenCV 그리기 함수 및 bbox JSON 직렬화에서 오류가 발생한다.
        """
        self.x = int(round(self.x))
        self.y = int(round(self.y))
        self.width = int(round(self.width))
        self.height = int(round(self.height))

    def to_dict(self) -> Dict:
        """이벤트를 딕셔너리 형식으로 변환"""
        # critical 여부 — _CRITICAL_EVENT_TYPES 세트에서 결정
        severity = "critical" if self.event_type in _CRITICAL_EVENT_TYPES else "normal"
        metadata = dict(self.metadata or {})
        if self.event_type == EventType.FALL_DETECTED and self.keypoints:
            metadata.setdefault("skeleton_keypoints", self.keypoints)
            metadata.setdefault("skeleton_format", "coco17_xyc")
        return {
            "type": self.event_type.value,
            "severity": severity,
            "bbox": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "object_id": self.object_id,
            "class_idx": self.class_idx,
            "class_name": self.class_name if self.class_name else None,
            "keypoints": self.keypoints if self.keypoints else None,
            "metadata": metadata if metadata else None,
        }
    
    def bbox_dict(self) -> Dict:
        """바운딩 박스 좌표를 딕셔너리로 반환."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def __repr__(self) -> str:
        return (f"DetectionEvent(type={self.event_type.value}, "
                f"bbox=({self.x},{self.y},{self.width},{self.height}), "
                f"conf={self.confidence:.2f}, id={self.object_id})")
