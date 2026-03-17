"""
events.py - AI 추론 결과 데이터 모델
"""

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

class EventType(Enum):
    """감지 결과 이벤트 타입 열거형"""
    HELMET = "helmet"
    HEAD = "head"
    DANGER_ZONE = "danger_zone"
    FALL_DETECTED = "fall_detected"
    NOT_FALL = "not_fall"
    UNSAFE_BEHAVIOR = "unsafe_behavior"
    PERSON = "person"
    OTHER = "other"
    
    @classmethod
    def from_string(cls, value: str) -> 'EventType':
        """문자열을 EventType으로 변환"""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.OTHER


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
    keypoints: Optional[list] = None  # YOLOv8-pose 키포인트 정보 (낙상 이벤트에만 저장)

    def to_dict(self) -> Dict:
        """이벤트를 딕셔너리 형식으로 변환"""
        # 낙상은 always critical — ActionBridge 알람 우선처리에 사용됨
        severity = "critical" if self.event_type == EventType.FALL_DETECTED else "normal"
        return {
            "type": self.event_type.value,
            "severity": severity,
            "bbox": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "object_id": self.object_id,
            "class_idx": self.class_idx,
            "keypoints": self.keypoints if self.keypoints else None
        }
    
    def __repr__(self) -> str:
        return (f"DetectionEvent(type={self.event_type.value}, "
                f"bbox=({self.x},{self.y},{self.width},{self.height}), "
                f"conf={self.confidence:.2f}, id={self.object_id})")