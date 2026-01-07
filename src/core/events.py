"""
events.py - AI 추론 결과 데이터 모델
"""

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

class EventType(Enum):
    HELMET_WEARING = "helmet_wearing"
    HELMET_MISSING = "helmet_missing"
    DANGER_ZONE = "danger_zone"
    FALL_DETECTED = "fall_detected"
    NOT_FALL = "not_fall"
    UNSAFE_BEHAVIOR = "unsafe_behavior"
    PERSON = "person"
    OTHER = "other"


@dataclass
class DetectionEvent:
    event_type: EventType
    x: int
    y: int
    width: int
    height: int
    confidence: float
    timestamp: float
    object_id: Optional[int] = None
    class_idx: Optional[int] = None
    keypoints: Optional[list] = None  # YOLOv8-pose 관절 정보 (낙상 시에만 저장)

    def to_dict(self) -> Dict:
        return {
            "type": self.event_type.value,
            "bbox": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "object_id": self.object_id,
            "class_idx": self.class_idx
        }