"""
events.py - AI inference result data model
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum

class EventType(Enum):
    """Event type enumeration for detection results"""
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
        """Convert string to EventType"""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.OTHER


@dataclass
class DetectionEvent:
    """Detection event data class"""
    event_type: EventType
    x: int
    y: int
    width: int
    height: int
    confidence: float
    timestamp: float
    object_id: Optional[int] = None
    class_idx: Optional[int] = None
    keypoints: Optional[list] = None  # YOLOv8-pose keypoint info (saved only for falls)

    def to_dict(self) -> Dict:
        """Convert event to dictionary format"""
        return {
            "type": self.event_type.value,
            "bbox": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "object_id": self.object_id,
            "class_idx": self.class_idx,
            "keypoints": self.keypoints if self.keypoints else None
        }
    
    @property
    def center(self) -> tuple:
        """Get bounding box center coordinates"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Get bounding box area"""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Get bounding box aspect ratio"""
        return self.width / max(self.height, 1)
    
    def overlaps_with(self, other: 'DetectionEvent', threshold: float = 0.5) -> bool:
        """Check if this event overlaps with another event"""
        # Calculate intersection
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        iou = intersection / max(union, 1)
        
        return iou >= threshold
    
    def __repr__(self) -> str:
        return (f"DetectionEvent(type={self.event_type.value}, "
                f"bbox=({self.x},{self.y},{self.width},{self.height}), "
                f"conf={self.confidence:.2f}, id={self.object_id})")