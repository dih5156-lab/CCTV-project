"""
visualizer.py - Detection result visualization
"""

from typing import List, Union, Dict, Tuple, Optional, Any
from ..core.events import EventType, DetectionEvent
import cv2
import logging

logger = logging.getLogger(__name__)

# 시각화 상수
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 1
LABEL_OFFSET_Y = 20  # 레이블을 박스 위로 올리는 거리
BBOX_THICKNESS = 2

# 이벤트 타입별 색상 (BGR)
EVENT_COLORS: Dict[EventType, Tuple[int, int, int]] = {
    EventType.HELMET_WEARING: (255, 0, 0),      # 파란색
    EventType.HELMET_MISSING: (0, 0, 255),      # 빨간색
    EventType.FALL_DETECTED: (0, 100, 100),     # 갈색
    EventType.DANGER_ZONE: (0, 255, 255),       # 노란색
    EventType.PERSON: (0, 255, 0),              # 초록색
    EventType.OTHER: (200, 200, 200),           # 회색
}

DEFAULT_COLOR = (255, 255, 255)  # 흰색


def _parse_event_data(event: Union[Dict, DetectionEvent]) -> Optional[Dict]:
    """Parse event data to standardized dictionary format."""
    if isinstance(event, dict):
        event_type_str = event.get("type", "unknown")
        if event_type_str == "other":
            return None
        
        return {
            "type_str": event_type_str,
            "color": DEFAULT_COLOR,
            "confidence": event.get('confidence', 0),
            "bbox": event.get("bbox", {}),
            "keypoints": event.get("keypoints", None),
        }
    
    elif isinstance(event, DetectionEvent):
        # OTHER 타입 필터링
        if event.event_type == EventType.OTHER:
            return None
        
        data = event.to_dict()
        return {
            "type_str": event.event_type.value,
            "color": EVENT_COLORS.get(event.event_type, DEFAULT_COLOR),
            "confidence": event.confidence,
            "bbox": data.get("bbox", {}),
            "keypoints": event.keypoints,
        }
    
    else:
        logger.warning(f"알 수 없는 이벤트 타입: {type(event)}")
        return None


def _draw_bbox_with_label(
    frame,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    color: Tuple[int, int, int]
) -> None:
    """Draw bounding box with label on frame."""
    try:
        # 바운딩 박스 그리기
        cv2.rectangle(
            frame,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            color,
            BBOX_THICKNESS
        )
        
        # 레이블 배경 크기 계산
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            LABEL_FONT,
            LABEL_FONT_SCALE,
            LABEL_FONT_THICKNESS
        )
        
        # 레이블 배경 그리기
        label_y = int(y - LABEL_OFFSET_Y)
        cv2.rectangle(
            frame,
            (int(x), label_y),
            (int(x + text_width), label_y + text_height + 5),
            color,
            -1  # 채우기
        )
        
        # 레이블 텍스트 그리기
        cv2.putText(
            frame,
            label,
            (int(x), label_y + text_height),
            LABEL_FONT,
            LABEL_FONT_SCALE,
            (255, 255, 255),  # 흰색 텍스트
            LABEL_FONT_THICKNESS
        )
        
    except Exception as e:
        logger.warning(f"바운딩 박스 그리기 실패: {e}")


def draw_events(frame, events: List[Union[Dict, DetectionEvent]]):
    """Draw detection events with bounding boxes and labels on frame."""
    if frame is None:
        return frame
    
    if not events:
        return frame

    # 그리기 순서: 1) person (큰 박스) 먼저, 2) 헬멧 (작은 박스) 나중에
    person_events = []
    helmet_events = []
    fall_events = []
    other_events = []
    
    for event in events:
        parsed = _parse_event_data(event)
        if parsed is None:
            continue
        
        event_type = parsed["type_str"]
        if event_type == "person":
            person_events.append((event, parsed))
        elif event_type in ["helmet_wearing", "helmet_missing"]:
            helmet_events.append((event, parsed))
        elif event_type == "fall_detected":
            fall_events.append((event, parsed))
        else:
            other_events.append((event, parsed))
    
    # 그리기 순서: person → fall → helmet → others
    for event, parsed in person_events + fall_events + other_events:
        bbox = parsed["bbox"]
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)
        
        if w <= 0 or h <= 0:
            continue
        
        label = f"{parsed['type_str']} {parsed['confidence']:.2f}"
        _draw_bbox_with_label(frame, x, y, w, h, label, parsed["color"])
        
        # 낙상 감지 시 관절 표시
        if parsed["type_str"] == "fall_detected" and parsed["keypoints"] is not None:
            _draw_keypoints(frame, parsed["keypoints"])
    
    # 헬멧은 마지막에 그려서 person 박스 위에 표시
    for event, parsed in helmet_events:
        bbox = parsed["bbox"]
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)
        
        if w <= 0 or h <= 0:
            continue
        
        label = f"{parsed['type_str']} {parsed['confidence']:.2f}"
        _draw_bbox_with_label(frame, x, y, w, h, label, parsed["color"])

    return frame


def _draw_keypoints(frame, keypoints):
    """Draw YOLOv8-pose keypoints skeleton on frame (COCO 17 keypoints)."""
    if keypoints is None or len(keypoints) != 17:
        return
    
    # COCO keypoints skeleton (연결선)
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 다리
        [6, 12], [7, 13],  # 몸통
        [6, 8], [7, 9], [8, 10], [9, 11],  # 팔
        [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]  # 얼굴 + 어깨
    ]
    
    # 관절 그리기
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:  # 신뢰도 임계값
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)  # 노란색 점
    
    # 연결선 그리기
    for pt1_idx, pt2_idx in skeleton:
        pt1_idx -= 1  # COCO는 1-based, 배열은 0-based
        pt2_idx -= 1
        
        if pt1_idx < 0 or pt1_idx >= 17 or pt2_idx < 0 or pt2_idx >= 17:
            continue
        
        x1, y1, conf1 = keypoints[pt1_idx]
        x2, y2, conf2 = keypoints[pt2_idx]
        
        if conf1 > 0.3 and conf2 > 0.3:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # 노란색 선


