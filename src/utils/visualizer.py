"""
visualizer.py - 감지 결과 시각화
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2

from ..core.events import DetectionEvent, EventType

logger = logging.getLogger(__name__)

# 시각화 상수
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 1
LABEL_OFFSET_Y = 20  # 레이블을 박스 위로 올리는 거리
BBOX_THICKNESS = 2

# 이벤트 타입별 색상 (BGR)
EVENT_COLORS: Dict[EventType, Tuple[int, int, int]] = {
    EventType.HELMET: (255, 0, 0),      # 파란색
    EventType.HEAD: (0, 0, 255),      # 빨간색
    EventType.FALL_DETECTED: (0, 100, 100),     # 갈색
    EventType.DANGER_ZONE: (255, 0, 255),       # 자주색
    EventType.PERSON: (0, 255, 0),              # 초록색
    EventType.OTHER: (200, 200, 200),           # 회색
}

# 이벤트 타입별 그리기 순서 (0=먼저, 큰 박스 먼저 그리기)
_DRAW_PRIORITY: Dict[str, int] = {
    "person":       0,
    "fall_detected": 1,
    "danger_zone":  2,
    "helmet":       3,
    "head":         3,
}

def _parse_event_data(event: Union[Dict, DetectionEvent]) -> Optional[Dict]:
    """이벤트 데이터를 표준화된 딕셔너리 형식으로 파싱"""
    if isinstance(event, dict):
        event_type_str = event.get("type", "unknown")
        if event_type_str == "other":
            return None
        
        return {
            "type_str": event_type_str,
            "color": EVENT_COLORS.get(EventType(event_type_str.upper()), EVENT_COLORS[EventType.OTHER]),
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
            "color": EVENT_COLORS.get(event.event_type, EVENT_COLORS[EventType.OTHER]),
            "confidence": event.confidence,
            "bbox": data.get("bbox", {}),
            "keypoints": event.keypoints,
        }
    
    else:
        logger.warning("알 수 없는 이벤트 타입: %s", type(event))
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
    """프레임에 레이블이 있는 바운딩 박스 그리기"""
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
        logger.warning("바운딩 박스 그리기 실패: %s", e)


def draw_events(frame, events: List[Union[Dict, DetectionEvent]]):
    """프레임에 바운딩 박스와 레이블로 감지 이벤트 그리기"""
    if frame is None or not events:
        return frame

    # _DRAW_PRIORITY 순서로 정렬: 큰 박스(사람) 먼저, 헬멧/head 나중
    ordered: List = []
    for event in events:
        parsed = _parse_event_data(event)
        if parsed is None:
            continue
        priority = _DRAW_PRIORITY.get(parsed["type_str"], 2)
        ordered.append((priority, event, parsed))
    ordered.sort(key=lambda t: t[0])

    for _, event, parsed in ordered:
        bbox = parsed["bbox"]
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)

        if w <= 0 or h <= 0:
            continue

        label = f"{parsed['type_str']} {parsed['confidence']:.2f}"
        _draw_bbox_with_label(frame, x, y, w, h, label, parsed["color"])

        if parsed["type_str"] == "fall_detected" and parsed["keypoints"] is not None:
            _draw_keypoints(frame, parsed["keypoints"])

    return frame


def _draw_keypoints(frame, keypoints):
    """프레임에 YOLOv8-pose 키포인트 스켈레톤 그리기 (COCO 17 키포인트)"""
    if keypoints is None or len(keypoints) != 17:
        return
    
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13],
        [6, 8], [7, 9], [8, 10], [9, 11],
        [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    for _, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
    
    for pt1_idx, pt2_idx in skeleton:
        pt1_idx -= 1
        pt2_idx -= 1
        
        if pt1_idx < 0 or pt1_idx >= 17 or pt2_idx < 0 or pt2_idx >= 17:
            continue
        
        x1, y1, conf1 = keypoints[pt1_idx]
        x2, y2, conf2 = keypoints[pt2_idx]
        
        if conf1 > 0.3 and conf2 > 0.3:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)


