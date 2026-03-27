"""
visualizer.py - 감지 결과 시각화
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.events import DetectionEvent, EventType

logger = logging.getLogger(__name__)

# 시각화 상수
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 1
LABEL_OFFSET_Y = 20  # 레이블을 박스 위로 올리는 거리
BBOX_THICKNESS = 2
_PIL_FONT_SIZE = 18
_PIL_FONT: Optional[ImageFont.FreeTypeFont] = None
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# 이벤트 타입별 색상 (BGR)
EVENT_COLORS: Dict[EventType, Tuple[int, int, int]] = {
    EventType.HELMET: (255, 0, 0),      # 파란색
    EventType.HEAD: (0, 0, 255),      # 빨간색
    EventType.FACE_RECOGNIZED: (0, 200, 255),  # 주황색
    EventType.FACE_UNKNOWN: (80, 80, 255),     # 붉은 주황
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
    "face_recognized": 4,
    "face_unknown": 4,
}

def _parse_event_data(event: Union[Dict, DetectionEvent]) -> Optional[Dict]:
    """이벤트 데이터를 표준화된 딕셔너리 형식으로 파싱"""
    if isinstance(event, dict):
        event_type_str = event.get("type", "unknown")
        if event_type_str == "other":
            return None
        
        return {
            "type_str": event_type_str,
            "color": EVENT_COLORS.get(EventType(event_type_str), EVENT_COLORS[EventType.OTHER]),
            "confidence": event.get('confidence', 0),
            "bbox": event.get("bbox", {}),
            "keypoints": event.get("keypoints", None),
            "object_id": event.get("object_id"),
            "metadata": event.get("metadata") or {},
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
            "object_id": event.object_id,
            "metadata": data.get("metadata") or {},
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
        
        font = _get_pil_font()
        text_width, text_height = _measure_text(label, font)
        
        # 레이블 배경 그리기
        label_y = int(y - LABEL_OFFSET_Y)
        cv2.rectangle(
            frame,
            (int(x), label_y),
            (int(x + text_width), label_y + text_height + 5),
            color,
            -1  # 채우기
        )
        
        # 레이블 텍스트 그리기 (한글 지원)
        _draw_unicode_text(
            frame,
            label,
            (int(x), label_y + 2),
            color=(255, 255, 255),
            font=font,
        )
        
    except Exception as e:
        logger.warning("바운딩 박스 그리기 실패: %s", e)


def _get_pil_font() -> Optional[ImageFont.FreeTypeFont]:
    global _PIL_FONT
    if _PIL_FONT is not None:
        return _PIL_FONT

    for candidate in _FONT_CANDIDATES:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            _PIL_FONT = ImageFont.truetype(str(path), _PIL_FONT_SIZE)
            return _PIL_FONT
        except Exception:
            continue
    return None


def _measure_text(label: str, font: Optional[ImageFont.FreeTypeFont]) -> Tuple[int, int]:
    if font is None:
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            LABEL_FONT,
            LABEL_FONT_SCALE,
            LABEL_FONT_THICKNESS,
        )
        return text_width, text_height

    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), label, font=font)
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


def _draw_unicode_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font: Optional[ImageFont.FreeTypeFont],
) -> None:
    if font is None:
        cv2.putText(
            frame,
            text,
            (int(position[0]), int(position[1] + 14)),
            LABEL_FONT,
            LABEL_FONT_SCALE,
            color,
            LABEL_FONT_THICKNESS,
        )
        return

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((int(position[0]), int(position[1])), text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_events(frame, events: List[Union[Dict, DetectionEvent]]):
    """프레임에 바운딩 박스와 레이블로 감지 이벤트 그리기"""
    if frame is None or not events:
        return frame

    recognized_labels: Dict[int, Tuple[str, float, bool]] = {}
    for event in events:
        parsed = _parse_event_data(event)
        if parsed is None:
            continue
        if parsed["type_str"] not in {"face_recognized", "face_unknown"}:
            continue
        object_id = parsed.get("object_id")
        metadata = parsed.get("metadata") or {}
        if object_id is None:
            continue
        face_name = metadata.get("face_name") or ("unknown" if parsed["type_str"] == "face_unknown" else None)
        if not face_name:
            continue
        recognized_labels[int(object_id)] = (
            str(face_name),
            float(metadata.get("face_score", parsed["confidence"])),
            parsed["type_str"] == "face_recognized",
        )

    # _DRAW_PRIORITY 순서로 정렬: 큰 박스(사람) 먼저, 헬멧/head 나중
    ordered: List = []
    for event in events:
        parsed = _parse_event_data(event)
        if parsed is None:
            continue
        if parsed["type_str"] in {"face_recognized", "face_unknown"}:
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

        label_type = parsed["type_str"]
        label_confidence = parsed["confidence"]
        if label_type == "person":
            face_info = recognized_labels.get(parsed.get("object_id"))
            if face_info is not None:
                face_name, face_score, matched = face_info
                if matched:
                    label_type = face_name
                else:
                    label_type = "unknown" if str(face_name).lower() == "unknown" else f"unknown ({face_name})"
                label_confidence = face_score

        label = f"{label_type} {label_confidence:.2f}"
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


