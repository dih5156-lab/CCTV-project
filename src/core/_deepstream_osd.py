"""DeepStream OSD overlay 유틸리티."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

_COCO_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def label_color(label: str) -> Tuple[float, float, float, float]:
    """OSD label별 RGBA 색상을 반환한다."""
    normalized = (label or "").strip().lower().replace("-", "_")
    if normalized in {"fall", "fall_detected"}:
        return (1.0, 0.0, 1.0, 1.0)
    if normalized in {"head", "hardhat_off", "no_helmet", "helmet_off", "helmet_missing"}:
        return (1.0, 0.05, 0.05, 1.0)
    if normalized in {"helmet", "hardhat", "head_protected"}:
        return (0.05, 0.9, 0.2, 1.0)
    if normalized == "person":
        return (0.05, 0.55, 1.0, 1.0)
    return (1.0, 0.75, 0.05, 1.0)


def fall_skeleton_points(
    detection: Dict[str, Any],
    *,
    min_keypoint_confidence: float,
) -> Dict[int, Tuple[int, int]]:
    """낙상 detection의 keypoint 중 OSD에 그릴 점만 추출한다."""
    if not detection.get("is_fall"):
        return {}

    keypoints = detection.get("keypoints")
    if not keypoints:
        return {}

    threshold = float(
        os.environ.get("DS_OSD_KEYPOINT_CONFIDENCE", str(min_keypoint_confidence))
    )
    points: Dict[int, Tuple[int, int]] = {}
    for idx, keypoint in enumerate(keypoints[:17]):
        if len(keypoint) < 3:
            continue
        try:
            x_coord = int(float(keypoint[0]))
            y_coord = int(float(keypoint[1]))
            confidence = float(keypoint[2])
        except (TypeError, ValueError):
            continue
        if confidence >= threshold:
            points[idx] = (x_coord, y_coord)
    return points


def add_fall_skeleton_overlay(
    *,
    pyds_module: Any,
    batch_meta: Any,
    frame_meta: Any,
    detection: Dict[str, Any],
    min_keypoint_confidence: float,
) -> None:
    """낙상 keypoint skeleton을 OSD display meta에 추가한다."""
    points = fall_skeleton_points(
        detection,
        min_keypoint_confidence=min_keypoint_confidence,
    )
    if not points:
        return

    line_segments = [
        (points[start], points[end])
        for start, end in _COCO_SKELETON_EDGES
        if start in points and end in points
    ]
    circles = list(points.values())
    max_elements = int(os.environ.get("DS_OSD_MAX_ELEMENTS_PER_META", "16"))

    for start in range(0, len(line_segments), max_elements):
        display_meta = pyds_module.nvds_acquire_display_meta_from_pool(batch_meta)
        chunk = line_segments[start : start + max_elements]
        display_meta.num_lines = len(chunk)
        for idx, ((x1, y1), (x2, y2)) in enumerate(chunk):
            line_params = display_meta.line_params[idx]
            line_params.x1 = int(x1)
            line_params.y1 = int(y1)
            line_params.x2 = int(x2)
            line_params.y2 = int(y2)
            line_params.line_width = 4
            line_params.line_color.set(1.0, 0.0, 1.0, 1.0)
        pyds_module.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    for start in range(0, len(circles), max_elements):
        display_meta = pyds_module.nvds_acquire_display_meta_from_pool(batch_meta)
        chunk = circles[start : start + max_elements]
        display_meta.num_circles = len(chunk)
        for idx, (x_coord, y_coord) in enumerate(chunk):
            circle_params = display_meta.circle_params[idx]
            circle_params.xc = int(x_coord)
            circle_params.yc = int(y_coord)
            circle_params.radius = 5
            circle_params.circle_color.set(1.0, 0.0, 1.0, 1.0)
            circle_params.has_bg_color = 1
            circle_params.bg_color.set(0.0, 0.0, 0.0, 0.85)
        pyds_module.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def add_osd_overlays(
    *,
    pyds_module: Any,
    batch_meta: Any,
    frame_meta: Any,
    detections: List[Dict[str, Any]],
    min_keypoint_confidence: float,
) -> None:
    """bbox/label/fall skeleton OSD display meta를 추가한다."""
    if not detections:
        return

    max_elements = int(os.environ.get("DS_OSD_MAX_ELEMENTS_PER_META", "16"))
    for start in range(0, len(detections), max_elements):
        chunk = detections[start : start + max_elements]
        display_meta = pyds_module.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_rects = len(chunk)
        display_meta.num_labels = len(chunk)

        for idx, detection in enumerate(chunk):
            x, y, width, height = detection["box"]
            label = "fall_detected" if detection.get("is_fall") else str(detection["label"])
            confidence = float(detection["confidence"])
            red, green, blue, alpha = label_color(label)

            rect_params = display_meta.rect_params[idx]
            rect_params.left = float(x)
            rect_params.top = float(y)
            rect_params.width = float(width)
            rect_params.height = float(height)
            rect_params.border_width = 4
            rect_params.has_bg_color = 0
            rect_params.border_color.set(red, green, blue, alpha)

            text_params = display_meta.text_params[idx]
            text_params.display_text = f"{label} {confidence:.2f}"
            text_params.x_offset = int(x)
            text_params.y_offset = max(0, int(y) - 12)
            text_params.font_params.font_name = "Serif"
            text_params.font_params.font_size = 14
            text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            text_params.set_bg_clr = 1
            text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.75)

        pyds_module.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    for detection in detections:
        add_fall_skeleton_overlay(
            pyds_module=pyds_module,
            batch_meta=batch_meta,
            frame_meta=frame_meta,
            detection=detection,
            min_keypoint_confidence=min_keypoint_confidence,
        )
