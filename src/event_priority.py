"""이벤트 표시/조치 우선순위 정책."""

from __future__ import annotations

from typing import Any, Mapping

from .canonical_event import (
    get_payload_confidence,
    get_payload_event_type,
    get_payload_metadata,
    get_payload_severity,
)

EVENT_TYPE_PRIORITY: dict[str, int] = {
    "fall_detected": 0,
    "intrusion": 1,
    "danger_zone": 1,
    "head": 2,
    "face_unknown": 3,
    "zone_entered": 4,
    "zone_dwelling": 4,
    "zone_object_detected": 4,
    "crowd_warning": 5,
    "temperature_alert": 6,
    "sensor_temperature": 6,
    "tilt_alert": 6,
    "sensor_event": 7,
    "zone_exited": 12,
    "person": 20,
    "helmet": 30,
    "face_recognized": 30,
}

EVENT_SEVERITY_PRIORITY: dict[str, int] = {
    "emergency": 0,
    "critical": 0,
    "warning": 4,
    "warn": 4,
    "normal": 20,
    "low": 30,
    "info": 30,
}


def event_priority(payload: Mapping[str, Any]) -> int:
    event_type = get_payload_event_type(payload).lower()
    severity = get_payload_severity(payload).lower()
    return min(
        EVENT_TYPE_PRIORITY.get(event_type, 20),
        EVENT_SEVERITY_PRIORITY.get(severity, 20),
    )


def event_risk_level(payload: Mapping[str, Any]) -> str:
    priority = event_priority(payload)
    if priority <= 1:
        return "critical"
    if priority <= 7:
        return "warning"
    if priority >= 30:
        return "low"
    return "normal"


def _clamp_score(value: float) -> int:
    return int(round(max(0.0, min(100.0, value))))


def _bbox_quality_penalty(payload: Mapping[str, Any]) -> float:
    bbox = payload.get("bbox")
    raw = payload.get("raw")
    if not isinstance(bbox, Mapping) and isinstance(raw, Mapping):
        bbox = raw.get("bbox")
    if not isinstance(bbox, Mapping):
        return 0.0

    metadata = get_payload_metadata(payload)
    frame_width = metadata.get("frame_width") or metadata.get("source_width")
    frame_height = metadata.get("frame_height") or metadata.get("source_height")
    try:
        x = float(bbox.get("x", 0))
        y = float(bbox.get("y", 0))
        width = float(bbox.get("width", 0))
        height = float(bbox.get("height", 0))
        frame_w = float(frame_width)
        frame_h = float(frame_height)
    except (TypeError, ValueError):
        return 0.0
    if width <= 0 or height <= 0 or frame_w <= 0 or frame_h <= 0:
        return 0.0

    penalty = 0.0
    area_ratio = (width * height) / (frame_w * frame_h)
    if area_ratio < 0.002:
        penalty += 15.0
    elif area_ratio < 0.006:
        penalty += 8.0

    edge_margin_x = frame_w * 0.02
    edge_margin_y = frame_h * 0.02
    if (
        x <= edge_margin_x
        or y <= edge_margin_y
        or x + width >= frame_w - edge_margin_x
        or y + height >= frame_h - edge_margin_y
    ):
        penalty += 6.0
    return penalty


def event_risk_score(
    payload: Mapping[str, Any],
    *,
    review_status: str | None = None,
) -> int:
    """운영용 이벤트 신뢰도/위험도 점수(0~100)를 계산한다.

    기존 priority는 정렬 순서용 낮은 숫자 우선 정책이고, risk_score는 사람이
    해석하기 쉬운 높은 숫자 우선 점수다.
    """
    event_type = get_payload_event_type(payload).lower()
    severity = get_payload_severity(payload).lower()
    confidence = get_payload_confidence(payload)
    confidence_score = 50.0 * max(0.0, min(1.0, confidence if confidence is not None else 0.5))

    type_weight = {
        "fall_detected": 30.0,
        "intrusion": 28.0,
        "danger_zone": 28.0,
        "zone_entered": 20.0,
        "zone_dwelling": 20.0,
        "zone_object_detected": 20.0,
        "head": 18.0,
        "face_unknown": 16.0,
        "crowd_warning": 16.0,
        "temperature_alert": 16.0,
        "sensor_temperature": 16.0,
        "tilt_alert": 16.0,
        "person": 8.0,
        "helmet": 2.0,
        "face_recognized": 2.0,
    }.get(event_type, 10.0)

    severity_weight = {
        "emergency": 20.0,
        "critical": 20.0,
        "warning": 12.0,
        "warn": 12.0,
        "normal": 5.0,
        "low": 0.0,
        "info": 0.0,
    }.get(severity, 5.0)

    review_adjustment = {
        "true_positive": 12.0,
        "false_positive": -30.0,
        "uncertain": -6.0,
    }.get(str(review_status or "").lower(), 0.0)

    score = confidence_score + type_weight + severity_weight + review_adjustment
    score -= _bbox_quality_penalty(payload)
    return _clamp_score(score)
