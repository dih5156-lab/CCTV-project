"""이벤트 표시/조치 우선순위 정책."""

from __future__ import annotations

from typing import Any, Mapping

from .canonical_event import get_payload_event_type, get_payload_severity

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
