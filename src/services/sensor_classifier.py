"""Sensor payload classification helpers.

Public API와 센서 브리지에서 공통으로 쓰기 쉬운 순수 함수만 둔다.
새 센서 호환성은 가능하면 이 파일에 추가하고, API endpoint는 호출만 하게 유지한다.
"""

from __future__ import annotations

from typing import Any, Optional


def as_float(value: Any) -> Optional[float]:
    """Return a float value, or None when the input is empty/invalid."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_sensor_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract decoded sensor data from supported payload shapes."""
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    decoded = payload.get("decoded")
    if isinstance(decoded, dict):
        return decoded
    return {}


def classify_sensor_payload(payload: dict[str, Any]) -> dict[str, Optional[str]]:
    """Normalize TLV/AIoT sensor values into a dashboard/action risk state."""
    data = extract_sensor_data(payload)
    event_type = str(payload.get("type") or payload.get("event_type") or "").strip()
    severity = str(payload.get("severity") or "").strip().lower()
    if event_type:
        return {
            "status": "alert" if severity in {"critical", "warning", "warn"} else "normal",
            "severity": "warning" if severity == "warn" else (severity or "normal"),
            "event_type": event_type,
            "reason": str(payload.get("reason") or event_type),
        }

    temperature = as_float(data.get("temperature") or data.get("temp"))
    angle_x = abs(as_float(data.get("angle_x") or data.get("tilt_x") or data.get("x_angle")) or 0.0)
    angle_y = abs(as_float(data.get("angle_y") or data.get("tilt_y") or data.get("y_angle")) or 0.0)
    event_code = as_float(data.get("event_code") or data.get("code"))

    if temperature is not None and temperature >= 50.0:
        severity = "critical" if temperature >= 70.0 else "warning"
        return {
            "status": "alert",
            "severity": severity,
            "event_type": "temperature_alert",
            "reason": f"temperature={temperature:g} >= {'70' if severity == 'critical' else '50'}",
        }
    if max(angle_x, angle_y) >= 30.0:
        return {
            "status": "alert",
            "severity": "warning",
            "event_type": "tilt_alert",
            "reason": f"tilt={max(angle_x, angle_y):g} >= 30",
        }
    if event_code is not None and event_code != 0:
        return {
            "status": "alert",
            "severity": "warning",
            "event_type": "sensor_event",
            "reason": f"event_code={event_code:g}",
        }
    return {"status": "normal", "severity": "normal", "event_type": None, "reason": None}
