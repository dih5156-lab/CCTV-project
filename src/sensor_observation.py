"""원본 TLV 해석값을 단위·결측 정보가 있는 관측 레코드로 변환한다."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any, Mapping

from .time_utils import timestamp_to_kst_iso

_PROFILE_FIELDS = {
    "t34954": {
        "temperature": ("temperature_c", "degC"),
        "humidity": ("humidity_pct", "%"),
    },
    "t34955": {
        "angle_x": ("angle_x_deg", "deg"),
        "angle_y": ("angle_y_deg", "deg"),
    },
    "t34957": {
        "temperature": ("temperature_c", "degC"),
        "angle_x": ("angle_x_deg", "deg"),
        "angle_y": ("angle_y_deg", "deg"),
    },
}


def _measurement(data: Any, field: str, unit: str) -> dict[str, Any]:
    result = {"value": None, "unit": unit, "quality": "invalid", "source_field": field}
    if not isinstance(data, Mapping):
        return result
    value = data.get(field)
    if value is None:
        result["quality"] = "missing"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            numeric = float(value)
        except OverflowError:
            return result
        if math.isfinite(numeric):
            result.update(value=numeric, quality="valid")
    return result


def _observed_time(data: Any) -> tuple[str | None, str]:
    if not isinstance(data, Mapping):
        return None, "invalid"
    value = data.get("created_at")
    if value is None:
        return None, "missing"
    # TLV created_at은 밀리초 정수다. 수신 시각이나 초 단위로 추측하지 않는다.
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return timestamp_to_kst_iso(value / 1000), "valid"
        except (ValueError, OverflowError, OSError):
            pass
    return None, "invalid"


def build_sensor_observation(
    payload: Mapping[str, Any],
    *,
    namespace: str,
    source_record_id: str,
    device_id: str,
    received_at: str,
) -> dict[str, Any] | None:
    """지원하는 TLV 프로필만 변환한다. ID·수신 시각은 입력 경계에서 제공한다."""
    data_key = "data" if "data" in payload else "decoded"
    data = payload.get(data_key, {})
    profile = payload.get("table") or payload.get("tableName")
    if not profile and isinstance(data, Mapping):
        profile = data.get("tableName")
    if not isinstance(profile, str) or profile not in _PROFILE_FIELDS:
        return None

    for name, value in (
        ("namespace", namespace),
        ("source_record_id", source_record_id),
        ("device_id", device_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(received_at, str):
        raise ValueError("received_at must include a timezone")
    receipt = datetime.fromisoformat(
        received_at[:-1] + "+00:00" if received_at.endswith("Z") else received_at
    )
    if receipt.utcoffset() is None:
        raise ValueError("received_at must include a timezone")

    identity = json.dumps(
        [namespace, device_id, profile, source_record_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    observed_at, time_quality = _observed_time(data)
    return {
        "schema_version": "1.0",
        "message_type": "sensor_observation",
        "observation_id": "obs_" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "source_namespace": namespace,
        "source_record_id": source_record_id,
        "device_id": device_id,
        "profile": profile,
        "received_at": received_at,
        "observed_at": observed_at,
        "observed_at_source": (
            f"{data_key}.created_at"
            if isinstance(data, Mapping) and "created_at" in data
            else None
        ),
        "observed_at_quality": time_quality,
        "measurements": {
            name: _measurement(data, field, unit)
            for name, (field, unit) in _PROFILE_FIELDS[profile].items()
        },
    }
