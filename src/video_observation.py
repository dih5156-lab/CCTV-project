"""DetectionEvent 원본을 시각 출처·트랙·속성 품질이 있는 관측으로 변환한다."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any, Mapping

from .time_utils import timestamp_to_kst_iso

_BOOLEAN_FIELDS = ("has_helmet", "has_backpack", "has_handbag", "has_suitcase")
_CATEGORY_FIELDS = (
    "upper_color",
    "lower_color",
    "helmet_color",
    "gender",
    "age_group",
    "face_name",
)


def _attribute(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = {"value": None, "unit": None, "quality": "missing", "source_field": None}
    for container_name in ("attributes", "metadata"):
        container = payload.get(container_name)
        if container is None:
            continue
        if not isinstance(container, Mapping):
            result.update(quality="invalid", source_field=container_name)
            return result
        if field not in container:
            continue
        result["source_field"] = f"{container_name}.{field}"
        value = container[field]
        if value is None:
            return result
        if isinstance(value, str) and value.strip().lower() == "unknown":
            result["quality"] = "unknown"
        elif field in _BOOLEAN_FIELDS and isinstance(value, bool):
            result.update(value=value, quality="valid")
        elif field in _CATEGORY_FIELDS and isinstance(value, str) and value.strip():
            result.update(value=value.strip(), quality="valid")
        else:
            result["quality"] = "invalid"
        return result
    return result


def _track_reference(value: Any) -> dict[str, Any]:
    result = {"value": None, "quality": "invalid", "source_field": "object_id"}
    if value is None:
        result["quality"] = "missing"
    elif isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or value == 0xFFFFFFFFFFFFFFFF:
            result["quality"] = "unknown"
        elif value < 0xFFFFFFFFFFFFFFFF:
            result.update(value=value, quality="valid")
    return result


def _processing_time(value: Any) -> tuple[str | None, str]:
    if value is None:
        return None, "missing"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(value):
                return timestamp_to_kst_iso(value), "valid"
        except (ValueError, OverflowError, OSError):
            pass
    return None, "invalid"


def build_video_observation(
    payload: Mapping[str, Any],
    *,
    namespace: str,
    source_record_id: str,
    device_id: str,
    received_at: str,
) -> dict[str, Any]:
    """직렬화된 DetectionEvent를 변환한다. 수신 시각·기존 ID는 호출자가 제공한다."""
    for name, value in (
        ("namespace", namespace),
        ("source_record_id", source_record_id),
        ("device_id", device_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if device_id.strip().lower() == "unknown":
        raise ValueError("device_id must identify a camera")
    if not isinstance(received_at, str):
        raise ValueError("received_at must include a timezone")
    receipt = datetime.fromisoformat(
        received_at[:-1] + "+00:00" if received_at.endswith("Z") else received_at
    )
    if receipt.utcoffset() is None:
        raise ValueError("received_at must include a timezone")

    profile = "detection"
    identity = json.dumps(
        [namespace, device_id, profile, source_record_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    processed_at, time_quality = _processing_time(payload.get("timestamp"))
    return {
        "schema_version": "1.0",
        "message_type": "video_observation",
        "observation_id": "obs_" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "source_namespace": namespace,
        "source_record_id": source_record_id,
        "device_id": device_id,
        "profile": profile,
        "received_at": received_at,
        # 현재 DetectionEvent에는 검증된 카메라 촬영 시각이 없다.
        "observed_at": None,
        "observed_at_source": None,
        "observed_at_quality": "missing",
        "processed_at": processed_at,
        "processed_at_source": "timestamp" if "timestamp" in payload else None,
        "processed_at_quality": time_quality,
        "track_id": _track_reference(payload.get("object_id")),
        "frame_num": metadata.get("frame_num"),
        "model": {
            "backend": metadata.get("backend", payload.get("backend")),
            "task": metadata.get("model_task"),
            "gie_id": metadata.get("gie_id"),
            "version": metadata.get("model_version"),
            "attribute_backend": metadata.get("attribute_backend"),
        },
        "measurements": {
            field: _attribute(payload, field)
            for field in (*_BOOLEAN_FIELDS, *_CATEGORY_FIELDS)
        },
    }
