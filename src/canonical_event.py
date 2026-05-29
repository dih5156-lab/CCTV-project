"""표준 이벤트 페이로드 구성 및 조회 헬퍼."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_none(data: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


_APPEARANCE_ATTRIBUTE_KEYS = frozenset(
    {
        "upper_color",
        "lower_color",
        "has_helmet",
        "helmet_color",
        "has_backpack",
        "has_handbag",
        "has_suitcase",
        "gender",
        "age_group",
        "face_name",
        "attribute_backend",
        "attribute_scores",
    }
)


def _extract_payload_attributes(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """외형 속성 필드를 top-level attributes 형태로 모은다."""
    attributes: Dict[str, Any] = {}

    existing = payload.get("attributes")
    if isinstance(existing, Mapping):
        attributes.update(_strip_none(existing))

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in _APPEARANCE_ATTRIBUTE_KEYS:
            if key in metadata and key not in attributes and metadata[key] is not None:
                attributes[key] = metadata[key]

    raw = payload.get("raw")
    if isinstance(raw, Mapping):
        raw_attributes = raw.get("attributes")
        if isinstance(raw_attributes, Mapping):
            for key, value in raw_attributes.items():
                if key not in attributes and value is not None:
                    attributes[key] = value

    return attributes


def _coerce_iso_timestamp(value: Any) -> str:
    if value is None or value == "":
        return _utc_now_iso()
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip()
        if not text:
            return _utc_now_iso()
        try:
            numeric = float(text)
        except ValueError:
            return text

    # LoRa 메타데이터는 초/밀리초/마이크로초가 혼재할 수 있어 크기로 판별한다.
    if abs(numeric) >= 1e14:
        numeric /= 1_000_000.0
    elif abs(numeric) >= 1e11:
        numeric /= 1_000.0
    return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()


def build_canonical_event(
    *,
    camera_id: str,
    event_type: str,
    message_type: str,
    occurred_at: Any,
    source: str,
    severity: Optional[str] = None,
    confidence: Optional[float] = None,
    message_id: Optional[str] = None,
    message: Optional[str] = None,
    display_message: Optional[str] = None,
    tts_message: Optional[str] = None,
    source_type: Optional[str] = None,
    device: Optional[Mapping[str, Any]] = None,
    gateway: Optional[Mapping[str, Any]] = None,
    decoded: Optional[Mapping[str, Any]] = None,
    raw: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    occurred_at_iso = _coerce_iso_timestamp(occurred_at)
    return {
        "schema_version": "1.0",
        "message_type": message_type,
        "message_id": message_id,
        "occurred_at": occurred_at_iso,
        "device": _strip_none(device or {}),
        "gateway": _strip_none(gateway or {}),
        "event": _strip_none(
            {
                "event_type": event_type,
                "severity": severity,
                "source": source,
                "source_type": source_type,
                "confidence": confidence,
                "message": message,
                "display_message": display_message,
                "tts_message": tts_message,
            }
        ),
        "decoded": dict(decoded or {}),
        "raw": dict(raw or {}),
    }


def build_event_id(
    *,
    camera_id: str,
    event_type: str,
    occurred_at: Any,
    object_id: Optional[Any] = None,
    message_id: Optional[str] = None,
    payload: Optional[Mapping[str, Any]] = None,
) -> str:
    """재처리에도 안정적인 event_id를 생성한다."""
    occurred_at_iso = _coerce_iso_timestamp(occurred_at)
    if message_id:
        base = f"{camera_id}|{event_type}|{occurred_at_iso}|{message_id}"
    else:
        payload_part = ""
        if payload:
            try:
                payload_part = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except TypeError:
                payload_part = str(dict(payload))
        base = f"{camera_id}|{event_type}|{occurred_at_iso}|{object_id or ''}|{payload_part}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"evt_{digest}"


def get_payload_camera_id(payload: Mapping[str, Any]) -> str:
    event_device = payload.get("device")
    if isinstance(event_device, Mapping):
        for key in ("camera_id", "device_id", "dev_eui"):
            value = event_device.get(key)
            if value:
                return str(value)
    for key in ("camera_id", "device_id", "dev_eui"):
        value = payload.get(key)
        if value:
            return str(value)
    return "unknown"


def get_payload_event_type(payload: Mapping[str, Any]) -> str:
    event_info = payload.get("event")
    if isinstance(event_info, Mapping):
        for key in ("event_type", "type"):
            value = event_info.get(key)
            if value:
                return str(value)
    value = payload.get("type") or payload.get("event_type")
    if value:
        return str(value)
    return "unknown"


def get_payload_severity(payload: Mapping[str, Any]) -> str:
    event_info = payload.get("event")
    if isinstance(event_info, Mapping):
        value = event_info.get("severity")
        if value:
            return str(value)
    value = payload.get("severity")
    if value:
        return str(value)
    return ""


def get_payload_confidence(payload: Mapping[str, Any]) -> Optional[float]:
    event_info = payload.get("event")
    value: Any = None
    if isinstance(event_info, Mapping):
        value = event_info.get("confidence")
    if value is None:
        value = payload.get("confidence")
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_payload_display_message(payload: Mapping[str, Any]) -> Optional[str]:
    event_info = payload.get("event")
    if isinstance(event_info, Mapping):
        value = event_info.get("display_message")
        if value:
            return str(value)
        value = event_info.get("message")
        if value:
            return str(value)
    value = payload.get("message")
    if value:
        return str(value)
    return None


def get_payload_message_id(payload: Mapping[str, Any]) -> Optional[str]:
    value = payload.get("message_id")
    if value:
        return str(value)
    return None


def get_payload_occurred_at(payload: Mapping[str, Any]) -> Optional[str]:
    value = payload.get("occurred_at") or payload.get("timestamp")
    if value:
        return _coerce_iso_timestamp(value)
    return None


def get_payload_event_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("event_id")
    if value:
        return str(value)
    camera_id = get_payload_camera_id(payload)
    event_type = get_payload_event_type(payload)
    occurred_at = get_payload_occurred_at(payload)
    object_id = payload.get("object_id")
    return build_event_id(
        camera_id=camera_id,
        event_type=event_type,
        occurred_at=occurred_at,
        object_id=object_id,
        message_id=get_payload_message_id(payload),
        payload=payload,
    )


def canonicalize_event_payload(
    payload: Mapping[str, Any],
    *,
    message_type: str = "ai_detection_event",
    source: str = "cctv-ai-engine",
    source_type: str = "vision",
) -> Dict[str, Any]:
    """레거시 이벤트 payload에 표준 이벤트 필드를 보강한다.

    기존 MQTT/Kuiper 호환을 위해 top-level 필드는 유지하고,
    표준 소비자가 사용할 수 있는 schema_version/event/device/raw/event_id를 추가한다.
    """
    normalized = dict(payload)
    camera_id = get_payload_camera_id(normalized)
    event_type = get_payload_event_type(normalized)
    occurred_at = get_payload_occurred_at(normalized) or _utc_now_iso()
    severity = get_payload_severity(normalized) or None
    confidence = get_payload_confidence(normalized)

    normalized.setdefault("camera_id", camera_id)
    normalized.setdefault("type", event_type)
    if confidence is not None:
        normalized.setdefault("confidence", confidence)
    if severity is not None:
        normalized.setdefault("severity", severity)

    attributes = _extract_payload_attributes(normalized)
    if attributes:
        normalized.setdefault("attributes", attributes)

    if "schema_version" not in normalized or "event" not in normalized:
        raw_fields = {
            "bbox": normalized.get("bbox"),
            "object_id": normalized.get("object_id"),
            "class_idx": normalized.get("class_idx"),
            "class_name": normalized.get("class_name"),
            "keypoints": normalized.get("keypoints"),
            "metadata": normalized.get("metadata"),
            "attributes": normalized.get("attributes"),
        }
        normalized.update(
            build_canonical_event(
                camera_id=camera_id,
                event_type=event_type,
                message_type=message_type,
                occurred_at=occurred_at,
                source=source,
                source_type=source_type,
                severity=severity,
                confidence=confidence,
                message_id=get_payload_message_id(normalized),
                message=get_payload_display_message(normalized),
                display_message=get_payload_display_message(normalized),
                tts_message=get_payload_tts_message(normalized),
                device={"camera_id": camera_id},
                decoded={},
                raw=raw_fields,
            )
        )

    normalized.setdefault("event_id", get_payload_event_id(normalized))
    return normalized


def get_payload_tts_message(payload: Mapping[str, Any]) -> Optional[str]:
    event_info = payload.get("event")
    if isinstance(event_info, Mapping):
        value = event_info.get("tts_message")
        if value:
            return str(value)
        value = event_info.get("message")
        if value:
            return str(value)
    value = payload.get("message")
    if value:
        return str(value)
    return None
