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
    value = payload.get("type")
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
