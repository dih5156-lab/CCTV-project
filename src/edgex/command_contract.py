"""EdgeX 장치 제어 명령의 공통 계약을 정의한다."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4


def build_command_topic(
    topic_prefix: str,
    jetson_id: str,
    device: str,
    device_id: Optional[str] = None,
) -> str:
    """Jetson과 장치 식별자를 포함한 EdgeX 명령 토픽을 만든다."""
    normalized_prefix = topic_prefix.strip().strip("/")
    normalized_jetson_id = jetson_id.strip().strip("/")
    normalized_device = device.strip().strip("/")
    if not normalized_prefix or not normalized_jetson_id or not normalized_device:
        raise ValueError("명령 토픽의 prefix, Jetson ID, 장치명은 비어 있을 수 없습니다")
    normalized_device_id = (device_id or "").strip().strip("/")
    suffix = f"/{normalized_device_id}" if normalized_device_id else ""
    return f"{normalized_prefix}/{normalized_jetson_id}/{normalized_device}{suffix}"


def build_command_request(
    *,
    event_id: str,
    device: str,
    device_id: Optional[str] = None,
    action: str,
    payload: Mapping[str, Any],
    source: str = "cctv-action-layer",
    request_id: Optional[str] = None,
    issued_at: Optional[str] = None,
) -> Dict[str, Any]:
    """장치 제어 요청을 추적 가능한 공통 딕셔너리로 변환한다."""
    normalized_event_id = event_id.strip()
    normalized_device = device.strip()
    normalized_action = action.strip()
    normalized_source = source.strip()
    if not normalized_event_id or not normalized_device or not normalized_action:
        raise ValueError("event_id, device, action은 비어 있을 수 없습니다")
    if not normalized_source:
        raise ValueError("source는 비어 있을 수 없습니다")
    normalized_device_id = (device_id or "").strip()

    return {
        "version": "1",
        "request_id": (request_id or str(uuid4())).strip(),
        "event_id": normalized_event_id,
        "source": normalized_source,
        "device": normalized_device,
        **({"device_id": normalized_device_id} if normalized_device_id else {}),
        "action": normalized_action,
        "issued_at": issued_at or datetime.now(timezone.utc).isoformat(),
        "payload": dict(payload),
    }
