"""EdgeX Core Command HTTP 요청을 공통 장치 요청으로 변환한다."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Mapping
from urllib.parse import unquote, urlparse


def handle_command_request(
    service: Any,
    device_id: str | Collection[str],
    path: str,
    payload: Mapping[str, Any],
    command_id: str,
    device_type: str = "speaker",
) -> tuple[int, dict[str, Any]]:
    """EdgeX v3 장치 경로를 서비스 실행 요청으로 변환한다."""
    path_parts = [unquote(part) for part in urlparse(path).path.split("/") if part]
    if len(path_parts) != 6 or path_parts[:4] != ["api", "v3", "device", "name"]:
        return 404, {"error_code": "invalid_command_path"}
    allowed_device_ids = {device_id} if isinstance(device_id, str) else set(device_id)
    if path_parts[4] not in allowed_device_ids:
        return 404, {"error_code": "device_not_found"}

    request = {
        "request_id": command_id or "edgex-command",
        "event_id": str(payload.get("event_id") or command_id or "edgex-event"),
        "device": device_type,
        "action": path_parts[5],
        "payload": dict(payload),
    }
    result = service.execute_request(request)
    body = result.to_dict()
    return (200 if body.get("status") in {"acknowledged", "simulated"} else 502), body
