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

    # 고정 대체 ID는 서로 다른 HTTP 명령까지 같은 작업으로 묶으므로 누락을 거부한다.
    if not command_id or not command_id.strip():
        return 400, {"error_code": "missing_request_id"}
    parameters = dict(payload)
    event_id = str(parameters.pop("event_id", None) or command_id)
    request = {
        "request_id": command_id,
        "event_id": event_id,
        "device": device_type,
        # 경로로 검증한 대상을 넘겨 다중 장치 서비스의 기본 장치로 잘못 실행하지 않는다.
        "device_id": path_parts[4],
        "action": path_parts[5],
        "payload": parameters,
    }
    result = service.execute_request(request)
    body = result.to_dict()
    status_code = 200 if body.get("status") in {"acknowledged", "simulated"} else 502
    if body.get("error_code") == "request_id_conflict":
        status_code = 409
    return status_code, {
        "apiVersion": "v3",
        "statusCode": status_code,
        **body,
    }


def handle_request_status(service: Any, path: str) -> tuple[int, dict[str, Any]]:
    """내부 장치 서비스의 요청 이력을 읽기 전용으로 조회한다."""
    path = urlparse(path).path
    if not path.startswith("/requests/") or not path.removeprefix("/requests/"):
        return 404, {"error_code": "not_found"}
    request_id = unquote(path.removeprefix("/requests/"))
    result = service.get_request_status(request_id)
    status = 503 if result.get("error_code") == "request_store_unavailable" else (
        404 if result["status"] == "unknown" else 200
    )
    return status, result
