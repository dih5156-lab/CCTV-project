"""Action Layer와 EdgeX 사이의 명령 식별자 규칙을 정의한다."""

from __future__ import annotations


def build_device_command_id(event_id: str, device: str) -> str:
    """이벤트와 논리 장치를 조합해 Action Layer 명령 ID를 만든다."""
    normalized_event_id = str(event_id or "").strip()
    normalized_device = str(device or "").strip()
    if not normalized_event_id or not normalized_device:
        raise ValueError("event_id와 device는 비어 있을 수 없습니다")
    return f"{normalized_event_id}:{normalized_device}"


def build_transport_request_id(command_id: str, device_id: str) -> str:
    """논리 명령과 물리 장치 ID를 조합해 EdgeX 요청 ID를 만든다."""
    normalized_command_id = str(command_id or "").strip()
    normalized_device_id = str(device_id or "").strip()
    if not normalized_command_id:
        raise ValueError("command_id는 비어 있을 수 없습니다")
    return (
        f"{normalized_command_id}:{normalized_device_id}"
        if normalized_device_id
        else normalized_command_id
    )


def resolve_logical_command_id(request_id: str, device_id: str) -> str:
    """물리 장치 요청 ID에서 원래 논리 명령 ID를 복원한다."""
    normalized_request_id = str(request_id or "").strip()
    normalized_device_id = str(device_id or "").strip()
    if not normalized_request_id:
        raise ValueError("request_id는 비어 있을 수 없습니다")
    if normalized_device_id and normalized_request_id.endswith(
        f":{normalized_device_id}"
    ):
        return normalized_request_id[: -(len(normalized_device_id) + 1)]
    return normalized_request_id
