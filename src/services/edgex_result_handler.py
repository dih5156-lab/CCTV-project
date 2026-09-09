"""EdgeX 장치 결과를 Action Layer 상태로 반영한다."""

from __future__ import annotations

import logging
from typing import Callable, Dict

from .command_identifiers import resolve_logical_command_id

logger = logging.getLogger(__name__)


class EdgeXResultHandler:
    """EdgeX 결과 검증, 명령 이력 저장, 상태 이벤트 발행을 담당한다."""

    def __init__(self, *, repository, publish_status: Callable[[str, Dict], None]) -> None:
        """결과 저장소와 상태 발행 콜백을 보관한다."""
        self._repository = repository
        self._publish_status = publish_status

    def handle(self, topic: str, payload: Dict) -> None:
        """EdgeX 결과를 검증하고 devices/result 상태로 발행한다."""
        if not isinstance(payload, dict):
            return

        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            logger.warning("request_id 없는 EdgeX 결과 무시: topic=%s", topic)
            return

        status = str(payload.get("status") or "failed")
        message = str(payload.get("error_code") or "")
        device_id = str(payload.get("device_id") or "").strip()
        command_id = resolve_logical_command_id(request_id, device_id)
        self._repository.record_command(command_id, topic, status, payload, message)
        self._publish_status(
            "devices/result",
            {
                "request_id": request_id,
                "command_id": command_id,
                "event_id": payload.get("event_id"),
                "device_id": payload.get("device_id"),
                "status": status,
                "error_code": payload.get("error_code"),
            },
        )
