"""EdgeX 장치 결과를 Action Layer 상태로 반영한다."""

from __future__ import annotations

import logging
from typing import Callable, Dict

from .command_identifiers import resolve_logical_command_id

logger = logging.getLogger(__name__)


class EdgeXResultHandler:
    """EdgeX 결과 검증, 명령 이력 저장, 상태 이벤트 발행을 담당한다."""

    def __init__(self, *, repository, publish_status: Callable[[str, Dict], None], on_result=None) -> None:
        """결과 저장소와 상태 발행 콜백을 보관한다."""
        self._repository = repository
        self._publish_status = publish_status
        self.on_result = on_result

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
        # 실제 장치 결과만 완료 판단에 쓰고 잘못된 상관관계·상태는 보존값을 덮지 않는다.
        if not device_id or not payload.get("event_id") or status not in {
            "acknowledged", "simulated", "failed", "uncertain",
        }:
            logger.warning("장치 식별자/상태가 잘못된 EdgeX 결과 무시: topic=%s", topic)
            return
        command_id = resolve_logical_command_id(request_id, device_id)
        if command_id != f"{payload['event_id']}:{topic.rsplit('/', 1)[-1]}":
            logger.warning("요청 ID와 이벤트/장치 토픽 불일치: %s", request_id)
            return
        # 수신 결과가 저장소의 대상 목록·집계 필드를 주입하지 못하게 계약 필드만 전달한다.
        # 단순 ACK로 큐 점유를 해제하지 않는다. 두 명시적 종료 증거가 함께 있어야 한다.
        completion = ({"execution_completed": True, "resource_idle": True}
                      if payload.get("execution_completed") is True and payload.get("resource_idle") is True else {})
        payload = {"request_id": request_id, "event_id": payload["event_id"],
                   "device_id": device_id, "status": status, "error_code": payload.get("error_code"), **completion}
        stored = self._repository.record_command(command_id, topic, status, payload, message)
        if stored is None:
            return
        if self.on_result is not None:
            self.on_result(payload)
        self._publish_status(
            "devices/result",
            {
                "request_id": request_id,
                "command_id": command_id,
                "event_id": payload.get("event_id"),
                "device_id": payload.get("device_id"),
                "status": status,
                "error_code": payload.get("error_code"),
                "command_status": stored["status"],
                "device_results": stored["payload"].get("device_results", []),
            },
        )
