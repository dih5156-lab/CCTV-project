"""MQTT 관리 명령을 처리하고 결과를 기록한다."""

from __future__ import annotations

import uuid
from typing import Callable, Dict


class CommandEventHandler:
    """모드 변경과 수동 이벤트 승인·거부 명령을 처리한다."""

    def __init__(
        self,
        *,
        repository,
        publish_status: Callable[[str, Dict], None],
        set_mode: Callable,
        approve_event: Callable,
        reject_event: Callable,
        control_mode_factory: Callable,
        mode_topic: str,
        approve_topic: str,
        reject_topic: str,
    ) -> None:
        """관리 명령 처리에 필요한 저장소와 ActionBridge 콜백을 보관한다."""
        self._repository = repository
        self._publish_status = publish_status
        self._set_mode = set_mode
        self._approve_event = approve_event
        self._reject_event = reject_event
        self._control_mode_factory = control_mode_factory
        self._mode_topic = mode_topic
        self._approve_topic = approve_topic
        self._reject_topic = reject_topic

    def dispatch(self, topic: str, payload: Dict) -> None:
        """관리 명령을 실행하고 commands/result 결과를 발행한다."""
        command_id = str(payload.get("command_id", uuid.uuid4()))
        self._repository.record_command(command_id, topic, "queued", payload)
        command_status = "ignored"
        message = ""

        if topic == self._mode_topic:
            mode_value = payload.get("mode")
            if mode_value:
                try:
                    self._set_mode(
                        self._control_mode_factory(mode_value),
                        site_id=payload.get("site_id"),
                    )
                    command_status = "success"
                    message = f"mode set to {mode_value}"
                except ValueError:
                    command_status = "error"
                    message = f"invalid mode: {mode_value!r}"

        elif topic == self._approve_topic:
            event_id = payload.get("event_id")
            if event_id:
                ok, message = self._approve_event(event_id)
                command_status = "success" if ok else "error"

        elif topic == self._reject_topic:
            event_id = payload.get("event_id")
            if event_id:
                ok, message = self._reject_event(event_id)
                command_status = "success" if ok else "error"

        result_payload = {
            "command_id": command_id,
            "topic": topic,
            "status": command_status,
            "message": message,
            "site_id": payload.get("site_id"),
            "event_id": payload.get("event_id"),
        }
        self._publish_status("commands/result", result_payload)
        self._repository.record_command(
            command_id, topic, command_status, payload, message
        )
