"""Action Layer의 이벤트 실행 정책을 담당한다."""

from __future__ import annotations

from typing import Any, Mapping

from ..canonical_event import get_payload_camera_id, get_payload_event_type
from ..event_priority import event_priority


class ActionPolicy:
    """알람 규칙과 중복 방지 슬롯을 통과한 이벤트만 실행하도록 판단한다."""

    def __init__(self, alarm) -> None:
        self._alarm = alarm

    def should_execute(self, topic: str, payload: Mapping[str, Any]) -> bool:
        """알람 규칙과 이벤트 슬롯 획득 여부를 기준으로 실행 가능 여부를 반환한다."""
        if not self._alarm.should_alarm(topic, payload):
            return False

        camera_id = get_payload_camera_id(payload)
        event_type = get_payload_event_type(payload).lower()
        return self._alarm.try_acquire_slot(
            camera_id,
            event_type,
            priority=event_priority(payload),
            object_id=payload.get("object_id"),
            force=self._alarm.is_demo_event(payload),
        )
