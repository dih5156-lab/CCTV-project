"""ActionBridge 알람 토픽, 쿨다운, 재생 잠금 관리."""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Dict, Set, Tuple

from ..canonical_event import get_payload_event_type, get_payload_severity

logger = logging.getLogger(__name__)


class _AlarmCoordinator:
    """알람 토픽, 쿨다운, 재생 잠금을 관리한다."""

    _COOLDOWN_EXEMPT: frozenset = frozenset({"head", "fall_detected"})
    _DEVICE_OUTPUT_SUPPRESSED: frozenset = frozenset({"person"})

    def __init__(
        self,
        alarm_topics: Set[str],
        alarm_cooldown_seconds: int,
    ) -> None:
        self.alarm_topics = alarm_topics
        self.alarm_cooldown_seconds = max(1, int(alarm_cooldown_seconds))
        self._last_alarm_ts: Dict[Tuple[str, str], float] = {}
        self._block_until: Dict[str, float] = {}
        self._lock = Lock()

    @staticmethod
    def _mqtt_topic_matches(pattern: str, topic: str) -> bool:
        pat_parts = pattern.split("/")
        top_parts = topic.split("/")
        pattern_index = topic_index = 0
        while pattern_index < len(pat_parts) and topic_index < len(top_parts):
            if pat_parts[pattern_index] == "#":
                return True
            if (
                pat_parts[pattern_index] == "+"
                or pat_parts[pattern_index] == top_parts[topic_index]
            ):
                pattern_index += 1
                topic_index += 1
            else:
                return False
        return pattern_index == len(pat_parts) and topic_index == len(top_parts)

    def should_alarm(self, topic: str, payload: Dict) -> bool:
        event_type = get_payload_event_type(payload).lower()
        severity = get_payload_severity(payload).lower()
        if event_type in self._DEVICE_OUTPUT_SUPPRESSED:
            return False
        return (
            topic == "rest/inbound"
            or event_type in self._COOLDOWN_EXEMPT
            or any(
                self._mqtt_topic_matches(pattern, topic)
                for pattern in self.alarm_topics
            )
            or severity == "critical"
        )

    @staticmethod
    def is_demo_event(payload: Dict) -> bool:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return False
        return metadata.get("demo") is True or metadata.get("source") == "public-demo-ui"

    def try_acquire_slot(
        self, camera_id: str, event_type: str, *, force: bool = False
    ) -> bool:
        if force:
            logger.info(
                "데모 이벤트 - 알람 쿨다운 우회: camera=%s type=%s",
                camera_id,
                event_type,
            )
            return True
        now = time.time()
        with self._lock:
            block_until = self._block_until.get(camera_id, 0.0)
            if now < block_until:
                remaining = int(block_until - now)
                logger.info(
                    "재생 잠금 중 - 스킵 (camera=%s, 남은 %d초)",
                    camera_id,
                    remaining,
                )
                return False

            if event_type not in self._COOLDOWN_EXEMPT:
                key = (camera_id, event_type)
                last_ts = self._last_alarm_ts.get(key, 0.0)
                if now - last_ts < self.alarm_cooldown_seconds:
                    logger.info(
                        "알람 쿨다운 - 스킵 (camera=%s, type=%s)",
                        camera_id,
                        event_type,
                    )
                    return False

            key = (camera_id, event_type)
            self._last_alarm_ts[key] = now
            self._block_until[camera_id] = now + self.alarm_cooldown_seconds
        return True
