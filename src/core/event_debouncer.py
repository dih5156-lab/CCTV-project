"""이벤트 디바운싱과 큐 포화 시 로컬 백업을 담당하는 헬퍼."""

import json
import logging
import os
import time
from threading import Lock
from typing import Dict, Optional, Tuple

from ..config import AppConfig

logger = logging.getLogger(__name__)


class EventDebouncer:
    """이벤트 중복 전송 방지, 로컬 백업, 만료 정리를 담당한다."""

    def __init__(self, config: AppConfig, increment_stat) -> None:
        self._config = config
        self._increment_stat = increment_stat
        self._last_events: Dict[Tuple[str, str, int], float] = {}
        self._lock = Lock()
        # 낙상 지속 감지용 상태 추적 (camera_id, object_id) 기준
        self._fall_first_seen: Dict[Tuple[str, int], float] = {}
        self._fall_last_seen: Dict[Tuple[str, int], float] = {}
        self._fall_alerted: Dict[Tuple[str, int], float] = {}
        # 헬멧 미착용(head) 상태 추적 (camera_id, object_id) 기준
        self._head_last_seen: Dict[Tuple[str, int], float] = {}
        self._head_alerted: Dict[Tuple[str, int], float] = {}

    def should_send(self, camera_id: str, event_type: str, object_id: int) -> bool:
        """중복 전송을 방지하기 위해 이벤트를 보내야 하는지 반환한다."""
        if not self._config.events.debounce_enabled:
            return True

        if event_type == "fall_detected":
            return self._should_send_fall(camera_id, object_id)

        if event_type == "head":
            return self._should_send_head(camera_id, object_id)

        key = (camera_id, event_type, object_id)
        now = time.time()
        with self._lock:
            last_time = self._last_events.get(key, 0)
            if now - last_time >= self._config.events.debounce_seconds:
                self._last_events[key] = now
                return True
            self._increment_stat("events_filtered")
            return False

    def _should_send_head(self, camera_id: str, object_id: int) -> bool:
        """헬멧 미착용(head) 이벤트 전송 여부 판단."""
        cfg = self._config.events
        key = (camera_id, object_id)
        now = time.time()
        with self._lock:
            last_seen = self._head_last_seen.get(key, 0)
            last_alert = self._head_alerted.get(key, 0)
            is_state_change = (now - last_seen) > cfg.head_gap_reset_seconds
            self._head_last_seen[key] = now

            if is_state_change or (now - last_alert) >= cfg.head_resend_cooldown:
                self._head_alerted[key] = now
                if is_state_change:
                    logger.info(
                        "[%s] 헬멧 미착용 재등장 감지 -> 즉시 전송 (object_id=%s)",
                        camera_id,
                        object_id,
                    )
                return True

            self._increment_stat("events_filtered")
            return False

    def _should_send_fall(self, camera_id: str, object_id: int) -> bool:
        """낙상이 sustained_seconds 이상 지속될 때만 True 반환."""
        cfg = self._config.events
        key = (camera_id, object_id)
        now = time.time()
        with self._lock:
            last_seen = self._fall_last_seen.get(key, 0)
            if now - last_seen > cfg.fall_gap_reset_seconds:
                self._fall_first_seen[key] = now
            self._fall_last_seen[key] = now

            duration = now - self._fall_first_seen.get(key, now)
            if duration < cfg.fall_sustained_seconds:
                self._increment_stat("events_filtered")
                return False

            last_alert = self._fall_alerted.get(key, 0)
            if now - last_alert < cfg.fall_resend_cooldown:
                self._increment_stat("events_filtered")
                return False

            self._fall_alerted[key] = now
            logger.info(
                "[%s] 낙상 지속 %.1f초 확인 -> 이벤트 전송 (object_id=%s)",
                camera_id,
                duration,
                object_id,
            )
            return True

    def cleanup(self, max_age_hours: Optional[int] = None) -> int:
        """보존 기간이 지난 이벤트 키를 제거하고 제거 수를 반환한다."""
        if max_age_hours is None:
            max_age_hours = self._config.events.event_retention_hours
        cutoff = time.time() - max_age_hours * 3600
        with self._lock:
            before = len(self._last_events)
            self._last_events = {
                k: v for k, v in self._last_events.items() if v > cutoff
            }
            self._fall_first_seen = {
                k: v for k, v in self._fall_first_seen.items() if v > cutoff
            }
            self._fall_last_seen = {
                k: v for k, v in self._fall_last_seen.items() if v > cutoff
            }
            self._fall_alerted = {
                k: v for k, v in self._fall_alerted.items() if v > cutoff
            }
            self._head_last_seen = {
                k: v for k, v in self._head_last_seen.items() if v > cutoff
            }
            self._head_alerted = {
                k: v for k, v in self._head_alerted.items() if v > cutoff
            }
            return before - len(self._last_events)

    def save_locally(self, event_data: Dict) -> None:
        """큐 포화 시 이벤트를 로컬 JSON 파일로 백업한다."""
        try:
            backup_dir = os.path.join(os.getcwd(), "event_backup")
            os.makedirs(backup_dir, exist_ok=True)
            ts_ns = time.time_ns()
            cam_id = event_data.get("camera_id", "unknown")
            filename = f"event_{ts_ns}_{cam_id}.json"
            filepath = os.path.join(backup_dir, filename)
            with open(filepath, "w", encoding="utf-8") as fp:
                json.dump(event_data, fp, ensure_ascii=False, indent=2)
            logger.debug("이벤트 로컬 저장: %s", filepath)
        except Exception as exc:
            logger.error("로컬 저장 실패: %s", exc)


_EventDebouncer = EventDebouncer
