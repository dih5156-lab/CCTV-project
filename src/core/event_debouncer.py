"""이벤트 디바운싱과 큐 포화 시 로컬 백업을 담당하는 헬퍼."""

import json
import logging
import os
import time
from threading import Lock
from typing import Dict, Optional, Tuple

from ..config import AppConfig

logger = logging.getLogger(__name__)

_FALL_REASSOCIATE_IOU = 0.30
_FALL_REASSOCIATE_CENTER_RATIO = 0.25
_FALL_REASSOCIATE_AREA_DELTA = 0.50
BBox = Tuple[int, int, int, int]


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
        self._fall_last_bbox: Dict[Tuple[str, int], BBox] = {}
        # 헬멧 미착용(head) 상태 추적 (camera_id, object_id) 기준
        self._head_last_seen: Dict[Tuple[str, int], float] = {}
        self._head_alerted: Dict[Tuple[str, int], float] = {}

    def should_send(
        self,
        camera_id: str,
        event_type: str,
        object_id: int,
        event=None,
    ) -> bool:
        """중복 전송을 방지하기 위해 이벤트를 보내야 하는지 반환한다."""
        if not self._config.events.debounce_enabled:
            return True

        if event_type == "fall_detected":
            return self._should_send_fall(camera_id, object_id, event=event)

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

    def _should_send_fall(self, camera_id: str, object_id: int, event=None) -> bool:
        """낙상이 sustained_seconds 이상 지속될 때만 True 반환."""
        cfg = self._config.events
        now = time.time()
        with self._lock:
            key = self._resolve_fall_key(camera_id, object_id, event, now)
            last_seen = self._fall_last_seen.get(key, 0)
            if now - last_seen > cfg.fall_gap_reset_seconds:
                self._fall_first_seen[key] = now
                logger.info(
                    "[%s] 낙상 후보 감지 시작 (object_id=%s, sustained=%.1fs, gap_reset=%.1fs)",
                    camera_id,
                    object_id,
                    cfg.fall_sustained_seconds,
                    cfg.fall_gap_reset_seconds,
                )
            self._fall_last_seen[key] = now
            bbox = self._event_bbox(event)
            if bbox is not None:
                self._fall_last_bbox[key] = bbox

            duration = now - self._fall_first_seen.get(key, now)
            if duration < cfg.fall_sustained_seconds:
                logger.debug(
                    "[%s] 낙상 후보 지속 시간 부족: %.1fs/%.1fs (object_id=%s)",
                    camera_id,
                    duration,
                    cfg.fall_sustained_seconds,
                    object_id,
                )
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

    def _resolve_fall_key(
        self,
        camera_id: str,
        object_id: int,
        event,
        now: float,
    ) -> Tuple[str, int]:
        """ID가 바뀐 낙상 후보를 최근 bbox 기준으로 같은 상태에 재연결한다."""
        key = (camera_id, object_id)
        bbox = self._event_bbox(event)
        if bbox is None or key in self._fall_last_seen:
            return key

        cfg = self._config.events
        best_key: Optional[Tuple[str, int]] = None
        best_score = -1.0
        for candidate_key, last_seen in self._fall_last_seen.items():
            candidate_camera, _ = candidate_key
            if candidate_camera != camera_id:
                continue
            if now - last_seen > cfg.fall_gap_reset_seconds:
                continue
            candidate_bbox = self._fall_last_bbox.get(candidate_key)
            if candidate_bbox is None:
                continue
            score = self._fall_bbox_match_score(bbox, candidate_bbox)
            if score > best_score:
                best_score = score
                best_key = candidate_key

        if best_key is None or best_score < 0:
            return key

        self._fall_first_seen[key] = self._fall_first_seen.pop(best_key, now)
        self._fall_last_seen[key] = self._fall_last_seen.pop(best_key, now)
        if best_key in self._fall_alerted:
            self._fall_alerted[key] = self._fall_alerted.pop(best_key)
        if best_key in self._fall_last_bbox:
            self._fall_last_bbox.pop(best_key, None)
        logger.info(
            "[%s] 낙상 후보 track 재연결: old_object_id=%s -> new_object_id=%s",
            camera_id,
            best_key[1],
            object_id,
        )
        return key

    @staticmethod
    def _event_bbox(event) -> Optional[BBox]:
        if event is None:
            return None
        try:
            return (
                int(getattr(event, "x")),
                int(getattr(event, "y")),
                int(getattr(event, "width")),
                int(getattr(event, "height")),
            )
        except (AttributeError, TypeError, ValueError):
            return None

    @classmethod
    def _fall_bbox_match_score(cls, current: BBox, previous: BBox) -> float:
        iou = cls._bbox_iou(current, previous)
        center_ratio = cls._center_distance_ratio(current, previous)
        area_delta = cls._area_delta(current, previous)
        if iou >= _FALL_REASSOCIATE_IOU and area_delta <= _FALL_REASSOCIATE_AREA_DELTA:
            return iou - (center_ratio * 0.1) - (area_delta * 0.05)
        if (
            center_ratio <= _FALL_REASSOCIATE_CENTER_RATIO
            and area_delta <= _FALL_REASSOCIATE_AREA_DELTA
        ):
            return 0.01 - center_ratio - (area_delta * 0.05)
        return -1.0

    @staticmethod
    def _bbox_iou(first: BBox, second: BBox) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[0] + first[2], second[0] + second[2])
        y2 = min(first[1] + first[3], second[1] + second[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        first_area = max(0, first[2]) * max(0, first[3])
        second_area = max(0, second[2]) * max(0, second[3])
        union = first_area + second_area - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center_distance_ratio(first: BBox, second: BBox) -> float:
        first_cx = first[0] + first[2] / 2.0
        first_cy = first[1] + first[3] / 2.0
        second_cx = second[0] + second[2] / 2.0
        second_cy = second[1] + second[3] / 2.0
        distance = ((first_cx - second_cx) ** 2 + (first_cy - second_cy) ** 2) ** 0.5
        scale = max(first[2], first[3], second[2], second[3], 1)
        return distance / scale

    @staticmethod
    def _area_delta(first: BBox, second: BBox) -> float:
        first_area = max(0, first[2]) * max(0, first[3])
        second_area = max(0, second[2]) * max(0, second[3])
        return abs(first_area - second_area) / max(first_area, second_area, 1)

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
            self._fall_last_bbox = {
                k: v for k, v in self._fall_last_bbox.items()
                if self._fall_last_seen.get(k, 0) > cutoff
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
