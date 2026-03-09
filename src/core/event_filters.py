"""감지 이벤트 추적 및 누적 필터링 공통 헬퍼 모듈."""

import logging
import time
from collections import deque
from threading import Lock
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .events import DetectionEvent
from ..utils.geometry import calculate_iou

logger = logging.getLogger(__name__)


class TrackManager:
    """IOU 기반 중복 제거를 위한 스레드 안전 추적 레지스트리."""

    def __init__(
        self,
        track_timeout: float = 0.5,
        track_iou_threshold: float = 0.5,
        min_track_frames: int = 2,
    ) -> None:
        self.track_timeout = track_timeout
        self.track_iou_threshold = track_iou_threshold
        self.min_track_frames = min_track_frames
        self._tracks: Dict[str, Dict[int, Tuple[float, DetectionEvent, int]]] = {}
        self._lock = Lock()

    def update(self, camera_id: str, events: List[DetectionEvent]) -> Tuple[List[DetectionEvent], Set[int]]:
        """카메라의 활성 트랙을 갱신하고 만료되거나 중복된 ID를 제거한다."""
        filtered: List[DetectionEvent] = []
        removed_ids: Set[int] = set()
        now = time.time()

        with self._lock:
            tracks = self._tracks.setdefault(camera_id, {})
            current_ids = set()

            for event in events:
                if event.object_id is None:
                    filtered.append(event)
                    continue

                track_id = event.object_id
                current_ids.add(track_id)
                _, _, existing_count = tracks.get(track_id, (now, event, 0))
                frame_count = existing_count + 1

                duplicates: List[int] = []
                for existing_id, (_, existing_event, _) in list(tracks.items()):
                    if existing_id == track_id or existing_event.event_type != event.event_type:
                        continue
                    if calculate_iou(event, existing_event) > self.track_iou_threshold:
                        duplicates.append(existing_id)

                for old_id in duplicates:
                    tracks.pop(old_id, None)
                    removed_ids.add(old_id)

                tracks[track_id] = (now, event, frame_count)
                filtered.append(event)

            expired = [
                track_id
                for track_id, (last_seen, _, _) in list(tracks.items())
                if track_id not in current_ids and now - last_seen > self.track_timeout
            ]
            for track_id in expired:
                tracks.pop(track_id, None)
                removed_ids.add(track_id)

        return filtered, removed_ids

    def get_frame_count(self, camera_id: str, object_id: int) -> int:
        with self._lock:
            camera_tracks = self._tracks.get(camera_id, {})
            track_info = camera_tracks.get(object_id)
            return track_info[2] if track_info else 0

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            self._tracks.pop(camera_id, None)


class CumulativeViolationFilter:
    """산발적 위반 이벤트를 억제하는 슬라이딩 윈도우 필터."""

    def __init__(
        self,
        history_max_size: int,
        violation_threshold: int,
        violation_types: Optional[Set[str]] = None,
        enabled: bool = True,
    ) -> None:
        self.history_max_size = history_max_size
        self.violation_threshold = violation_threshold
        self.violation_types: Set[str] = (
            violation_types if violation_types is not None else {"head", "fall_detected"}
        )
        self.enabled = enabled
        self._history: Dict[Tuple[str, int], deque] = {}
        self._last_seen: Dict[Tuple[str, int], float] = {}
        self._lock = Lock()

    def filter(self, camera_id: str, events: List[DetectionEvent]) -> List[DetectionEvent]:
        if not self.enabled or not events:
            return events

        filtered: List[DetectionEvent] = []
        seen_ids = {event.object_id for event in events if event.object_id is not None}
        violation_ids = {
            event.object_id
            for event in events
            if event.object_id is not None and event.event_type.value in self.violation_types
        }

        violation_summary: Dict[Tuple[str, int], Tuple[int, int]] = {}
        now = time.time()

        with self._lock:
            for object_id in seen_ids:
                key = (camera_id, object_id)
                if key not in self._history:
                    self._history[key] = deque(maxlen=self.history_max_size)
                history = self._history[key]
                history.append(object_id in violation_ids)
                self._last_seen[key] = now

            for event in events:
                if event.object_id is None:
                    continue
                if event.event_type.value not in self.violation_types:
                    continue
                key = (camera_id, event.object_id)
                history = self._history.get(key, [])
                violation_summary[key] = (sum(history), len(history))

        for event in events:
            if event.event_type.value not in self.violation_types or event.object_id is None:
                filtered.append(event)
                continue

            key = (camera_id, event.object_id)
            violation_count, history_size = violation_summary.get(key, (0, 0))
            if violation_count >= self.violation_threshold:
                filtered.append(event)
                logger.info(
                    "[%s] 객체 %s: 누적 판정 결과 위반 (%s/%s) -> %s",
                    camera_id,
                    event.object_id,
                    violation_count,
                    history_size,
                    event.event_type.value,
                )
            else:
                logger.debug(
                    "[%s] 객체 %s: 누적 판정 진행 중 (%s/%s) - 아직 경고 아님",
                    camera_id,
                    event.object_id,
                    violation_count,
                    history_size,
                )

        return filtered

    def purge(self, camera_id: str, object_ids: Optional[Iterable[int]] = None) -> int:
        with self._lock:
            if object_ids is None:
                keys = [key for key in list(self._history.keys()) if key[0] == camera_id]
            else:
                obj_set = {obj_id for obj_id in object_ids if obj_id is not None}
                if not obj_set:
                    return 0
                keys = [
                    key
                    for key in list(self._history.keys())
                    if key[0] == camera_id and key[1] in obj_set
                ]

            for key in keys:
                self._history.pop(key, None)
                self._last_seen.pop(key, None)

        return len(keys)

    def cleanup(self, timeout_seconds: float) -> int:
        cutoff = time.time() - timeout_seconds
        removed = 0
        with self._lock:
            for key, last_seen in list(self._last_seen.items()):
                if last_seen < cutoff:
                    self._last_seen.pop(key, None)
                    self._history.pop(key, None)
                    removed += 1
        return removed

    def history_size(self) -> int:
        with self._lock:
            return len(self._history)
