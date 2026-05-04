"""DetectionEvent에 안정적인 임시 object_id를 부여하는 유틸리티."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from .events import DetectionEvent


def event_iou(first: DetectionEvent, second: DetectionEvent) -> float:
    """두 DetectionEvent bbox의 IoU를 계산한다."""
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x + first.width, second.x + second.width)
    y2 = min(first.y + first.height, second.y + second.height)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    first_area = max(0, first.width) * max(0, first.height)
    second_area = max(0, second.width) * max(0, second.height)
    union = first_area + second_area - inter
    return inter / union if union > 0 else 0.0


class SyntheticObjectIdAssigner:
    """트래커 메타가 없는 이벤트에 IoU 기반 임시 ID를 부여한다."""

    def __init__(self, *, track_iou: float, track_timeout: float) -> None:
        self._track_iou = track_iou
        self._track_timeout = track_timeout
        self._tracks: Dict[str, Dict[int, Tuple[float, DetectionEvent]]] = {}
        self._next_object_id = 1

    def remove_camera(self, camera_id: str) -> None:
        self._tracks.pop(camera_id, None)

    def assign(self, camera_name: str, events: List[DetectionEvent]) -> List[DetectionEvent]:
        """Raw tensor 결과에 기존 후처리용 stable object_id를 붙인다."""
        now = time.time()
        tracks = self._tracks.setdefault(camera_name, {})
        self._purge_stale(tracks, now)

        for event in events:
            if event.object_id is not None:
                continue

            best_track_id, best_iou = self._find_best_track(event, tracks)
            if best_track_id is None or best_iou < self._track_iou:
                best_track_id = self._allocate()

            event.object_id = best_track_id
            tracks[best_track_id] = (now, event)

        return events

    def _purge_stale(
        self,
        tracks: Dict[int, Tuple[float, DetectionEvent]],
        now: float,
    ) -> None:
        for track_id, (last_seen, _) in list(tracks.items()):
            if now - last_seen > self._track_timeout:
                tracks.pop(track_id, None)

    @staticmethod
    def _find_best_track(
        event: DetectionEvent,
        tracks: Dict[int, Tuple[float, DetectionEvent]],
    ) -> Tuple[Optional[int], float]:
        best_track_id: Optional[int] = None
        best_iou = 0.0
        for track_id, (_, tracked_event) in tracks.items():
            if tracked_event.event_type != event.event_type:
                continue
            iou = event_iou(event, tracked_event)
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        return best_track_id, best_iou

    def _allocate(self) -> int:
        object_id = self._next_object_id
        self._next_object_id += 1
        return object_id
