"""DetectionEvent에 안정적인 임시 object_id를 부여하는 유틸리티."""

from __future__ import annotations

import time
from collections.abc import Mapping
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
        self._clocks: Dict[str, Tuple[Optional[int], Optional[float]]] = {}
        self._next_object_id = 1

    def remove_camera(self, camera_id: str) -> None:
        self._tracks.pop(camera_id, None)
        self._clocks.pop(camera_id, None)

    def assign(self, camera_name: str, events: List[DetectionEvent]) -> List[DetectionEvent]:
        """Raw tensor 결과에 기존 후처리용 stable object_id를 붙인다."""
        if not events:
            return []
        now = self._observation_time(camera_name, events)
        if now is None:
            return []
        tracks = self._tracks.setdefault(camera_name, {})
        self._purge_stale(tracks, now)

        # 같은 프레임의 두 사람을 합치지 않도록 이전 트랙은 한 관측에만 배정한다.
        reserved_ids = {event.object_id for event in events if event.object_id is not None}
        available_tracks = {key: value for key, value in tracks.items() if key not in reserved_ids}

        for event in events:
            if event.object_id is not None:
                continue

            best_track_id, best_iou = self._find_best_track(event, available_tracks)
            if best_track_id is None or best_iou < self._track_iou:
                best_track_id = self._allocate()
                while best_track_id in reserved_ids:
                    best_track_id = self._allocate()

            event.object_id = best_track_id
            available_tracks.pop(best_track_id, None)
            tracks[best_track_id] = (now, event)

        return events

    def _observation_time(
        self, camera_name: str, events: List[DetectionEvent]
    ) -> Optional[float]:
        """처리 지연을 검출 공백으로 오인하지 않고 같은 영상 세대에서만 연결한다."""
        clocks = []
        for event in events:
            metadata = event.metadata if event.metadata is not None else {}
            if not isinstance(metadata, Mapping):
                self._tracks.pop(camera_name, None)
                return None
            if "frame_identity" not in metadata:
                clocks.append((None, None))
                continue
            identity = metadata["frame_identity"]
            if not isinstance(identity, Mapping):
                self._tracks.pop(camera_name, None)
                return None
            generation, pts = identity.get("generation"), identity.get("source_pts_ns")
            if (
                identity.get("camera_id") != camera_name
                or type(generation) is not int or generation < 0
                or type(pts) is not int or not 0 <= pts < 2**64 - 1
            ):
                self._tracks.pop(camera_name, None)
                return None
            clocks.append((generation, pts / 1_000_000_000))
        generation, pts = clocks[0]
        if any(clock != clocks[0] for clock in clocks):
            self._tracks.pop(camera_name, None)
            return None
        previous_generation, previous_pts = self._clocks.get(camera_name, (None, None))
        if generation is not None and previous_generation is not None and generation < previous_generation:
            return None
        if (
            generation != previous_generation
            or (pts is None) != (previous_pts is None)
            or (pts is not None and previous_pts is not None and pts < previous_pts)
        ):
            self._tracks.pop(camera_name, None)
        self._clocks[camera_name] = (generation, pts)
        return time.time() if pts is None else pts

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
        ex2 = event.x + event.width
        ey2 = event.y + event.height
        for track_id, (_, tracked_event) in tracks.items():
            if tracked_event.event_type != event.event_type:
                continue
            # 빠른 비중첩 사전 검사 — IoU 계산보다 훨씬 저렴
            if (
                ex2 <= tracked_event.x
                or tracked_event.x + tracked_event.width <= event.x
                or ey2 <= tracked_event.y
                or tracked_event.y + tracked_event.height <= event.y
            ):
                continue
            iou = event_iou(event, tracked_event)
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
                if best_iou > 0.85:   # 충분히 높은 매칭 → 조기 종료
                    break
        return best_track_id, best_iou

    def _allocate(self) -> int:
        object_id = self._next_object_id
        self._next_object_id += 1
        return object_id
