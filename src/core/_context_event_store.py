"""최근 DetectionEvent 컨텍스트 저장소."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Callable, Dict, List, Tuple

from .events import DetectionEvent, EventType

_EventKey = Tuple[str, int, int, int, int, int | None]


class ContextEventStore:
    """외형 분석에 필요한 최근 주변 이벤트를 카메라별로 짧게 보관한다."""

    def __init__(
        self,
        *,
        ttl_sec: float,
        maxlen: int,
        time_factory: Callable[[], float] = time.time,
    ) -> None:
        self._ttl_sec = ttl_sec
        self._maxlen = max(16, maxlen)
        self._time_factory = time_factory
        self._events: Dict[str, deque] = {}
        self._lock = Lock()

    def clear_camera(self, camera_id: str) -> None:
        """카메라 제거 시 해당 카메라의 context cache를 정리한다."""
        with self._lock:
            self._events.pop(camera_id, None)

    def remember(self, camera_id: str, events: List[DetectionEvent]) -> None:
        """현재 frame의 주변 이벤트를 cache에 저장한다."""
        if not events:
            return
        now = self._time_factory()
        cutoff = self._cutoff(now)
        with self._lock:
            bucket = self._bucket(camera_id)
            self._drop_stale(bucket, cutoff)
            for event in events:
                if event.event_type == EventType.PERSON:
                    continue
                bucket.append((now, event))

    def collect(
        self,
        camera_id: str,
        current_events: List[DetectionEvent],
    ) -> List[DetectionEvent]:
        """현재 frame 이벤트와 최근 cached 이벤트를 중복 없이 병합한다."""
        now = self._time_factory()
        cutoff = self._cutoff(now)
        merged: List[DetectionEvent] = list(current_events)
        seen_keys = {self._event_key(event) for event in current_events}

        with self._lock:
            bucket = self._bucket(camera_id)
            self._drop_stale(bucket, cutoff)
            for _, event in bucket:
                key = self._event_key(event)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append(event)
        return merged

    def _bucket(self, camera_id: str) -> deque:
        return self._events.setdefault(camera_id, deque(maxlen=self._maxlen))

    def _cutoff(self, now: float) -> float:
        return now - max(0.1, self._ttl_sec)

    @staticmethod
    def _drop_stale(bucket: deque, cutoff: float) -> None:
        while bucket and bucket[0][0] < cutoff:
            bucket.popleft()

    @staticmethod
    def _event_key(event: DetectionEvent) -> _EventKey:
        return (
            event.event_type.value,
            int(event.x),
            int(event.y),
            int(event.width),
            int(event.height),
            int(event.object_id) if event.object_id is not None else None,
        )
