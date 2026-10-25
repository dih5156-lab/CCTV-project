"""이벤트 큐 적재와 큐 포화 시 백업 처리를 담당하는 헬퍼."""

# Jetson B의 Python 3.8에서 union 타입 표기의 즉시 평가를 피한다.
from __future__ import annotations

import logging
from queue import Full, Queue
from typing import Dict

from .events import DetectionEvent

logger = logging.getLogger(__name__)


class EventDispatcher:
    """선택한 영속 접수를 수행한 뒤 이벤트를 큐에 적재한다."""

    def __init__(
        self, event_queue: Queue, backup_store, increment_stat, *, video_ingest=None
    ) -> None:
        self._event_queue = event_queue
        self._backup_store = backup_store
        self._increment_stat = increment_stat
        self._video_ingest = video_ingest

    def enqueue(self, camera_id: str, event_data: Dict | DetectionEvent) -> bool:
        """큐 적재 성공 여부를 반환한다."""
        if self._video_ingest is not None:
            try:
                event_data = self._video_ingest.accept(event_data, camera_id)
            except Exception:
                self._increment_stat("events_failed")
                raise
        elif isinstance(event_data, DetectionEvent):
            event_data = event_data.to_dict()
            event_data["camera_id"] = camera_id
        try:
            self._event_queue.put_nowait(event_data)
            self._increment_stat("events_detected")
            return True
        except Full:
            self._increment_stat("events_dropped")
            if self._video_ingest is not None:
                self._video_ingest.defer(event_data)
            else:
                self._backup_store.save_locally(event_data)
            logger.warning("[%s] 이벤트 큐 가득 참: 로컬 저장", camera_id)
            return False
