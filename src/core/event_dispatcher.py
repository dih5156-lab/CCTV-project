"""이벤트 큐 적재와 큐 포화 시 백업 처리를 담당하는 헬퍼."""

import logging
from queue import Full, Queue
from typing import Dict

logger = logging.getLogger(__name__)


class EventDispatcher:
    """이벤트 payload를 비블로킹으로 큐에 적재한다."""

    def __init__(self, event_queue: Queue, backup_store, increment_stat) -> None:
        self._event_queue = event_queue
        self._backup_store = backup_store
        self._increment_stat = increment_stat

    def enqueue(self, camera_id: str, event_data: Dict) -> bool:
        """큐 적재 성공 여부를 반환한다."""
        try:
            self._event_queue.put_nowait(event_data)
            self._increment_stat("events_detected")
            return True
        except Full:
            self._increment_stat("events_dropped")
            self._backup_store.save_locally(event_data)
            logger.warning("[%s] 이벤트 큐 가득 참: 로컬 저장", camera_id)
            return False
