"""ActionBridge REST 이벤트 큐 처리 헬퍼."""

from __future__ import annotations

import logging
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Dict, Tuple

from .cctv_metrics import rest_action_queue_depth, rest_events_dropped

logger = logging.getLogger(__name__)


def new_rest_action_queue(max_size: int) -> Queue[Tuple[str, Dict]]:
    return Queue(maxsize=max_size)


def new_rest_action_worker_stop() -> Event:
    return Event()


def enqueue_rest_event(bridge: Any, payload: Dict, topic: str = "rest/inbound") -> bool:
    """REST 수신 이벤트를 백그라운드 큐에 넣는다."""
    bridge._start_rest_action_worker()
    try:
        bridge._rest_action_queue.put_nowait((topic, dict(payload)))
        rest_action_queue_depth.set(bridge._rest_action_queue.qsize())
        return True
    except Full:
        logger.error("REST action queue 가득 참 - 이벤트 거부: topic=%s", topic)
        rest_events_dropped.labels(reason="queue_full").inc()
        rest_action_queue_depth.set(bridge._rest_action_queue.qsize())
        return False


def start_rest_action_worker(bridge: Any) -> None:
    """REST action worker를 필요할 때 시작한다."""
    if bridge._rest_action_worker and bridge._rest_action_worker.is_alive():
        return
    bridge._rest_action_worker_stop.clear()
    bridge._rest_action_worker = Thread(
        target=bridge._rest_action_worker_loop,
        daemon=True,
        name="RestActionWorker",
    )
    bridge._rest_action_worker.start()


def rest_action_worker_loop(bridge: Any) -> None:
    """REST 이벤트 큐를 소비해 실제 액션을 수행한다."""
    while (
        not bridge._rest_action_worker_stop.is_set()
        or not bridge._rest_action_queue.empty()
    ):
        try:
            topic, payload = bridge._rest_action_queue.get(timeout=0.2)
        except Empty:
            continue
        try:
            bridge._handle_event(payload, topic=topic)
        except Exception as exc:
            logger.error("REST action worker 처리 오류: %s", exc, exc_info=True)
        finally:
            bridge._rest_action_queue.task_done()
            rest_action_queue_depth.set(bridge._rest_action_queue.qsize())


def stop_rest_action_worker(bridge: Any) -> None:
    """REST action worker를 종료한다."""
    bridge._rest_action_worker_stop.set()
    if bridge._rest_action_worker and bridge._rest_action_worker.is_alive():
        bridge._rest_action_worker.join(timeout=5.0)
