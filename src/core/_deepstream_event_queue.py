"""DeepStream 이벤트 큐 적재 헬퍼."""

from __future__ import annotations

import logging
from queue import Full
from typing import Any, Callable, Dict, Iterable, List, Sequence

from ..utils.zone_detection import ZoneEvent
from .events import DetectionEvent

logger = logging.getLogger(__name__)


def enqueue_queue_item(
    *,
    event_queue: Any,
    queue_item: Any,
    camera_name: str,
    increment_stat: Callable[[str], int],
) -> bool:
    """DeepStream 이벤트 큐 적재와 관련 통계를 한 곳에서 처리한다."""
    try:
        event_queue.put_nowait(queue_item)
        increment_stat("events_detected")
        return True
    except Full:
        increment_stat("events_dropped")
        logger.warning("[%s] DeepStream 이벤트 큐 가득 참", camera_name)
        return False


def enqueue_zone_events(
    *,
    camera_name: str,
    zone_events: Iterable[Any],
    enqueue_event_dict: Callable[[Dict[str, Any], str], bool],
) -> None:
    for zone_event in zone_events:
        event_dict = zone_event.to_dict()
        if "type" not in event_dict:
            event_dict["type"] = event_dict.get("event_type")
        event_dict.setdefault("camera_id", camera_name)
        event_dict["backend"] = "deepstream"
        enqueue_event_dict(event_dict, camera_name)


def apply_existing_event_pipeline(
    *,
    camera_name: str,
    events: Sequence[DetectionEvent],
    assign_synthetic_object_ids: Callable[[str, List[DetectionEvent]], List[DetectionEvent]],
    track_manager: Any,
    violation_filter: Any,
    submit_face_work: Callable[[str, List[DetectionEvent]], None],
    zone_manager: Any,
    enqueue_zone_events_cb: Callable[[str, List[ZoneEvent]], None],
    enqueue_event: Callable[[DetectionEvent, str], bool],
    add_filtered_event_count: Callable[[int], None],
) -> None:
    """DeepStream raw/tensor 이벤트에 tracking/filtering/zone 후처리를 적용한다."""
    if not events:
        return

    prepared_events = assign_synthetic_object_ids(camera_name, list(events))
    tracked_events, removed_ids = track_manager.update(camera_name, prepared_events)
    if removed_ids:
        violation_filter.purge(camera_name, removed_ids)

    filtered_events = violation_filter.filter(camera_name, tracked_events)
    add_filtered_event_count(max(0, len(tracked_events) - len(filtered_events)))

    submit_face_work(camera_name, filtered_events)

    zone_events: List[ZoneEvent] = []
    if zone_manager is not None:
        try:
            zone_events = zone_manager.check_zones(camera_name, filtered_events)
        except Exception as exc:
            logger.warning("[%s] DeepStream 구역 감지 오류: %s", camera_name, exc)
    enqueue_zone_events_cb(camera_name, zone_events)

    for event in filtered_events:
        enqueue_event(event, camera_name)
