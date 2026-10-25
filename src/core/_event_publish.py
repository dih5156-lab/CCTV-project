"""이벤트 큐 항목 발행 유틸리티."""

from __future__ import annotations

import logging
import os
from queue import Empty
from typing import Any, Callable, Dict, Optional, Tuple

from ..canonical_event import canonicalize_event_payload
from ..time_utils import now_kst_iso
from ..video_observation import build_video_observation
from .events import DetectionEvent

logger = logging.getLogger(__name__)


def normalize_event_queue_item(
    queue_item: Any,
    *,
    default_backend: Optional[str] = "deepstream",
) -> Tuple[Dict[str, Any], str, str]:
    """큐 항목을 MQTT 발행용 dict, event_type, camera_id로 정규화한다."""
    if isinstance(queue_item, DetectionEvent):
        event_data = queue_item.to_dict()
        event_type = queue_item.event_type.value
    else:
        event_data = dict(queue_item)
        event_type = str(
            event_data.get("type") or event_data.get("event_type") or "unknown"
        )

    metadata = event_data.get("metadata") or {}
    camera_id = event_data.get("camera_id", metadata.get("camera_id", "unknown"))
    event_data["camera_id"] = camera_id
    if default_backend is not None:
        event_data.setdefault("backend", default_backend)
    source_payload = event_data
    event_data = canonicalize_event_payload(event_data)
    if (
        isinstance(queue_item, DetectionEvent)
        and isinstance(camera_id, str)
        and camera_id.strip().lower() not in ("", "unknown")
    ):
        event_data["observation"] = build_video_observation(
            source_payload,
            namespace="video_detection",
            source_record_id=event_data["event_id"],
            device_id=str(camera_id),
            received_at=now_kst_iso(),
        )
    return event_data, event_type, str(camera_id)


def publish_queue_item(
    queue_item: Any,
    *,
    topic_prefix: str,
    mqtt_publish: Optional[Callable[[str, dict], None]],
    event_publisher: Any,
    backend: Optional[str] = "deepstream",
) -> bool:
    """큐 항목 1개를 발행하고 성공 여부를 반환한다."""
    if os.environ.get("EDGE_RESULT_CONTEXT_DB"):
        raise ValueError("edge results require durable video intake")
    event_data, event_type, camera_id = normalize_event_queue_item(
        queue_item,
        default_backend=backend,
    )
    if mqtt_publish is not None:
        topic = f"{topic_prefix}/{camera_id}/{event_type}"
        mqtt_publish(topic, event_data)
        return True
    return bool(event_publisher.publish_event(event_data))


def run_publish_loop(
    *,
    is_running: Callable[[], bool],
    stop_event: Any,
    event_queue: Any,
    topic_prefix: str,
    mqtt_publish: Optional[Callable[[str, dict], None]],
    event_publisher: Any,
    increment_stat: Callable[[str], int],
    queue_timeout_sec: float = 1.0,
    video_ingest: Any = None,
) -> None:
    """이벤트 큐를 소비하며 publish 통계를 누적한다."""
    while is_running() and not stop_event.is_set():
        if video_ingest is not None:
            video_ingest.poll_recovery()
        try:
            queue_item = event_queue.get(timeout=queue_timeout_sec)
        except Empty:
            continue

        try:
            if video_ingest is not None:
                published = video_ingest.publish(
                    queue_item,
                    mqtt_publish=mqtt_publish,
                    event_publisher=event_publisher,
                )
            else:
                published = publish_queue_item(
                    queue_item,
                    topic_prefix=topic_prefix,
                    mqtt_publish=mqtt_publish,
                    event_publisher=event_publisher,
                )
            if published:
                increment_stat("events_sent")
                continue
            increment_stat("events_failed")
        except Exception as exc:
            logger.error("MQTT 발행 오류: %s", exc)
            increment_stat("events_failed")
