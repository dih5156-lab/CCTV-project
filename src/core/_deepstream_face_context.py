"""DeepStream context 얼굴 인식 후처리."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from .ai._constants import _FACE_TRACK_COOLDOWN_SEC
from .ai._yolo_helpers import age_to_group
from .events import DetectionEvent, EventType


FaceCache = Dict[Tuple[str, int], Dict[str, Any]]
logger = logging.getLogger(__name__)


def remove_camera_face_cache(cache: FaceCache, camera_id: str) -> None:
    """카메라 제거 시 해당 카메라의 얼굴 인식 cache를 정리한다."""
    for key in [key for key in cache if key[0] == camera_id]:
        cache.pop(key, None)


def run_deepstream_face_recognition(
    *,
    frame: Any,
    person_events: List[DetectionEvent],
    camera_name: str,
    recognizer: Any,
    cache: FaceCache,
    timestamp_factory: Callable[[], float],
    snapshot_saver: Callable[[Any, str, str, Dict[str, int], float, float], str | None],
) -> List[DetectionEvent]:
    """DeepStream preview frame과 person 이벤트를 얼굴 인식 이벤트로 확장한다."""
    if frame is None or not person_events or not getattr(recognizer, "enabled", False):
        return []

    face_events: List[DetectionEvent] = []
    now = timestamp_factory()
    for person in person_events:
        if person.object_id is None:
            continue

        cache_key = (camera_name, int(person.object_id))
        cached_event = _cached_event(cache, cache_key, now)
        if cached_event is not None:
            face_events.append(cached_event)
            continue

        results = recognizer.detect_and_recognize(
            frame,
            {"x": person.x, "y": person.y, "width": person.width, "height": person.height},
        )
        if not results:
            cache.pop(cache_key, None)
            continue

        best = max(results, key=lambda item: item.confidence)
        event = _build_face_event(
            best=best,
            person=person,
            frame=frame,
            camera_name=camera_name,
            recognizer_name=getattr(recognizer, "backend_name", "unknown"),
            now=now,
            snapshot_saver=snapshot_saver,
        )
        face_events.append(event)
        cache[cache_key] = {"timestamp": now, "event": event}

    _cleanup_cache(cache, now)
    return face_events


def _cached_event(cache: FaceCache, cache_key: Tuple[str, int], now: float) -> DetectionEvent | None:
    cached = cache.get(cache_key)
    if not cached:
        return None
    if now - float(cached.get("timestamp", 0.0)) >= _FACE_TRACK_COOLDOWN_SEC:
        return None
    return cached.get("event")


def _build_face_event(
    *,
    best: Any,
    person: DetectionEvent,
    frame: Any,
    camera_name: str,
    recognizer_name: str,
    now: float,
    snapshot_saver: Callable[[Any, str, str, Dict[str, int], float, float], str | None],
) -> DetectionEvent:
    event_type = EventType.FACE_RECOGNIZED if best.matched else EventType.FACE_UNKNOWN
    face_meta: Dict[str, object] = {
        "backend": "deepstream_context",
        "camera_id": camera_name,
        "person_object_id": person.object_id,
        "face_name": best.label,
        "face_score": round(best.confidence, 4),
        "recognizer": recognizer_name,
    }
    if best.age is not None:
        face_meta["age"] = round(best.age, 1)
        face_meta["age_group"] = age_to_group(best.age)
    if best.gender is not None:
        face_meta["gender"] = best.gender

    snapshot_path = (
        snapshot_saver(frame, camera_name, best.label, best.bbox, best.confidence, now)
        if best.matched
        else None
    )
    if snapshot_path:
        face_meta["snapshot_path"] = snapshot_path

    if best.matched:
        logger.info(
            "[얼굴] camera=%s track=%s 이름=%s score=%.4f",
            camera_name,
            person.object_id,
            best.label,
            best.confidence,
        )

    return DetectionEvent(
        event_type=event_type,
        x=best.bbox["x"],
        y=best.bbox["y"],
        width=best.bbox["width"],
        height=best.bbox["height"],
        confidence=best.confidence,
        timestamp=now,
        object_id=person.object_id,
        class_name="face",
        metadata=face_meta,
    )


def _cleanup_cache(cache: FaceCache, now: float) -> None:
    stale_keys = [
        key for key, item in cache.items()
        if now - float(item.get("timestamp", 0.0)) > (_FACE_TRACK_COOLDOWN_SEC * 5)
    ]
    for key in stale_keys:
        cache.pop(key, None)
