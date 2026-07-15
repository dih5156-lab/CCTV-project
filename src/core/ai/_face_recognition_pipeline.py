"""얼굴 인식 이벤트 생성 파이프라인."""

from __future__ import annotations

import time
from typing import Callable, Dict, List

from ..events import DetectionEvent, EventType
from ._constants import _FACE_TRACK_COOLDOWN_SEC
from ._yolo_helpers import age_to_group


class FaceRecognitionPipeline:
    """사람 이벤트를 얼굴 인식 이벤트로 확장한다."""

    def __init__(self, recognizer_getter: Callable[[], object]) -> None:
        self._recognizer_getter = recognizer_getter
        self._identity_cache: Dict[int, Dict] = {}

    def run(self, frame, person_events: List[DetectionEvent]) -> List[DetectionEvent]:
        """사람 ROI 상단에서 얼굴 검출/인식을 수행한다."""
        recognizer = self._recognizer_getter()
        if frame is None or not person_events or not getattr(recognizer, "enabled", False):
            return []

        face_events: List[DetectionEvent] = []
        now = time.time()

        for person in person_events:
            object_id = person.object_id
            if object_id is None:
                continue

            cached_event = self._cached_event(object_id, now)
            if cached_event is not None:
                face_events.append(cached_event)
                continue

            results = recognizer.detect_and_recognize(
                frame,
                {"x": person.x, "y": person.y, "width": person.width, "height": person.height},
            )
            if not results:
                self._identity_cache.pop(object_id, None)
                continue

            event = self._build_event(
                best=max(results, key=lambda item: item.confidence),
                object_id=object_id,
                recognizer_name=getattr(recognizer, "backend_name", "unknown"),
                timestamp=now,
            )
            face_events.append(event)
            self._identity_cache[object_id] = {"timestamp": now, "event": event}

        self._cleanup_cache(now)
        return face_events

    def _cached_event(self, object_id: int, now: float) -> DetectionEvent | None:
        cached = self._identity_cache.get(object_id)
        if not cached:
            return None
        if now - float(cached.get("timestamp", 0.0)) >= _FACE_TRACK_COOLDOWN_SEC:
            return None
        return cached.get("event")

    @staticmethod
    def _build_event(
        *,
        best,
        object_id: int,
        recognizer_name: str,
        timestamp: float,
    ) -> DetectionEvent:
        event_type = EventType.FACE_RECOGNIZED if best.matched else EventType.FACE_UNKNOWN
        face_meta: Dict[str, object] = {
            "person_object_id": object_id,
            "face_name": best.label,
            "face_score": round(best.confidence, 4),
            "recognizer": recognizer_name,
        }
        if best.age is not None:
            face_meta["age"] = round(best.age, 1)
            face_meta["age_group"] = age_to_group(best.age)
        if best.gender is not None:
            face_meta["gender"] = best.gender
        optional_metadata = (
            ("decision", "face_decision"),
            ("person_id", "face_person_id"),
            ("category", "face_category"),
            ("model_id", "face_model_id"),
            ("second_best_similarity", "face_second_best_score"),
            ("margin", "face_margin"),
        )
        for attribute_name, metadata_name in optional_metadata:
            value = getattr(best, attribute_name, None)
            if value is not None:
                face_meta[metadata_name] = (
                    round(float(value), 4)
                    if attribute_name in {"second_best_similarity", "margin"}
                    else value
                )

        return DetectionEvent(
            event_type=event_type,
            x=best.bbox["x"],
            y=best.bbox["y"],
            width=best.bbox["width"],
            height=best.bbox["height"],
            confidence=best.confidence,
            timestamp=timestamp,
            object_id=object_id,
            metadata=face_meta,
        )

    def _cleanup_cache(self, now: float) -> None:
        stale_ids = [
            oid for oid, item in self._identity_cache.items()
            if now - float(item.get("timestamp", 0.0)) > (_FACE_TRACK_COOLDOWN_SEC * 5)
        ]
        for oid in stale_ids:
            self._identity_cache.pop(oid, None)
