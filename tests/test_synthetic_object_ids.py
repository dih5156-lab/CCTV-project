"""임시 object_id 부여 유틸리티 테스트."""

from __future__ import annotations

import time

from src.core._synthetic_object_ids import SyntheticObjectIdAssigner, event_iou
from src.core.events import DetectionEvent, EventType


def _event(x: int, y: int, object_id: int | None = None) -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.PERSON,
        x=x,
        y=y,
        width=20,
        height=40,
        confidence=0.9,
        timestamp=time.time(),
        object_id=object_id,
    )


def test_event_iou_returns_overlap_ratio():
    first = _event(0, 0)
    second = _event(10, 0)

    assert round(event_iou(first, second), 4) == 0.3333


def test_synthetic_object_id_assigner_reuses_overlapping_track():
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=10.0)

    first = assigner.assign("cam01", [_event(0, 0)])[0]
    second = assigner.assign("cam01", [_event(3, 0)])[0]

    assert first.object_id == 1
    assert second.object_id == 1


def test_synthetic_object_id_assigner_allocates_for_distant_track():
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=10.0)

    first = assigner.assign("cam01", [_event(0, 0)])[0]
    second = assigner.assign("cam01", [_event(100, 0)])[0]

    assert first.object_id == 1
    assert second.object_id == 2
