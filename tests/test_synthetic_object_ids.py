"""임시 object_id 부여 유틸리티 테스트."""

from __future__ import annotations

import time

import pytest

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


def test_two_overlapping_people_get_distinct_ids_in_same_frame():
    assigner = SyntheticObjectIdAssigner(track_iou=0.3, track_timeout=1.0)
    first = _video_event(0)
    second = _video_event(0)
    second.x = 8  # IoU 약0.43: NMS0.45를 통과하는 서로 다른 검출이다.
    result = assigner.assign("cam01", [first, second])
    assert len({event.object_id for event in result}) == 2
    next_first, next_second = _video_event(100_000_000), _video_event(100_000_000)
    next_first.x, next_second.x = 1, 9
    assigner.assign("cam01", [next_second, next_first])
    assert next_first.object_id == first.object_id
    assert next_second.object_id == second.object_id


@pytest.mark.parametrize("native_first", [True, False])
def test_provided_id_is_reserved_even_when_unassigned_event_comes_first(native_first):
    assigner = SyntheticObjectIdAssigner(track_iou=0.3, track_timeout=1.0)
    native, unknown = _video_event(0), _video_event(0)
    native.object_id = 1
    unknown.x = 8
    events = [native, unknown] if native_first else [unknown, native]
    assigner.assign("cam01", events)
    assert native.object_id == 1
    assert unknown.object_id != native.object_id


def test_existing_track_cannot_be_used_by_two_observations():
    assigner = SyntheticObjectIdAssigner(track_iou=0.3, track_timeout=1.0)
    original = assigner.assign("cam01", [_video_event(0)])[0]
    first, second = _video_event(100_000_000), _video_event(100_000_000)
    second.x = 8
    assigner.assign("cam01", [first, second])
    assert first.object_id == original.object_id
    assert first.object_id != second.object_id


def _video_event(pts_ns, *, camera="cam01", generation=1):
    event = _event(0, 0)
    event.metadata = {"frame_identity": {
        "camera_id": camera, "generation": generation, "source_pts_ns": pts_ns,
    }}
    return event


def test_video_track_survives_processing_delay(monkeypatch):
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=1.0)
    monkeypatch.setattr(time, "time", lambda: 100.0)
    first = assigner.assign("cam01", [_video_event(0)])[0]
    monkeypatch.setattr(time, "time", lambda: 110.0)
    second = assigner.assign("cam01", [_video_event(100_000_000)])[0]
    assert first.object_id == second.object_id


def test_video_track_expires_on_source_gap_even_when_processing_is_fast(monkeypatch):
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=1.0)
    monkeypatch.setattr(time, "time", lambda: 100.0)
    first = assigner.assign("cam01", [_video_event(0)])[0]
    second = assigner.assign("cam01", [_video_event(2_000_000_000)])[0]
    assert first.object_id != second.object_id


def test_video_track_generation_reset_rejects_late_old_generation():
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=1.0)
    first = assigner.assign("cam01", [_video_event(0)])[0]
    second = assigner.assign("cam01", [_video_event(0, generation=2)])[0]
    assert first.object_id != second.object_id
    assert assigner.assign("cam01", [_video_event(1, generation=1)]) == []
    assert assigner.assign("cam01", [_video_event(2, generation=2)])[0].object_id == second.object_id


@pytest.mark.parametrize("bad", [None, -1, True, 0.1, float("nan"), 2**64 - 1])
def test_invalid_video_pts_does_not_reuse_or_advance_tracks(bad):
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=1.0)
    first = assigner.assign("cam01", [_video_event(0)])[0]
    assert assigner.assign("cam01", [_video_event(bad)]) == []
    assert assigner.assign("cam01", [_video_event(1)])[0].object_id != first.object_id


def test_video_track_reset_keeps_other_camera_and_remove_clears_clock():
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=1.0)
    first = assigner.assign("cam01", [_video_event(100)])[0]
    other = assigner.assign("cam02", [_video_event(100, camera="cam02")])[0]
    assert assigner.assign("cam01", [_video_event(0)])[0].object_id != first.object_id
    assert assigner.assign("cam02", [_video_event(101, camera="cam02")])[0].object_id == other.object_id
    assigner.remove_camera("cam01")
    assert assigner.assign("cam01", [_video_event(0, generation=0)])


def test_video_track_clock_switch_does_not_match_legacy_track():
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=1.0)
    legacy = assigner.assign("cam01", [_event(0, 0)])[0]
    video = assigner.assign("cam01", [_video_event(0)])[0]
    back = assigner.assign("cam01", [_event(0, 0)])[0]
    assert len({legacy.object_id, video.object_id, back.object_id}) == 3


def test_video_track_rejects_camera_mismatch_and_mixed_frame_batch():
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=1.0)
    assert assigner.assign("cam01", [_video_event(0, camera="cam02")]) == []
    assert assigner.assign("cam01", [_video_event(0), _video_event(1)]) == []


def test_video_track_duplicate_pts_keeps_original_observations():
    assigner = SyntheticObjectIdAssigner(track_iou=0.2, track_timeout=1.0)
    first = assigner.assign("cam01", [_video_event(0)])[0]
    duplicate = _video_event(0)
    assert assigner.assign("cam01", [duplicate]) == [duplicate]
    assert duplicate.object_id == first.object_id
    assert duplicate.confidence == 0.9 and duplicate.keypoints is None
    assert assigner.assign("cam01", []) == []
