"""DeepStream detection 이벤트 변환 테스트."""

from __future__ import annotations

from types import SimpleNamespace

from src.core._deepstream_event_factory import (
    detections_to_events,
    object_meta_to_event,
)
from src.core.events import EventType


def test_object_meta_to_event_builds_deepstream_event():
    obj_meta = SimpleNamespace(
        obj_label="person",
        rect_params=SimpleNamespace(left=1, top=2, width=3, height=4),
        object_id=9,
        confidence=0.87,
        class_id=0,
    )

    event = object_meta_to_event(
        obj_meta,
        camera_name="cam01",
        source_id=2,
        frame_num=3,
        timestamp_factory=lambda: 1000.0,
        event_type_for_label=lambda label: EventType.PERSON,
    )

    assert event.event_type == EventType.PERSON
    assert event.object_id == 9
    assert event.class_name == "person"
    assert event.metadata == {
        "backend": "deepstream",
        "camera_id": "cam01",
        "source_id": 2,
        "frame_num": 3,
    }


def test_object_meta_to_event_skips_other_event_type():
    obj_meta = SimpleNamespace(
        obj_label="noise",
        rect_params=SimpleNamespace(left=1, top=2, width=3, height=4),
        object_id=-1,
        confidence=0.1,
        class_id=99,
    )

    event = object_meta_to_event(
        obj_meta,
        camera_name="cam01",
        source_id=2,
        frame_num=3,
        timestamp_factory=lambda: 1000.0,
        event_type_for_label=lambda label: EventType.OTHER,
    )

    assert event is None


def test_detections_to_events_builds_base_and_fall_events():
    detections = [
        {
            "box": (1, 2, 3, 4),
            "confidence": 0.9,
            "class_id": 0,
            "label": "person",
            "keypoints": [[1.0, 2.0, 0.9]],
            "is_fall": True,
            "gie_id": 1,
            "task": "pose",
        }
    ]

    events = detections_to_events(
        detections,
        camera_name="cam01",
        source_id=2,
        frame_num=3,
        timestamp_factory=lambda: 1000.0,
        event_type_for_label=lambda label: EventType.PERSON,
    )

    assert [event.event_type for event in events] == [
        EventType.PERSON,
        EventType.FALL_DETECTED,
    ]
    assert events[0].metadata["backend"] == "deepstream_tensor"
    assert events[0].metadata["camera_id"] == "cam01"
    assert events[1].metadata["derived_from"] == "pose"


def test_detections_to_events_skips_unknown_labels():
    events = detections_to_events(
        [{"box": (1, 2, 3, 4), "confidence": 0.9, "class_id": 99, "label": "noise"}],
        camera_name="cam01",
        source_id=2,
        frame_num=3,
        timestamp_factory=lambda: 1000.0,
        event_type_for_label=lambda label: EventType.OTHER,
    )

    assert events == []
