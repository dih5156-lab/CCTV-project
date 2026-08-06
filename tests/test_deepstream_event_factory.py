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
            "fall_score": 3.5,
            "fall_reasons": ["torso_horizontal:12.0", "low_vertical_span:0.20"],
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
    assert events[0].metadata["fall_score"] == 3.5
    assert events[0].metadata["fall_reasons"] == [
        "torso_horizontal:12.0",
        "low_vertical_span:0.20",
    ]
    assert events[1].metadata["derived_from"] == "pose"
    assert events[1].metadata["fall_score"] == 3.5
    assert events[1].metadata["fall_reasons"] == [
        "torso_horizontal:12.0",
        "low_vertical_span:0.20",
    ]
    fall_payload = events[1].to_dict()
    assert fall_payload["metadata"]["skeleton_keypoints"] == [[1.0, 2.0, 0.9]]
    assert fall_payload["metadata"]["skeleton_format"] == "coco17_xyc"


def test_detections_to_events_keeps_fall_near_miss_on_person_event():
    detections = [
        {
            "box": (1, 2, 3, 4),
            "confidence": 0.9,
            "class_id": 0,
            "label": "person",
            "keypoints": [[1.0, 2.0, 0.9]],
            "is_fall": False,
            "fall_near_miss": {
                "type": "folded_floor_pose",
                "score": 3.0,
                "reasons": ["folded_floor_pose:0.20"],
            },
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

    assert [event.event_type for event in events] == [EventType.PERSON]
    assert events[0].metadata["fall_near_miss"]["type"] == "folded_floor_pose"


def test_detections_to_events_preserves_detail_without_changing_event_type():
    detections = [
        {
            "box": (1, 2, 3, 4),
            "confidence": 0.9,
            "class_id": 0,
            "label": "person",
            "is_fall": True,
            "fall_direction": "뒤",
            "fall_type": "뒤로 넘어짐",
            "scene_cat_name": "후면낙상",
            "fall_detail_source": "direction_classifier_v1",
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

    assert events[-1].event_type == EventType.FALL_DETECTED
    assert events[-1].metadata["fall_detail_status"] == "classified"
    assert events[-1].metadata["fall_direction"] == "뒤"
    assert events[-1].metadata["fall_type"] == "뒤로 넘어짐"
    assert events[-1].metadata["scene_cat_name"] == "후면낙상"
    assert events[-1].metadata["fall_detail_source"] == "direction_classifier_v1"


def test_detections_to_events_marks_missing_detail_as_unclassified():
    detections = [
        {
            "box": (1, 2, 3, 4),
            "confidence": 0.9,
            "class_id": 0,
            "label": "person",
            "is_fall": True,
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

    assert events[-1].metadata["fall_detail_status"] == "unclassified"


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
