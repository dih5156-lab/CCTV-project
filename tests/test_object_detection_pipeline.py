"""YOLO 객체탐지 파이프라인 smoke 테스트."""

from __future__ import annotations

import numpy as np

from src.core.ai._object_detection_pipeline import ObjectDetectionPipeline
from src.core.events import DetectionEvent, EventType


def _person_event() -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.PERSON,
        x=10,
        y=20,
        width=30,
        height=80,
        confidence=0.9,
        timestamp=1000.0,
        object_id=7,
        class_name="person",
    )


def _appearance_event() -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.APPEARANCE_MATCH,
        x=10,
        y=20,
        width=30,
        height=80,
        confidence=0.95,
        timestamp=1000.1,
        object_id=7,
        class_name="person",
        metadata={"upper_color": "red"},
    )


class DummyAnalyzer:
    def __init__(self) -> None:
        self.pose_model = object()
        self.person_model = None
        self.helmet_model = object()
        self.confidence_threshold = 0.5
        self._person_warning_shown = False
        self._helmet_warning_shown = False
        self._last_bag_objects = [
            {"class_name": "backpack", "x": 8, "y": 35, "width": 20, "height": 30}
        ]
        self.appearance_nearby_objects = None

    def _run_pose_full_frame(self, frame):
        return [_person_event()], []

    def _run_helmet_on_person_rois(self, frame, person_events):
        return [
            DetectionEvent(
                event_type=EventType.HELMET,
                x=12,
                y=20,
                width=14,
                height=14,
                confidence=0.88,
                timestamp=1000.0,
                object_id=7,
                class_name="helmet",
            )
        ]

    def _filter_helmet_boxes(self, helmet_events):
        return helmet_events

    def _run_face_recognition(self, frame, person_events):
        return []

    def _build_appearance_nearby_objects(self, bag_objects, helmet_events):
        return list(bag_objects) + [
            {"class_name": event.class_name, "event_type": event.event_type.value}
            for event in helmet_events
        ]

    def _run_appearance_pipeline(self, *args, **kwargs):
        self.appearance_nearby_objects = kwargs["nearby_objects"]
        return [_appearance_event()]


def test_object_detection_pipeline_forwards_yolo_context_to_appearance():
    analyzer = DummyAnalyzer()
    pipeline = ObjectDetectionPipeline(analyzer)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    events = pipeline.run(
        frame,
        use_helmet=True,
        use_pose=True,
        use_person=False,
        use_face=False,
        use_appearance=True,
        camera_id="cam01",
    )

    assert [event.event_type for event in events] == [
        EventType.PERSON,
        EventType.HELMET,
        EventType.APPEARANCE_MATCH,
    ]
    assert analyzer.appearance_nearby_objects == [
        {"class_name": "backpack", "x": 8, "y": 35, "width": 20, "height": 30},
        {"class_name": "helmet", "event_type": "helmet"},
    ]
