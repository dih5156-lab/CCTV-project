"""얼굴 인식 파이프라인 테스트."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.core.ai._face_recognition_pipeline import FaceRecognitionPipeline
from src.core.events import DetectionEvent, EventType


def _person_event() -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.PERSON,
        x=10,
        y=20,
        width=40,
        height=80,
        confidence=0.9,
        timestamp=1000.0,
        object_id=7,
        class_name="person",
    )


class FakeRecognizer:
    enabled = True
    backend_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def detect_and_recognize(self, frame, roi):
        self.calls += 1
        return [
            SimpleNamespace(
                matched=True,
                label="tester",
                confidence=0.91,
                decision="matched",
                person_id="employee-1",
                category="employee",
                model_id="opencv-sface-tensorrt-v1",
                second_best_similarity=0.42,
                margin=0.49,
                age=31.2,
                gender="male",
                bbox={"x": roi["x"] + 1, "y": roi["y"] + 2, "width": 12, "height": 13},
            )
        ]


def test_face_recognition_pipeline_builds_event_metadata():
    recognizer = FakeRecognizer()
    pipeline = FaceRecognitionPipeline(lambda: recognizer)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    events = pipeline.run(frame, [_person_event()])

    assert len(events) == 1
    event = events[0]
    assert event.event_type == EventType.FACE_RECOGNIZED
    assert event.object_id == 7
    assert event.metadata["face_name"] == "tester"
    assert event.metadata["recognizer"] == "fake"
    assert event.metadata["age_group"] == "30대"
    assert event.metadata["gender"] == "male"
    assert event.metadata["face_decision"] == "matched"
    assert event.metadata["face_person_id"] == "employee-1"
    assert event.metadata["face_category"] == "employee"
    assert event.metadata["face_model_id"] == "opencv-sface-tensorrt-v1"
    assert event.metadata["face_second_best_score"] == 0.42
    assert event.metadata["face_margin"] == 0.49


def test_face_recognition_pipeline_reuses_recent_cache():
    recognizer = FakeRecognizer()
    pipeline = FaceRecognitionPipeline(lambda: recognizer)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    first = pipeline.run(frame, [_person_event()])
    second = pipeline.run(frame, [_person_event()])

    assert recognizer.calls == 1
    assert second[0] is first[0]
