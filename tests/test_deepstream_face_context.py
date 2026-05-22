"""DeepStream 얼굴 인식 context 후처리 테스트."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.core._deepstream_face_context import (
    remove_camera_face_cache,
    run_deepstream_face_recognition,
)
from src.core.events import DetectionEvent, EventType


def _person_event(object_id: int = 7) -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.PERSON,
        x=10,
        y=20,
        width=40,
        height=80,
        confidence=0.9,
        timestamp=1000.0,
        object_id=object_id,
        class_name="person",
    )


class FakeRecognizer:
    enabled = True
    backend_name = "fake-face"

    def __init__(self) -> None:
        self.calls = 0

    def detect_and_recognize(self, frame, roi):
        self.calls += 1
        return [
            SimpleNamespace(
                matched=True,
                label="tester",
                confidence=0.91,
                age=31.2,
                gender="male",
                bbox={"x": roi["x"] + 1, "y": roi["y"] + 2, "width": 12, "height": 13},
            )
        ]


def test_run_deepstream_face_recognition_builds_metadata_and_snapshot(caplog):
    recognizer = FakeRecognizer()
    cache = {}
    saved = []

    caplog.set_level("INFO")
    events = run_deepstream_face_recognition(
        frame=np.zeros((120, 120, 3), dtype=np.uint8),
        person_events=[_person_event()],
        camera_name="cam01",
        recognizer=recognizer,
        cache=cache,
        timestamp_factory=lambda: 1000.0,
        snapshot_saver=lambda *args: saved.append(args) or "snap.jpg",
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == EventType.FACE_RECOGNIZED
    assert event.object_id == 7
    assert event.metadata["backend"] == "deepstream_context"
    assert event.metadata["camera_id"] == "cam01"
    assert event.metadata["recognizer"] == "fake-face"
    assert event.metadata["age_group"] == "30대"
    assert event.metadata["snapshot_path"] == "snap.jpg"
    assert len(saved) == 1
    assert "이름=tester" in caplog.text


def test_run_deepstream_face_recognition_reuses_recent_cache():
    recognizer = FakeRecognizer()
    cache = {}
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    first = run_deepstream_face_recognition(
        frame=frame,
        person_events=[_person_event()],
        camera_name="cam01",
        recognizer=recognizer,
        cache=cache,
        timestamp_factory=lambda: 1000.0,
        snapshot_saver=lambda *args: None,
    )
    second = run_deepstream_face_recognition(
        frame=frame,
        person_events=[_person_event()],
        camera_name="cam01",
        recognizer=recognizer,
        cache=cache,
        timestamp_factory=lambda: 1001.0,
        snapshot_saver=lambda *args: None,
    )

    assert recognizer.calls == 1
    assert second[0] is first[0]


def test_remove_camera_face_cache_removes_only_matching_camera():
    cache = {
        ("cam01", 1): {"timestamp": 1000.0},
        ("cam02", 1): {"timestamp": 1000.0},
    }

    remove_camera_face_cache(cache, "cam01")

    assert ("cam01", 1) not in cache
    assert ("cam02", 1) in cache
