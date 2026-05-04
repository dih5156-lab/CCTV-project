"""DetectionEvent 컨텍스트 변환 테스트."""

from __future__ import annotations

from src.core._event_context import events_to_nearby_objects
from src.core.events import DetectionEvent, EventType


def test_events_to_nearby_objects_keeps_bbox_and_metadata():
    event = DetectionEvent(
        event_type=EventType.HELMET,
        x=1,
        y=2,
        width=3,
        height=4,
        confidence=0.75,
        timestamp=1000.0,
        class_name="helmet",
        metadata={"camera_id": "cam01"},
    )

    nearby = events_to_nearby_objects([event])

    assert nearby == [
        {
            "class_name": "helmet",
            "event_type": "helmet",
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 4,
            "confidence": 0.75,
            "metadata": {"camera_id": "cam01"},
        }
    ]
