"""이벤트 큐 발행 유틸리티 테스트."""

from __future__ import annotations

from src.core._event_publish import normalize_event_queue_item, publish_queue_item
from src.core.events import DetectionEvent, EventType


def test_normalize_event_queue_item_from_detection_event():
    event = DetectionEvent(
        event_type=EventType.PERSON,
        x=1,
        y=2,
        width=3,
        height=4,
        confidence=0.8,
        timestamp=1000.0,
        metadata={"camera_id": "cam01"},
    )

    event_data, event_type, camera_id = normalize_event_queue_item(event)

    assert event_type == "person"
    assert camera_id == "cam01"
    assert event_data["camera_id"] == "cam01"
    assert event_data["backend"] == "deepstream"
    assert event_data["schema_version"] == "1.0"
    assert event_data["message_type"] == "ai_detection_event"
    assert event_data["device"]["camera_id"] == "cam01"
    assert event_data["event"]["event_type"] == "person"
    assert event_data["event_id"].startswith("evt_")


def test_publish_queue_item_uses_callback_topic():
    published = []

    ok = publish_queue_item(
        {"type": "helmet", "metadata": {"camera_id": "cam02"}},
        topic_prefix="cctv/events",
        mqtt_publish=lambda topic, payload: published.append((topic, payload)),
        event_publisher=None,
    )

    assert ok is True
    assert published[0][0] == "cctv/events/cam02/helmet"
    assert published[0][1]["backend"] == "deepstream"
    assert published[0][1]["schema_version"] == "1.0"


def test_publish_queue_item_falls_back_to_event_publisher():
    class FakePublisher:
        def __init__(self) -> None:
            self.payloads = []

        def publish_event(self, payload):
            self.payloads.append(payload)
            return True

    publisher = FakePublisher()

    ok = publish_queue_item(
        {"event_type": "fall_detected", "camera_id": "cam03"},
        topic_prefix="unused",
        mqtt_publish=None,
        event_publisher=publisher,
    )

    assert ok is True
    assert publisher.payloads[0]["camera_id"] == "cam03"
    assert publisher.payloads[0]["event"]["event_type"] == "fall_detected"


def test_publish_queue_item_can_set_backend():
    class FakePublisher:
        def __init__(self) -> None:
            self.payloads = []

        def publish_event(self, payload):
            self.payloads.append(payload)
            return True

    publisher = FakePublisher()

    ok = publish_queue_item(
        {"type": "head", "camera_id": "cam04"},
        topic_prefix="unused",
        mqtt_publish=None,
        event_publisher=publisher,
        backend="opencv",
    )

    assert ok is True
    assert publisher.payloads[0]["backend"] == "opencv"
