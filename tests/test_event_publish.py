"""이벤트 큐 발행 유틸리티 테스트."""

from __future__ import annotations

from src.canonical_event import canonicalize_event_payload
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


def test_normalize_event_queue_item_promotes_appearance_attributes():
    event = DetectionEvent(
        event_type=EventType.APPEARANCE_MATCH,
        x=10,
        y=20,
        width=30,
        height=40,
        confidence=0.88,
        timestamp=1000.0,
        object_id=7,
        class_name="person",
        metadata={
            "camera_id": "cam-appearance",
            "condition_id": "cond-1",
            "upper_color": "black",
            "lower_color": "gray",
            "has_backpack": True,
            "gender": "male",
            "attribute_backend": "hsv",
        },
    )

    event_data, event_type, camera_id = normalize_event_queue_item(event)

    assert event_type == "appearance_match"
    assert camera_id == "cam-appearance"
    assert event_data["attributes"] == {
        "upper_color": "black",
        "lower_color": "gray",
        "has_backpack": True,
        "gender": "male",
        "attribute_backend": "hsv",
    }
    assert event_data["raw"]["attributes"] == event_data["attributes"]
    assert event_data["raw"]["metadata"]["condition_id"] == "cond-1"


def test_normalize_event_queue_item_preserves_fall_skeleton_metadata():
    keypoints = [[float(idx), float(idx + 1), 0.9] for idx in range(17)]
    event = DetectionEvent(
        event_type=EventType.FALL_DETECTED,
        x=10,
        y=20,
        width=120,
        height=80,
        confidence=0.93,
        timestamp=1000.0,
        object_id=42,
        keypoints=keypoints,
        metadata={"camera_id": "cam-fall"},
    )

    event_data, event_type, camera_id = normalize_event_queue_item(event)

    assert event_type == "fall_detected"
    assert camera_id == "cam-fall"
    assert event_data["keypoints"] == keypoints
    assert event_data["metadata"]["skeleton_keypoints"] == keypoints
    assert event_data["metadata"]["skeleton_format"] == "coco17_xyc"
    assert event_data["raw"]["keypoints"] == keypoints
    assert event_data["raw"]["metadata"]["skeleton_keypoints"] == keypoints


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


def test_publish_partial_event_preserves_origin_and_model_evidence():
    published = []
    event = {
        "event_id": "fall-event-1",
        "camera_id": "cam01",
        "type": "fall_detected",
        "timestamp": 1700000000.0,
        "device": {"camera_id": "cam01", "site_id": "site-1"},
        "event": {"event_type": "fall_detected", "model_version": "test-model"},
        "raw": {"metadata": {"skeleton_format": "coco17_xyc"}},
    }

    assert publish_queue_item(
        event,
        topic_prefix="cctv/events",
        mqtt_publish=lambda topic, payload: published.append((topic, payload)),
        event_publisher=None,
    )

    topic, payload = published[0]
    assert topic == "cctv/events/cam01/fall_detected"
    assert payload["event_id"] == "fall-event-1"
    assert payload["device"] == event["device"]
    assert payload["event"]["model_version"] == "test-model"
    assert payload["raw"]["metadata"] == event["raw"]["metadata"]
    assert payload["occurred_at"] == "2023-11-15T07:13:20+09:00"


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


def test_detection_observation_keeps_id_track_zero_and_attribute_quality():
    event = DetectionEvent(
        event_type=EventType.APPEARANCE_MATCH,
        x=1,
        y=2,
        width=30,
        height=40,
        confidence=0.8,
        timestamp=1700000000.0,
        object_id=0,
        metadata={
            "camera_id": "cam01",
            "has_helmet": False,
            "upper_color": "unknown",
            "attribute_backend": "hsv",
        },
    )
    legacy = event.to_dict()
    legacy.update(camera_id="cam01", backend="deepstream")
    expected_id = canonicalize_event_payload(legacy)["event_id"]
    published = []

    assert publish_queue_item(
        event,
        topic_prefix="cctv/events",
        mqtt_publish=lambda topic, payload: published.append((topic, payload)),
        event_publisher=None,
    )
    topic, payload = published[0]
    observation = payload["observation"]
    assert topic == "cctv/events/cam01/appearance_match"
    assert payload["event_id"] == expected_id == observation["source_record_id"]
    assert observation["device_id"] == "cam01"
    assert observation["track_id"]["value"] == "0"
    assert observation["measurements"]["has_helmet"]["value"] is False
    assert observation["measurements"]["upper_color"]["quality"] == "unknown"
    assert observation["measurements"]["lower_color"]["quality"] == "missing"
    assert observation["processed_at"] == "2023-11-15T07:13:20+09:00"
    assert observation["observed_at"] is None
    assert payload["raw"]["metadata"]["upper_color"] == "unknown"
    assert "observation" not in event.to_dict()
    assert normalize_event_queue_item(payload)[0] == payload


def test_detection_observation_preserves_epoch_zero_before_time_fallback():
    event = DetectionEvent(
        event_type=EventType.PERSON,
        x=0,
        y=0,
        width=1,
        height=1,
        confidence=0.5,
        timestamp=0,
        metadata={"camera_id": "cam01"},
    )
    payload, _, _ = normalize_event_queue_item(event)
    assert payload["observation"]["processed_at"] == "1970-01-01T09:00:00+09:00"


def test_dictionary_events_are_not_guessed_to_be_video_observations():
    payload, _, _ = normalize_event_queue_item(
        {"type": "temperature", "camera_id": "sensor01", "timestamp": 1700000000}
    )
    assert "observation" not in payload


def test_camera_less_detection_keeps_legacy_path_without_a_valid_observation():
    event = DetectionEvent(EventType.PERSON, 0, 0, 1, 1, 0.5, 1700000000)
    payload, _, camera_id = normalize_event_queue_item(event)
    assert camera_id == "unknown"
    assert "observation" not in payload


def test_simultaneous_tracks_have_distinct_observations():
    events = [
        DetectionEvent(
            EventType.PERSON,
            0,
            0,
            1,
            1,
            0.5,
            1700000000,
            object_id=track,
            metadata={"camera_id": "cam01", "frame_num": 7},
        )
        for track in (0, 1)
    ]
    payloads = [normalize_event_queue_item(event)[0] for event in events]
    assert payloads[0]["event_id"] != payloads[1]["event_id"]
    assert (
        payloads[0]["observation"]["observation_id"]
        != payloads[1]["observation"]["observation_id"]
    )
