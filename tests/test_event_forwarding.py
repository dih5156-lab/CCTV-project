from src.api._event_forwarding import (
    build_alert_action_payload,
    build_alert_action_topic,
)
from src.api.schemas.event import AlertIn
from src.canonical_event import SKIP_ALERT_FORWARD_METADATA_KEY
from src.event_routing import (
    ALERT_STORAGE_OWNER_METADATA_KEY,
    PUBLIC_API_ALERT_STORAGE_OWNER,
)


def test_build_alert_action_payload_keeps_optional_fields():
    alert = AlertIn(
        camera_id="cam-01",
        event_type="helmet",
        severity="normal",
        confidence=0.95,
        timestamp=1700000000.0,
        bbox={"x": 10, "y": 20, "width": 100, "height": 80},
        object_id=7,
        metadata={"zone_id": "zone-A"},
    )

    assert build_alert_action_payload(alert) == {
        "camera_id": "cam-01",
        "type": "helmet",
        "severity": "normal",
        "confidence": 0.95,
        "timestamp": 1700000000.0,
        "bbox": {"x": 10, "y": 20, "width": 100, "height": 80},
        "object_id": 7,
        "metadata": {
            "zone_id": "zone-A",
            SKIP_ALERT_FORWARD_METADATA_KEY: True,
            ALERT_STORAGE_OWNER_METADATA_KEY: PUBLIC_API_ALERT_STORAGE_OWNER,
        },
    }


def test_build_alert_action_topic_matches_alarm_topic_pattern():
    alert = AlertIn(
        camera_id="camera_1",
        event_type="head",
        severity="critical",
        confidence=0.92,
        timestamp=1700000000.0,
    )

    assert build_alert_action_topic(alert) == "cctv/ai/events/camera_1/head"


def test_build_alert_action_topic_maps_rule_events():
    intrusion = AlertIn(
        camera_id="camera_1",
        event_type="intrusion",
        severity="critical",
        confidence=0.92,
        timestamp=1700000000.0,
    )
    temperature = AlertIn(
        camera_id="camera_1",
        event_type="sensor_temperature",
        severity="critical",
        confidence=0.92,
        timestamp=1700000000.0,
    )

    assert build_alert_action_topic(intrusion) == "cctv/rules/intrusion/critical"
    assert build_alert_action_topic(temperature) == "aiot/rules/sensor/temperature"
