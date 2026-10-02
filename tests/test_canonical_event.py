from copy import deepcopy

import pytest

from src.canonical_event import (
    get_payload_camera_id,
    get_payload_event_type,
    get_payload_metadata,
    get_payload_occurred_at,
)


def test_edgex_projection_excludes_raw_and_keeps_media_reference():
    from src.canonical_event import project_edgex_event

    projected = project_edgex_event(
        {
            "schema_version": "1.0",
            "event_id": "event-1",
            "occurred_at": "2026-07-16T00:00:00Z",
            "device": {"camera_id": "camera-1", "device_type": "cctv"},
            "event": {
                "event_type": "fall_detected",
                "confidence": 0.94,
                "severity": "critical",
            },
            "media": {"snapshot_url": "/crops/event-1.jpg"},
            "raw": {"keypoints": [1, 2, 3]},
        }
    )
    assert "raw" not in projected
    assert projected["snapshot_url"] == "/crops/event-1.jpg"
    assert projected["resource"] == "fall_detection"


def test_payload_accessors_accept_legacy_and_canonical_shapes():
    payload = {
        "cameraId": "cam-legacy",
        "event": {
            "event_type": "fall_detected",
            "severity": "critical",
        },
        "raw": {
            "metadata": {"source": "deepstream"},
        },
        "queued_at": 1700000000.0,
    }

    assert get_payload_camera_id(payload) == "cam-legacy"
    assert get_payload_event_type(payload) == "fall_detected"
    assert get_payload_metadata(payload) == {"source": "deepstream"}
    assert get_payload_occurred_at(payload) == "2023-11-15T07:13:20+09:00"


def test_payload_camera_id_prefers_device_mapping():
    payload = {
        "camera_id": "top-level-camera",
        "device": {"deviceId": "device-camera"},
        "type": "helmet",
    }

    assert get_payload_camera_id(payload) == "device-camera"


def test_canonicalize_event_payload_preserves_supplied_event_id_for_legacy_shape():
    from src.canonical_event import canonicalize_event_payload

    normalized = canonicalize_event_payload(
        {
            "camera_id": "cam-01",
            "event_id": "uat-event-1",
            "type": "fall_detected",
            "severity": "critical",
        }
    )

    assert normalized["event_id"] == "uat-event-1"


@pytest.mark.parametrize(
    "missing_fields", [("schema_version",), ("event",), ("schema_version", "event")]
)
def test_canonicalize_partial_payload_preserves_producer_fields(missing_fields):
    from src.canonical_event import canonicalize_event_payload

    payload = {
        "schema_version": "1.0",
        "message_type": "sensor_event",
        "message_id": "packet-17",
        "event_id": "sensor-event-17",
        "occurred_at": "2026-09-18T10:00:00+09:00",
        "device": {"device_id": "sensor-1", "site_id": "site-1"},
        "gateway": {"gateway_id": "gateway-1"},
        "type": "temperature_high",
        "event": {
            "event_type": "temperature_high",
            "source": "tlv-parser",
            "source_type": "sensor",
            "rule_id": "rule-1",
        },
        "decoded": {"temperature": 32.5},
        "raw": {"payload_hex": "example-only", "metadata": {"packet_sequence": 17}},
        "metadata": {"legacy_source": "uplink"},
        "object_id": 0,
    }
    for field in missing_fields:
        payload.pop(field)
    original = deepcopy(payload)

    normalized = canonicalize_event_payload(
        payload, source="tlv-parser", source_type="sensor"
    )

    assert normalized["decoded"] == original["decoded"]
    for field in ("device", "gateway", "raw", "event"):
        for key, value in original.get(field, {}).items():
            assert normalized[field][key] == value
    assert normalized["message_type"] == "sensor_event"
    assert normalized["event_id"] == "sensor-event-17"
    assert normalized["message_id"] == "packet-17"
    assert normalized["raw"]["object_id"] == 0
    assert normalized["camera_id"] == "sensor-1"
    assert normalized["event"]["event_type"] == "temperature_high"
    assert canonicalize_event_payload(normalized) == normalized
    assert payload == original
