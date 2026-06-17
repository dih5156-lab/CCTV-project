from src.canonical_event import (
    get_payload_camera_id,
    get_payload_event_type,
    get_payload_metadata,
    get_payload_occurred_at,
)


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
