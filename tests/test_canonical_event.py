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
