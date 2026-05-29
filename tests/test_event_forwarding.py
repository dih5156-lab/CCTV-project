from src.api._event_forwarding import build_alert_action_payload
from src.api.schemas.event import AlertIn


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
        "metadata": {"zone_id": "zone-A"},
    }
