from src.event_priority import event_priority, event_risk_level


def test_event_priority_orders_fall_before_normal_events():
    assert event_priority({"type": "fall_detected", "severity": "critical"}) == 0
    assert event_risk_level({"type": "fall_detected", "severity": "critical"}) == "critical"


def test_event_priority_uses_canonical_event_shape():
    payload = {
        "event": {
            "event_type": "temperature_alert",
            "severity": "warning",
        }
    }

    assert event_priority(payload) == 4
    assert event_risk_level(payload) == "warning"


def test_event_priority_keeps_low_value_events_low():
    payload = {"event_type": "helmet", "severity": "low"}

    assert event_priority(payload) == 30
    assert event_risk_level(payload) == "low"
