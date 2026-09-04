from src.event_priority import event_priority, event_risk_level, event_risk_score


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


def test_event_priority_reads_routing_priority_from_event_type_map():
    assert event_priority({"type": "fall_detected"}) == 0
    assert event_priority({"type": "head"}) == 2
    assert event_priority({"type": "helmet", "severity": "low"}) == 30


def test_event_risk_score_rewards_critical_high_confidence_events():
    payload = {
        "event_type": "fall_detected",
        "severity": "critical",
        "confidence": 0.95,
        "bbox": {"x": 100, "y": 100, "width": 300, "height": 500},
        "metadata": {"frame_width": 1920, "frame_height": 1080},
    }

    assert event_risk_score(payload) >= 90


def test_event_risk_score_penalizes_small_edge_bbox_and_review():
    payload = {
        "event_type": "head",
        "severity": "warning",
        "confidence": 0.5,
        "bbox": {"x": 0, "y": 0, "width": 20, "height": 20},
        "metadata": {"frame_width": 1920, "frame_height": 1080},
    }

    plain = event_risk_score(payload)
    reviewed = event_risk_score(payload, review_status="false_positive")

    assert plain < 60
    assert reviewed < plain
