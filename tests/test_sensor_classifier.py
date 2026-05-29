from src.services.sensor_classifier import classify_sensor_payload, extract_sensor_data


def test_extract_sensor_data_prefers_data_field():
    payload = {"data": {"temperature": 25.0}, "decoded": {"temperature": 80.0}}

    assert extract_sensor_data(payload) == {"temperature": 25.0}


def test_classify_sensor_payload_marks_high_temperature_critical():
    risk = classify_sensor_payload({"data": {"temperature": 72.5}})

    assert risk["status"] == "alert"
    assert risk["severity"] == "critical"
    assert risk["event_type"] == "temperature_alert"


def test_classify_sensor_payload_marks_tilt_warning():
    risk = classify_sensor_payload({"data": {"angle_x": 31.2}})

    assert risk["status"] == "alert"
    assert risk["severity"] == "warning"
    assert risk["event_type"] == "tilt_alert"


def test_classify_sensor_payload_keeps_explicit_event_type():
    risk = classify_sensor_payload({"type": "vibration_alert", "severity": "warn"})

    assert risk == {
        "status": "alert",
        "severity": "warning",
        "event_type": "vibration_alert",
        "reason": "vibration_alert",
    }
