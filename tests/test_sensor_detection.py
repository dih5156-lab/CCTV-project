import pytest

from src.core.sensor_detection import SensorEventDetector
from src.devices.sensor_device import SensorReading


def _reading(**telemetry) -> SensorReading:
    return SensorReading(
        device_id="sensor-1",
        app_eui="a000000000000001",
        dev_eui="0080e11505c9e23c",
        table_name="t34957",
        telemetry=telemetry,
        received_at=1774938420097.0,
    )


def test_detect_tilt_alert_for_realistic_angles():
    detector = SensorEventDetector()

    events = detector.detect_events(_reading(angle_x=88.3, angle_y=2.1, temperature=27.8))

    assert len(events) == 1
    assert events[0].event_type == "tilt_alert"
    assert events[0].severity == "critical"


def test_ignore_implausible_large_tilt_values():
    detector = SensorEventDetector()

    events = detector.detect_events(_reading(angle_x=1119133802, angle_y=3212339505))

    assert events == []


def test_detect_temperature_alert():
    detector = SensorEventDetector()

    events = detector.detect_events(_reading(temperature=72.5))

    assert len(events) == 1
    assert events[0].event_type == "temperature_alert"
    assert events[0].severity == "critical"
