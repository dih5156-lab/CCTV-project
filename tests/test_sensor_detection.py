import pytest

from src.core.sensor_detection import SensorEventDetector, SensorRuleConfig
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


# ──────────────────────────────────────────────
# 기울기 — 위험 케이스 (critical: ≥ 45°)
# ──────────────────────────────────────────────

def test_detect_tilt_alert_for_realistic_angles():
    detector = SensorEventDetector()

    events = detector.detect_events(_reading(angle_x=88.3, angle_y=2.1, temperature=27.8))

    assert len(events) == 1
    assert events[0].event_type == "tilt_alert"
    assert events[0].severity == "critical"


def test_tilt_critical_x_only():
    """X축 단독 45° 이상 → critical."""
    events = SensorEventDetector().detect_events(_reading(angle_x=45.0, angle_y=0.0))
    assert len(events) == 1
    assert events[0].severity == "critical"


def test_tilt_critical_y_only():
    """Y축 단독 60° → critical."""
    events = SensorEventDetector().detect_events(_reading(angle_x=0.0, angle_y=60.5))
    assert len(events) == 1
    assert events[0].severity == "critical"


def test_tilt_critical_both_axes():
    """X·Y 모두 임계 초과 → 이벤트 1개, critical."""
    events = SensorEventDetector().detect_events(_reading(angle_x=50.0, angle_y=55.0))
    assert len(events) == 1
    assert events[0].severity == "critical"


def test_tilt_critical_negative_angle():
    """음수 방향 기울기도 절대값 기준 판정."""
    events = SensorEventDetector().detect_events(_reading(angle_x=-89.0, angle_y=0.0))
    assert len(events) == 1
    assert events[0].severity == "critical"


# ──────────────────────────────────────────────
# 기울기 — 경고 케이스 (warning: 30° ≤ peak < 45°)
# ──────────────────────────────────────────────

def test_tilt_warning_boundary_30():
    """경계값 30.0° → warning."""
    events = SensorEventDetector().detect_events(_reading(angle_x=30.0, angle_y=0.0))
    assert len(events) == 1
    assert events[0].severity == "warning"


def test_tilt_warning_just_below_critical():
    """44.9° → warning (critical 미만)."""
    events = SensorEventDetector().detect_events(_reading(angle_x=44.9, angle_y=0.0))
    assert len(events) == 1
    assert events[0].severity == "warning"


def test_tilt_warning_y_axis_35():
    """Y=35° → warning."""
    events = SensorEventDetector().detect_events(_reading(angle_x=1.0, angle_y=35.0))
    assert len(events) == 1
    assert events[0].severity == "warning"


# ──────────────────────────────────────────────
# 기울기 — 정상 케이스 (이벤트 없음)
# ──────────────────────────────────────────────

def test_tilt_normal_both_low():
    """X·Y 모두 작은 값 → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading(angle_x=1.2, angle_y=0.3))
    assert events == []


def test_tilt_normal_just_below_warning():
    """29.9° → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading(angle_x=29.9, angle_y=0.0))
    assert events == []


def test_tilt_normal_zero():
    """완전 수직(0°) → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading(angle_x=0.0, angle_y=0.0))
    assert events == []


def test_tilt_normal_small_negative():
    """-5° → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading(angle_x=-5.0, angle_y=-3.0))
    assert events == []


# ──────────────────────────────────────────────
# 기울기 — 오탐 필터 (implausible 값)
# ──────────────────────────────────────────────

def test_ignore_implausible_large_tilt_values():
    """정수 오버플로 수준의 값은 무시."""
    events = SensorEventDetector().detect_events(_reading(angle_x=1119133802, angle_y=3212339505))
    assert events == []


def test_ignore_tilt_beyond_180():
    """±180° 초과 값은 무시."""
    events = SensorEventDetector().detect_events(_reading(angle_x=181.0, angle_y=-200.0))
    assert events == []


def test_ignore_tilt_nan_string():
    """문자열 NaN → 무시."""
    events = SensorEventDetector().detect_events(_reading(angle_x="nan", angle_y=0.0))
    assert events == []


def test_ignore_tilt_none_value():
    """None → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading(angle_x=None, angle_y=None))
    assert events == []


def test_ignore_tilt_inf():
    """Inf → 무시."""
    events = SensorEventDetector().detect_events(_reading(angle_x=float("inf"), angle_y=0.0))
    assert events == []


# ──────────────────────────────────────────────
# 온도 — 위험 케이스 (critical: ≥ 70°C)
# ──────────────────────────────────────────────

def test_detect_temperature_alert():
    events = SensorEventDetector().detect_events(_reading(temperature=72.5))
    assert len(events) == 1
    assert events[0].event_type == "temperature_alert"
    assert events[0].severity == "critical"


def test_temperature_critical_boundary():
    """경계값 70.0°C → critical."""
    events = SensorEventDetector().detect_events(_reading(temperature=70.0))
    assert len(events) == 1
    assert events[0].severity == "critical"


def test_temperature_critical_extreme():
    """120°C 극단값 → critical."""
    events = SensorEventDetector().detect_events(_reading(temperature=120.0))
    assert len(events) == 1
    assert events[0].severity == "critical"


# ──────────────────────────────────────────────
# 온도 — 경고 케이스 (warning: 50° ≤ T < 70°)
# ──────────────────────────────────────────────

def test_temperature_warning_boundary():
    """경계값 50.0°C → warning."""
    events = SensorEventDetector().detect_events(_reading(temperature=50.0))
    assert len(events) == 1
    assert events[0].severity == "warning"


def test_temperature_warning_mid():
    """60°C → warning."""
    events = SensorEventDetector().detect_events(_reading(temperature=60.0))
    assert len(events) == 1
    assert events[0].severity == "warning"


def test_temperature_warning_just_below_critical():
    """69.9°C → warning."""
    events = SensorEventDetector().detect_events(_reading(temperature=69.9))
    assert len(events) == 1
    assert events[0].severity == "warning"


# ──────────────────────────────────────────────
# 온도 — 정상 케이스 (이벤트 없음)
# ──────────────────────────────────────────────

def test_temperature_normal_room():
    """실내 정상 온도 25°C → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading(temperature=25.0))
    assert events == []


def test_temperature_normal_just_below_warning():
    """49.9°C → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading(temperature=49.9))
    assert events == []


def test_temperature_normal_zero():
    """0°C → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading(temperature=0.0))
    assert events == []


def test_temperature_normal_negative():
    """겨울철 -10°C → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading(temperature=-10.0))
    assert events == []


# ──────────────────────────────────────────────
# 온도 — 오탐 필터
# ──────────────────────────────────────────────

def test_temperature_ignore_above_max():
    """150°C 초과 → 센서 오류로 무시."""
    events = SensorEventDetector().detect_events(_reading(temperature=151.0))
    assert events == []


def test_temperature_ignore_below_min():
    """-51°C → 센서 오류로 무시."""
    events = SensorEventDetector().detect_events(_reading(temperature=-51.0))
    assert events == []


def test_temperature_ignore_none():
    """온도 필드 없음 → 이벤트 없음."""
    events = SensorEventDetector().detect_events(_reading())
    assert events == []


def test_temperature_ignore_string():
    """문자열 → 무시."""
    events = SensorEventDetector().detect_events(_reading(temperature="hot"))
    assert events == []


# ──────────────────────────────────────────────
# 복합 케이스 (기울기 + 온도 동시 발생)
# ──────────────────────────────────────────────

def test_both_tilt_and_temperature_trigger():
    """기울기 critical + 온도 critical → 이벤트 2개."""
    events = SensorEventDetector().detect_events(
        _reading(angle_x=60.0, angle_y=0.0, temperature=80.0)
    )
    assert len(events) == 2
    types = {e.event_type for e in events}
    assert "tilt_alert" in types
    assert "temperature_alert" in types


def test_tilt_warning_temperature_critical():
    """기울기 warning + 온도 critical → 이벤트 2개."""
    events = SensorEventDetector().detect_events(
        _reading(angle_x=35.0, angle_y=0.0, temperature=75.0)
    )
    assert len(events) == 2
    severities = {e.event_type: e.severity for e in events}
    assert severities["tilt_alert"] == "warning"
    assert severities["temperature_alert"] == "critical"


def test_normal_tilt_with_warning_temperature():
    """정상 기울기 + 온도 warning → 이벤트 1개 (온도만)."""
    events = SensorEventDetector().detect_events(
        _reading(angle_x=5.0, angle_y=2.0, temperature=55.0)
    )
    assert len(events) == 1
    assert events[0].event_type == "temperature_alert"


def test_all_normal_no_events():
    """완전 정상 패킷 → 이벤트 없음."""
    events = SensorEventDetector().detect_events(
        _reading(angle_x=1.2, angle_y=0.3, temperature=27.8)
    )
    assert events == []


# ──────────────────────────────────────────────
# 커스텀 임계치 — 오탐 기준 조정
# ──────────────────────────────────────────────

def test_custom_tilt_threshold_stricter():
    """임계치를 20°로 낮추면 25°에서 warning 발생."""
    rules = SensorRuleConfig(tilt_warning_angle=20.0, tilt_critical_angle=40.0)
    events = SensorEventDetector(rules).detect_events(_reading(angle_x=25.0, angle_y=0.0))
    assert len(events) == 1
    assert events[0].severity == "warning"


def test_custom_tilt_threshold_looser():
    """임계치를 60°로 높이면 50°에서 이벤트 없음."""
    rules = SensorRuleConfig(tilt_warning_angle=60.0, tilt_critical_angle=80.0)
    events = SensorEventDetector(rules).detect_events(_reading(angle_x=50.0, angle_y=0.0))
    assert events == []


def test_custom_temperature_threshold_stricter():
    """온도 경고를 40°C로 낮추면 45°C에서 warning."""
    rules = SensorRuleConfig(temperature_warning=40.0, temperature_critical=60.0)
    events = SensorEventDetector(rules).detect_events(_reading(temperature=45.0))
    assert len(events) == 1
    assert events[0].severity == "warning"


def test_register_detector_adds_custom_sensor_rule():
    detector = SensorEventDetector()

    def _vibration_rule(reading: SensorReading):
        if reading.telemetry.get("vibration") != "high":
            return None
        return detector._build_alert_event(
            reading,
            event_type="vibration_alert",
            severity="warning",
            message="진동 이상 감지",
            telemetry={"vibration": "high"},
        )

    detector.register_detector(_vibration_rule)

    events = detector.detect_events(_reading(vibration="high"))

    assert len(events) == 1
    assert events[0].event_type == "vibration_alert"
    assert events[0].severity == "warning"
