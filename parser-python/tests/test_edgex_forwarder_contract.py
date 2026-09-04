"""EdgeX 센서 이벤트가 Device Profile 계약을 따르는지 검증한다."""

import sys
from pathlib import Path

PARSER_ROOT = Path(__file__).resolve().parents[1]
if str(PARSER_ROOT) not in sys.path:
    sys.path.insert(0, str(PARSER_ROOT))

from mqtt.edgex_forwarder import EdgeXForwarder


def _reading_map(event_body: dict) -> dict[str, dict]:
    return {
        reading["resourceName"]: reading
        for reading in event_body["event"]["readings"]
    }


def test_river_event_uses_profile_resource_names() -> None:
    forwarder = object.__new__(EdgeXForwarder)

    event_body = forwarder._build_edgex_event(
        "SNIOT-F-RVM-001",
        "t34950",
        {
            "water_level_m": 0.334,
            "flow_velocity_mps": 0.12,
            "rain_fall_mm": 1.5,
            "reporting_period_s": 60,
        },
        1_787_891_904_466,
    )

    readings = _reading_map(event_body)
    assert set(readings) == {
        "water_level",
        "flow_velocity",
        "rain_fall",
        "reporting_period",
    }
    assert readings["water_level"]["value"] == "0.334"


def test_imu_event_omits_fields_not_declared_by_profile() -> None:
    forwarder = object.__new__(EdgeXForwarder)

    event_body = forwarder._build_edgex_event(
        "factory-16",
        "t34958",
        {
            "acc_x_g": 0.1,
            "gyro_z_dps": 3.2,
            "angle_x_deg": 12.0,
            "event_code": True,
        },
        1_787_891_904_466,
    )

    readings = _reading_map(event_body)
    assert set(readings) == {"acc_x", "gyro_z", "event_code"}
    assert readings["event_code"]["valueType"] == "Int32"
    assert readings["event_code"]["value"] == "1"
