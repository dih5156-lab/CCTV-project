import json
from pathlib import Path

import pytest

from src.devices.sensor_device import SensorReading


def test_from_decoded_prefers_device_id_over_dev_eui():
    reading = SensorReading.from_decoded(
        {
            "device_id": "factory-24",
            "dev_eui": "0080E11505C9E23C",
            "timestamp": 1710000000,
        },
        {"tableName": "t34957", "data": {"temperature": 72.5}},
    )

    assert reading.device_id == "factory-24"
    assert reading.dev_eui == "0080e11505c9e23c"


def test_from_decoded_normalizes_millisecond_timestamp():
    reading = SensorReading.from_decoded(
        {"dev_eui": "dev-1", "timestamp": 1774938420097},
        {"tableName": "t34957", "data": {"temperature": 25.0}},
    )

    assert reading.received_at == 1774938420.097


def test_from_decoded_normalizes_iso_timestamp():
    reading = SensorReading.from_decoded(
        {"dev_eui": "dev-1", "timestamp": "2026-05-27T01:00:00+00:00"},
        {"tableName": "t34957", "data": {"temperature": 25.0}},
    )

    assert reading.received_at > 1_700_000_000


def test_from_decoded_ignores_non_mapping_data():
    reading = SensorReading.from_decoded(
        {"dev_eui": "dev-1", "timestamp": 1710000000},
        {"tableName": "t34957", "data": None},
    )

    assert reading.table_name == "t34957"
    assert reading.telemetry == {}


SENSOR_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "sensor_payloads.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case_name", SENSOR_FIXTURES.keys())
def test_from_decoded_accepts_standard_sensor_fixtures(case_name):
    from src.services.sensor_rule_bridge import build_sensor_bridge_inputs

    case = SENSOR_FIXTURES[case_name]
    uplink, decoded = build_sensor_bridge_inputs(case["message"])
    reading = SensorReading.from_decoded(uplink, decoded)

    assert reading.device_id == case["expected"]["device_id"]
    assert reading.table_name == case["expected"]["table"]
    assert isinstance(reading.telemetry, dict)
    assert reading.source == "lora_tlv"
