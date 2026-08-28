"""AIoT 장치 등록 입력과 EdgeX payload 계약을 검증한다."""

import argparse
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "edgex" / "register_aiot_devices.py"
SPEC = importlib.util.spec_from_file_location("register_aiot_devices", MODULE_PATH)
register_aiot_devices = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(register_aiot_devices)


def test_parse_device_spec_accepts_real_device_id() -> None:
    assert register_aiot_devices.parse_device_spec(
        "SNIOT-F-RVM-001:t34950"
    ) == ("SNIOT-F-RVM-001", "t34950")


@pytest.mark.parametrize("value", ["missing-table", ":t34950", "device:"])
def test_parse_device_spec_rejects_incomplete_value(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        register_aiot_devices.parse_device_spec(value)


def test_build_device_payload_uses_profile_for_selected_table() -> None:
    payload = register_aiot_devices.build_device_payload(
        "SNIOT-F-RVM-001", "t34950"
    )

    device = payload[0]["device"]
    assert device["name"] == "aiot-SNIOT-F-RVM-001"
    assert device["profileName"] == "aiot-t34950-river"
    assert device["serviceName"] == "device-rest"
    assert device["protocols"]["lora"] == {
        "device_id": "SNIOT-F-RVM-001",
        "primary_table": "t34950",
    }
