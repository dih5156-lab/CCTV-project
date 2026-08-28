"""EdgeX 장치 계약 감사 로직을 검증한다."""

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "health"
    / "check_edgex_device_contracts.py"
)
SPEC = importlib.util.spec_from_file_location("check_edgex_device_contracts", MODULE_PATH)
check_edgex_device_contracts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check_edgex_device_contracts)


def test_audit_reports_missing_device_and_unknown_resource() -> None:
    result = check_edgex_device_contracts.audit_device_contracts(
        devices=[],
        profiles=[{
            "name": "aiot-t34950-river",
            "deviceResources": [{"name": "water_level", "isHidden": True}],
            "deviceCommands": [{"name": "river_reading", "isHidden": True}],
        }],
        events=[{
            "deviceName": "aiot-SNIOT-F-RVM-001",
            "profileName": "aiot-t34950-river",
            "readings": [{"resourceName": "water_level_m"}],
        }],
    )

    assert {issue["code"] for issue in result["issues"]} == {
        "missing_metadata_device",
        "unknown_profile_resource",
    }
    assert result["ok"] is False


def test_audit_accepts_registered_uplink_event_contract() -> None:
    result = check_edgex_device_contracts.audit_device_contracts(
        devices=[{
            "name": "aiot-SNIOT-F-RVM-001",
            "profileName": "aiot-t34950-river",
        }],
        profiles=[{
            "name": "aiot-t34950-river",
            "deviceResources": [{"name": "water_level", "isHidden": True}],
            "deviceCommands": [{"name": "river_reading", "isHidden": True}],
        }],
        events=[{
            "deviceName": "aiot-SNIOT-F-RVM-001",
            "profileName": "aiot-t34950-river",
            "readings": [{"resourceName": "water_level"}],
        }],
    )

    assert result == {"ok": True, "issues": []}


def test_audit_reports_visible_polling_command_for_uplink_profile() -> None:
    result = check_edgex_device_contracts.audit_device_contracts(
        devices=[],
        profiles=[{
            "name": "aiot-t34957-tilt-temp",
            "deviceResources": [{"name": "temperature", "isHidden": False}],
            "deviceCommands": [{"name": "tilt_temp_reading", "isHidden": False}],
        }],
        events=[],
    )

    codes = [issue["code"] for issue in result["issues"]]
    assert codes == ["uplink_command_exposed"]


def test_audit_uses_latest_event_per_device_profile() -> None:
    result = check_edgex_device_contracts.audit_device_contracts(
        devices=[{
            "name": "aiot-SNIOT-F-RVM-001",
            "profileName": "aiot-t34950-river",
        }],
        profiles=[{
            "name": "aiot-t34950-river",
            "deviceResources": [{"name": "water_level", "isHidden": True}],
            "deviceCommands": [{"name": "river_reading", "isHidden": True}],
        }],
        events=[
            {
                "origin": 1,
                "deviceName": "aiot-SNIOT-F-RVM-001",
                "profileName": "aiot-t34950-river",
                "sourceName": "t34950",
                "readings": [{"resourceName": "water_level_m"}],
            },
            {
                "origin": 2,
                "deviceName": "aiot-SNIOT-F-RVM-001",
                "profileName": "aiot-t34950-river",
                "sourceName": "t34950",
                "readings": [{"resourceName": "water_level"}],
            },
        ],
    )

    assert result == {"ok": True, "issues": []}


def test_check_live_contracts_fetches_snapshots(monkeypatch) -> None:
    responses = {
        "http://metadata/api/v3/device/all?limit=1000": {"devices": []},
        "http://metadata/api/v3/deviceprofile/all?limit=1000": {
            "profiles": []
        },
        "http://core-data/api/v3/event/all?limit=25": {"events": []},
    }
    monkeypatch.setattr(
        check_edgex_device_contracts,
        "_get_json",
        lambda url: responses[url],
    )

    result = check_edgex_device_contracts.check_live_contracts(
        metadata_url="http://metadata/",
        core_data_url="http://core-data/",
        event_limit=25,
    )

    assert result == {
        "ok": True,
        "issues": [],
        "checked": {"devices": 0, "profiles": 0, "events": 0},
    }
