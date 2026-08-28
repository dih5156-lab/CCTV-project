"""EdgeX 프로파일이 장치의 실제 통신 방향을 올바르게 노출하는지 검증한다."""

from pathlib import Path

import yaml

PROFILES_DIR = Path(__file__).resolve().parents[1] / "edgex" / "device-profiles"


def test_uplink_sensor_profiles_do_not_expose_polling_commands() -> None:
    for profile_path in sorted(PROFILES_DIR.glob("aiot-*-profile.yaml")):
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

        assert profile["deviceResources"]
        assert all(
            resource.get("isHidden") is True
            for resource in profile["deviceResources"]
        ), profile_path.name
        assert all(
            command.get("isHidden") is True
            for command in profile.get("deviceCommands", [])
        ), profile_path.name


def test_signboard_commands_remain_visible() -> None:
    profile_path = PROFILES_DIR / "cctv-signboard-dabit-profile.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    assert {command["name"] for command in profile["deviceCommands"]} == {
        "display",
        "clear",
        "power",
    }
    assert all(
        command.get("isHidden", False) is False
        for command in profile["deviceCommands"]
    )
