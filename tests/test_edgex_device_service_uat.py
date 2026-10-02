import pytest

from scripts.ops.run_edgex_device_service_uat import build_cases, run_uat


def test_uat_cases_cover_all_output_devices_and_safe_actions():
    cases = build_cases()

    assert [case["device"] for case in cases] == ["speaker", "siren", "signboard"]
    assert [case["path"] for case in cases] == [
        "/api/v3/device/name/cctv-speaker-01/play",
        "/api/v3/device/name/cctv-siren-01/trigger",
        "/api/v3/device/name/cctv-signboard-01/display",
    ]
    assert all(case["payload"]["event_id"].startswith("uat-") for case in cases)


def test_uat_can_select_only_signboard():
    cases = build_cases("signboard")

    assert [case["device"] for case in cases] == ["signboard"]
    assert cases[0]["path"].endswith("/display")


def test_uat_rejects_unknown_device():
    with pytest.raises(ValueError, match="지원하지 않는 UAT 장치"):
        build_cases("unknown")


def test_dry_run_checks_health_without_sending_command(monkeypatch):
    requests = []

    def fake_request_json(url, method="GET", payload=None):
        requests.append((url, method, payload))
        return 200, {"status": "up"}

    monkeypatch.setattr(
        "scripts.ops.run_edgex_device_service_uat.request_json",
        fake_request_json,
    )

    results = run_uat("dry-run", device="speaker")

    assert requests == [("http://127.0.0.1:59991/health", "GET", None)]
    assert results == [
        {
            "device": "speaker",
            "health_code": 200,
            "health_status": "up",
            "command_code": None,
            "command_status": "not_sent",
            "expected_status": "not_sent",
            "passed": True,
        }
    ]
