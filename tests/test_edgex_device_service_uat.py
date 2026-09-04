from scripts.ops.run_edgex_device_service_uat import build_cases


def test_uat_cases_cover_all_output_devices_and_safe_actions():
    cases = build_cases()

    assert [case["device"] for case in cases] == ["speaker", "siren", "signboard"]
    assert [case["path"] for case in cases] == [
        "/api/v3/device/name/cctv-speaker-01/play",
        "/api/v3/device/name/cctv-siren-01/trigger",
        "/api/v3/device/name/cctv-signboard-01/display",
    ]
    assert all(case["payload"]["event_id"].startswith("uat-") for case in cases)
