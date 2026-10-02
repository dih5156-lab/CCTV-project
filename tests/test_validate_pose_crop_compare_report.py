from scripts.ops.validate_pose_crop_compare_report import validate_report


def _valid_report() -> dict:
    return {
        "mode": "crop-pose",
        "model_paths": {},
        "runtime_config": {},
        "metrics": {},
        "video_cases": [],
        "decision": "hold",
        "operator": "tester",
        "measured_at": "2026-09-09T12:00:00+09:00",
    }


def test_validate_report_accepts_minimal_valid_contract():
    assert validate_report(_valid_report()) == []


def test_validate_report_reports_missing_and_invalid_fields():
    errors = validate_report({"mode": "unknown", "decision": "ship"})
    assert "missing field: metrics" in errors
    assert "mode must be one of: full-frame, crop-pose" in errors
    assert "decision must be one of: promote, rollback, hold" in errors


def test_validate_report_rejects_mode_runtime_mismatch():
    report = _valid_report()
    report["runtime_config"] = {"pose_crop_mode": 0}
    errors = validate_report(report)
    assert "mode crop-pose requires runtime_config.pose_crop_mode enabled" in errors
