"""falldata RF metrics promotion checker tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "health"
    / "check_falldata_model_report.py"
)

spec = importlib.util.spec_from_file_location("check_falldata_model_report", SCRIPT_PATH)
check_falldata_model_report = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["check_falldata_model_report"] = check_falldata_model_report
spec.loader.exec_module(check_falldata_model_report)


def _report() -> dict:
    return {
        "dataset_version": "sample",
        "manifest": "data/fall_eval/sample_manifest.jsonl",
        "feature_cache": "data/fall_eval/falldata_feature_cache",
        "output_model": "models/experiments/model.pkl",
        "rows": 32,
        "class_counts": {"0": 24, "1": 8},
        "dataset_summary": {
            "group_class_counts": {"fall": 3, "non_fall": 2},
        },
        "holdout_split": {
            "method": "group_shuffle",
            "group_by": "scene_base",
            "train_groups": ["fall_a", "fall_b"],
            "test_groups": ["not_fall", "fall_c"],
        },
        "holdout_errors": {
            "false_negative_count": 0,
            "false_positive_count": 1,
        },
        "cross_validation": {"enabled": True},
    }


def test_evaluate_report_passes_standard_falldata_metrics() -> None:
    checks = check_falldata_model_report.evaluate_report(
        _report(),
        required_group_by="scene_base",
        min_class_groups=2,
        max_false_negatives=0,
        max_false_positives=1,
        require_cross_validation=True,
    )

    assert all(check.passed for check in checks)


def test_evaluate_report_fails_group_overlap_and_false_negative() -> None:
    report = _report()
    report["holdout_split"]["test_groups"] = ["fall_a"]
    report["holdout_errors"]["false_negative_count"] = 1

    checks = check_falldata_model_report.evaluate_report(
        report,
        required_group_by="scene_base",
        min_class_groups=2,
        max_false_negatives=0,
        max_false_positives=None,
        require_cross_validation=False,
    )
    failed = {check.name for check in checks if not check.passed}

    assert failed == {
        "holdout_split.group_overlap",
        "holdout_errors.false_negative_count",
    }


def test_evaluate_report_fails_when_class_group_count_is_too_small() -> None:
    report = _report()
    report["dataset_summary"]["group_class_counts"]["non_fall"] = 1

    checks = check_falldata_model_report.evaluate_report(
        report,
        required_group_by="scene_base",
        min_class_groups=2,
        max_false_negatives=0,
        max_false_positives=None,
        require_cross_validation=False,
    )
    failed = {check.name for check in checks if not check.passed}

    assert failed == {"dataset_summary.group_class_counts.non_fall"}


def test_build_payload_summarizes_report_identity(tmp_path) -> None:
    checks = check_falldata_model_report.evaluate_report(
        _report(),
        required_group_by="scene_base",
        min_class_groups=2,
        max_false_negatives=0,
        max_false_positives=None,
        require_cross_validation=False,
    )

    payload = check_falldata_model_report.build_payload(
        tmp_path / "metrics.json",
        _report(),
        checks,
    )

    assert payload["passed"] is True
    assert payload["dataset_version"] == "sample"
    assert payload["rows"] == 32


def test_build_manifest_entry_records_falldata_evaluation() -> None:
    checks = check_falldata_model_report.evaluate_report(
        _report(),
        required_group_by="scene_base",
        min_class_groups=2,
        max_false_negatives=0,
        max_false_positives=None,
        require_cross_validation=False,
    )

    entry = check_falldata_model_report.build_manifest_entry(
        model_name="falldata_sample_rf",
        metrics_path=Path("models/experiments/metrics.json"),
        report=_report(),
        checks=checks,
    )

    assert entry["name"] == "falldata_sample_rf"
    assert entry["task"] == "falldata_video_fall_verifier"
    assert entry["artifacts"]["pickle"] == "models/experiments/model.pkl"
    assert entry["latest_evaluation"]["false_negative_count"] == 0


def test_update_manifest_replaces_existing_model_entry(tmp_path) -> None:
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(
        """{
  "schema_version": 1,
  "models": [
    {
      "name": "falldata_sample_rf",
      "task": "old"
    }
  ]
}
""",
        encoding="utf-8",
    )
    checks = check_falldata_model_report.evaluate_report(
        _report(),
        required_group_by="scene_base",
        min_class_groups=2,
        max_false_negatives=0,
        max_false_positives=None,
        require_cross_validation=False,
    )

    check_falldata_model_report.update_manifest(
        manifest_path,
        model_name="falldata_sample_rf",
        metrics_path=Path("models/experiments/metrics.json"),
        report=_report(),
        checks=checks,
    )

    updated = check_falldata_model_report.load_json(manifest_path)
    assert len(updated["models"]) == 1
    assert updated["models"][0]["task"] == "falldata_video_fall_verifier"
    assert updated["models"][0]["artifacts"]["metrics"] == "models/experiments/metrics.json"
