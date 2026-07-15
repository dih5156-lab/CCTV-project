"""Operational fall threshold tuning tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ops"
    / "tune_fall_operational_thresholds.py"
)

spec = importlib.util.spec_from_file_location("tune_fall_operational_thresholds", SCRIPT_PATH)
tune_fall_operational_thresholds = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["tune_fall_operational_thresholds"] = tune_fall_operational_thresholds
spec.loader.exec_module(tune_fall_operational_thresholds)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_threshold_metrics_count_false_positive_and_false_negative(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {"scene_id": "fall_hit", "expected_fall": True, "detected": True, "max_fall_score": 5.0},
            {"scene_id": "fall_miss", "expected_fall": True, "detected": False, "max_fall_score": 1.0},
            {"scene_id": "safe_hit", "expected_fall": False, "detected": False, "max_fall_score": 0.5},
            {"scene_id": "safe_fp", "expected_fall": False, "detected": True, "max_fall_score": 4.0},
        ],
    )

    rows, errors = tune_fall_operational_thresholds.load_result_rows(
        [results_path],
        "max_fall_score",
    )
    payload = tune_fall_operational_thresholds.build_payload(
        [results_path],
        rows,
        errors,
        score_field="max_fall_score",
        thresholds=[2.0],
        max_false_positives=1,
    )

    assert payload["parse_errors"] == []
    assert payload["baseline_detected_metrics"]["tp"] == 1
    assert payload["baseline_detected_metrics"]["fp"] == 1
    assert payload["thresholds"][0]["tp"] == 1
    assert payload["thresholds"][0]["fp"] == 1
    assert payload["thresholds"][0]["fn"] == 1
    assert payload["thresholds"][0]["tn"] == 1


def test_recommends_lowest_fn_under_false_positive_limit(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {"scene_id": "fall_high", "expected_fall": True, "detected": False, "max_fall_score": 5.0},
            {"scene_id": "fall_mid", "expected_fall": True, "detected": False, "max_fall_score": 3.0},
            {"scene_id": "safe_low", "expected_fall": False, "detected": False, "max_fall_score": 1.0},
            {"scene_id": "safe_mid", "expected_fall": False, "detected": False, "max_fall_score": 2.5},
        ],
    )

    rows, errors = tune_fall_operational_thresholds.load_result_rows(
        [results_path],
        "max_fall_score",
    )
    payload = tune_fall_operational_thresholds.build_payload(
        [results_path],
        rows,
        errors,
        score_field="max_fall_score",
        thresholds=[2.0, 3.0, 4.0],
        max_false_positives=0,
    )

    assert payload["recommended_threshold"]["threshold"] == 3.0
    assert payload["recommended_threshold"]["fp"] == 0
    assert payload["recommended_threshold"]["fn"] == 0
