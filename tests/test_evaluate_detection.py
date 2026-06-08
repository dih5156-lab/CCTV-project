import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate_detection = _load_script_module("evaluate_detection", "scripts/ops/evaluate_detection.py")


def test_yolo_to_xyxy_converts_normalized_box():
    box = evaluate_detection.yolo_to_xyxy([0.5, 0.5, 0.5, 0.25], 200, 100)
    assert box == pytest.approx((50.0, 37.5, 150.0, 62.5))


def test_box_iou_returns_expected_overlap():
    iou = evaluate_detection.box_iou((0, 0, 100, 100), (50, 50, 150, 150))
    assert iou == pytest.approx(2500 / 17500)


def test_match_detections_counts_tp_fp_fn():
    predictions = [
        evaluate_detection.Box("helmet", (0, 0, 100, 100), 0.9),
        evaluate_detection.Box("helmet", (200, 200, 260, 260), 0.8),
    ]
    ground_truth = [
        evaluate_detection.Box("helmet", (5, 5, 95, 95)),
        evaluate_detection.Box("helmet", (300, 300, 360, 360)),
    ]

    result = evaluate_detection.match_detections(predictions, ground_truth, iou_threshold=0.5)

    assert result["helmet"] == {"tp": 1, "fp": 1, "fn": 1}


def test_summarize_counts_calculates_overall_metrics():
    summary = evaluate_detection.summarize_counts(
        {
            "helmet": {"tp": 8, "fp": 2, "fn": 1},
            "head": {"tp": 1, "fp": 1, "fn": 3},
        }
    )

    assert summary["overall"]["precision"] == pytest.approx(0.75)
    assert summary["overall"]["recall"] == pytest.approx(0.6923)
    assert summary["by_class"]["helmet"]["recall"] == pytest.approx(0.8889)


def test_normalize_model_names_accepts_ultralytics_mapping():
    names = evaluate_detection.normalize_model_names({0: "helmet", "1": "head", "bad": "skip"})

    assert names == {0: "helmet", 1: "head"}
