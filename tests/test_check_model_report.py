import importlib.util
import sys
from pathlib import Path


def _load_script_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_model_report = _load_script_module("check_model_report", "scripts/health/check_model_report.py")


def test_evaluate_criteria_passes_when_all_thresholds_met():
    criteria = {
        "min_precision": 0.85,
        "min_recall": 0.9,
        "max_avg_latency_ms": 50,
    }
    values = {"precision": 0.9, "recall": 0.92, "avg_latency_ms": 42.0}

    checks = check_model_report.evaluate_criteria(criteria, values)

    assert all(check.passed for check in checks)


def test_evaluate_criteria_fails_low_recall_and_high_latency():
    criteria = {
        "min_precision": 0.85,
        "min_recall": 0.9,
        "max_avg_latency_ms": 50,
    }
    values = {"precision": 0.86, "recall": 0.7, "avg_latency_ms": 80.0}

    checks = check_model_report.evaluate_criteria(criteria, values)
    failed = {check.metric for check in checks if not check.passed}

    assert failed == {"recall", "avg_latency_ms"}


def test_get_report_values_reads_overall_and_latency():
    report = {
        "metrics": {"overall": {"precision": 0.91, "recall": 0.87}},
        "latency": {"avg_ms": 33.5},
    }

    assert check_model_report.get_report_values(report) == {
        "precision": 0.91,
        "recall": 0.87,
        "avg_latency_ms": 33.5,
    }


def test_iter_artifact_checks_reports_missing_and_present(tmp_path):
    present = tmp_path / "model.onnx"
    present.write_bytes(b"model")
    manifest = {
        "models": [
            {
                "name": "sample",
                "artifacts": {
                    "onnx": "model.onnx",
                    "engine": "missing.engine",
                },
            }
        ]
    }

    checks = check_model_report.iter_artifact_checks(manifest, tmp_path)
    payload = check_model_report.build_artifact_payload(checks)

    assert payload["passed"] is False
    assert payload["artifact_count"] == 2
    assert payload["missing_count"] == 1
    assert {item["path"]: item["exists"] for item in payload["artifacts"]} == {
        "model.onnx": True,
        "missing.engine": False,
    }


def test_insightface_report_requires_model_id_and_measured_samples():
    errors = check_model_report.check_insightface_tensorrt_report(
        {"model_id": "wrong", "gallery_images": 0}
    )

    assert "unexpected InsightFace model_id: wrong" in errors
    assert "InsightFace gallery_images must be at least 2" in errors


def test_insightface_report_accepts_complete_poc_result():
    errors = check_model_report.check_insightface_tensorrt_report(
        {
            "model_id": "arcface-w600k-r50-tensorrt-v1",
            "gallery_images": 4,
            "identities": 2,
            "genuine_pairs": 2,
            "impostor_pairs": 4,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "p95_latency_ms": 40.0,
        }
    )

    assert errors == []
