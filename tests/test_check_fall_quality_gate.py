import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/health/check_fall_quality_gate.py"
    spec = importlib.util.spec_from_file_location("check_fall_quality_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluate_calculates_release_metrics(tmp_path):
    module = _load_module()
    fall = tmp_path / "fall.jsonl"
    nonfall = tmp_path / "nonfall.jsonl"
    fall.write_text(json.dumps({"expected_fall": True, "detected": True}) + "\n" + json.dumps({"expected_fall": True, "detected": False}) + "\n")
    nonfall.write_text(json.dumps({"expected_fall": False, "detected": False}) + "\n")

    result = module.evaluate(fall, nonfall, 0.5, 0.9)

    assert result["counts"] == {"tp": 1, "fn": 1, "fp": 0, "tn": 1, "total": 3}
    assert result["passed"] is True


def test_evaluate_fails_recall_gate(tmp_path):
    module = _load_module()
    fall = tmp_path / "fall.jsonl"
    nonfall = tmp_path / "nonfall.jsonl"
    fall.write_text(json.dumps({"expected_fall": True, "detected": False}) + "\n")
    nonfall.write_text(json.dumps({"expected_fall": False, "detected": False}) + "\n")

    result = module.evaluate(fall, nonfall, 0.8, 0.7)

    assert result["passed"] is False
