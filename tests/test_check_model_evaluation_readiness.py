from pathlib import Path

from scripts.ops.check_model_evaluation_readiness import check_readiness


def test_exploratory_model_is_held_even_with_valid_metrics(tmp_path: Path) -> None:
    model = tmp_path / "fall_temporal_partial140.pt"
    metrics = tmp_path / "fall_temporal_partial140_metrics.json"
    model.write_bytes(b"checkpoint")
    metrics.write_text('{"validation": {}}', encoding="utf-8")

    result = check_readiness(model, metrics)

    assert result["ready"] is False
    assert result["decision"] == "hold"
    assert "exploratory" in result["reasons"][0]


def test_final_model_and_object_metrics_are_ready(tmp_path: Path) -> None:
    model = tmp_path / "fall_temporal_final.pt"
    metrics = tmp_path / "fall_temporal_final_metrics.json"
    model.write_bytes(b"checkpoint")
    metrics.write_text('{"validation": {}}', encoding="utf-8")

    result = check_readiness(model, metrics)

    assert result["ready"] is True
    assert result["decision"] == "ready"
    assert result["metrics_keys"] == ["validation"]


def test_invalid_or_missing_metrics_is_held(tmp_path: Path) -> None:
    model = tmp_path / "fall_temporal_final.pt"
    metrics = tmp_path / "fall_temporal_final_metrics.json"
    model.write_bytes(b"checkpoint")
    metrics.write_text("not-json", encoding="utf-8")

    result = check_readiness(model, metrics)

    assert result["ready"] is False
    assert "metrics file is not valid JSON" in result["reasons"]


def test_metrics_without_evaluation_split_is_held(tmp_path: Path) -> None:
    model = tmp_path / "fall_temporal_final.pt"
    metrics = tmp_path / "fall_temporal_final_metrics.json"
    model.write_bytes(b"checkpoint")
    metrics.write_text('{"training": {}}', encoding="utf-8")

    result = check_readiness(model, metrics)

    assert result["ready"] is False
    assert "validation or holdout" in result["reasons"][-1]


def test_model_and_metrics_must_use_matching_stems(tmp_path: Path) -> None:
    model = tmp_path / "fall_temporal_final.pt"
    metrics = tmp_path / "different_experiment_metrics.json"
    model.write_bytes(b"checkpoint")
    metrics.write_text('{"validation": {}}', encoding="utf-8")

    result = check_readiness(model, metrics)

    assert result["ready"] is False
    assert "does not match model" in result["reasons"][0]
