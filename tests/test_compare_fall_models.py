from scripts.compare_fall_models import compare_reports


def _report(*, precision: float, recall: float, threshold: float) -> dict:
    return {
        "model_params": {"decision_threshold": threshold},
        "validation": {
            "classification_report": {
                "fall": {
                    "precision": precision,
                    "recall": recall,
                    "f1-score": 2 * precision * recall / (precision + recall),
                }
            }
        },
    }


def test_compare_reports_preserves_each_model_decision_threshold() -> None:
    result = compare_reports(
        _report(precision=0.99, recall=0.80, threshold=0.71),
        _report(precision=0.995, recall=0.81, threshold=0.78),
        min_precision=0.99,
    )

    assert result["baseline_decision_threshold"] == 0.71
    assert result["candidate_decision_threshold"] == 0.78
    assert result["decision_threshold"] == 0.78


def test_dataset_version_mismatch_blocks_promotion() -> None:
    baseline = _report(precision=0.80, recall=0.70, threshold=0.70)
    candidate = _report(precision=0.90, recall=0.80, threshold=0.70)
    baseline["dataset_version"] = "split-a"
    candidate["dataset_version"] = "split-b"

    result = compare_reports(baseline, candidate)

    assert result["dataset_compatible"] is False
    assert result["checks"]["dataset_compatible"] is False
    assert result["promote_candidate"] is False
