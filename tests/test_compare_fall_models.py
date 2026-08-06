from scripts.compare_fall_models import compare_reports


def _report(precision: float, recall: float, f1: float) -> dict:
    return {"validation": {"classification_report": {"fall": {
        "precision": precision, "recall": recall, "f1-score": f1,
    }}}}


def test_compare_promotes_strictly_better_candidate():
    result = compare_reports(_report(0.9, 0.8, 0.85), _report(0.92, 0.81, 0.86))
    assert result["promote_candidate"] is True
    assert result["checks"]["strict_improvement"] is True


def test_compare_rejects_candidate_with_lower_recall():
    result = compare_reports(_report(0.9, 0.8, 0.85), _report(0.95, 0.7, 0.82))
    assert result["promote_candidate"] is False
