from scripts.quality_gate_fall_replay import evaluate_results


def test_quality_gate_passes_clean_results():
    result = evaluate_results(
        [{"result": "TP"}, {"result": "TP"}, {"result": "TN"}],
        min_precision=0.9,
        min_recall=0.9,
    )
    assert result["passed"] is True


def test_quality_gate_rejects_false_positive_or_no_result():
    result = evaluate_results(
        [{"result": "TP"}, {"result": "FP"}, {"result": "NO_RESULT"}],
        min_precision=0.9,
        min_recall=0.5,
    )
    assert result["passed"] is False
