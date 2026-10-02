from scripts.ops.render_fall_model_comparison_markdown import render_markdown


def test_render_markdown_contains_decision_metrics_and_dataset_gate() -> None:
    rendered = render_markdown(
        {
            "promote_candidate": False,
            "baseline": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
            "candidate": {"precision": 0.82, "recall": 0.8, "f1": 0.81},
            "deltas": {"precision": 0.02, "recall": 0.1, "f1": 0.06},
            "checks": {"dataset_compatible": False},
            "baseline_dataset_version": "split-a",
            "candidate_dataset_version": "split-b",
        }
    )

    assert "판정: `HOLD`" in rendered
    assert "| recall | 0.7000 | 0.8000 | +0.1000 |" in rendered
    assert "데이터셋 호환성: `False`" in rendered
    assert "split-b" in rendered
