import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ops"
    / "summarize_fall_shadow_review.py"
)

spec = importlib.util.spec_from_file_location("summarize_fall_shadow_review", SCRIPT_PATH)
summarize_fall_shadow_review = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["summarize_fall_shadow_review"] = summarize_fall_shadow_review
spec.loader.exec_module(summarize_fall_shadow_review)


def test_summarize_records_counts_pending_unconfirmed() -> None:
    rows = [
        {
            "event_id": "fall-1",
            "review_source": "falldata_aux",
            "falldata_aux_publish_pending": True,
            "falldata_aux": {
                "status": "ok",
                "confirmed": False,
                "fall_probability": 0.2,
            },
        },
        {
            "event_id": "fall-2",
            "review_source": "falldata_aux",
            "falldata_aux_publish_pending": True,
            "falldata_aux": {
                "status": "ok",
                "confirmed": True,
                "fall_probability": 0.91,
            },
        },
    ]

    summary = summarize_fall_shadow_review.summarize_records(rows)

    assert summary["total_records"] == 2
    assert summary["status_counts"] == {"ok": 2}
    assert summary["pending_unconfirmed_count"] == 1
    assert "review pending" in summary["recommendation"]


def test_summarize_records_flags_unavailable_aux() -> None:
    summary = summarize_fall_shadow_review.summarize_records(
        [
            {
                "event_id": "fall-1",
                "falldata_aux": {
                    "status": "missing_dependency",
                    "confirmed": False,
                },
            }
        ]
    )

    assert summary["runtime_failure_count"] == 1
    assert "fix aux runtime" in summary["recommendation"]


def test_summarize_records_counts_cooldown_separately() -> None:
    summary = summarize_fall_shadow_review.summarize_records(
        [
            {
                "event_id": "fall-1",
                "falldata_aux": {
                    "status": "skipped_cooldown",
                    "confirmed": False,
                },
            }
        ]
    )

    assert summary["cooldown_skip_count"] == 1
    assert summary["runtime_failure_count"] == 0


def test_summarize_records_counts_labels_and_labeling_candidates() -> None:
    summary = summarize_fall_shadow_review.summarize_records(
        [
            {
                "event_id": "fall-high",
                "clip_path": "/clips/high.mp4",
                "label": None,
                "review_status": "unlabeled",
                "falldata_aux": {
                    "status": "ok",
                    "confirmed": True,
                    "fall_probability": 0.92,
                },
            },
            {
                "event_id": "fall-low",
                "clip_path": "/clips/low.mp4",
                "label": None,
                "review_status": "unlabeled",
                "falldata_aux": {
                    "status": "ok",
                    "confirmed": False,
                    "fall_probability": 0.45,
                },
            },
            {
                "event_id": "fall-labeled",
                "clip_path": "/clips/labeled.mp4",
                "label": "non_fall",
                "review_status": "reviewed",
                "falldata_aux": {
                    "status": "ok",
                    "confirmed": False,
                    "fall_probability": 0.8,
                },
            },
        ]
    )

    assert summary["label_counts"] == {"non_fall": 1, "unlabeled": 2}
    assert summary["review_status_counts"] == {"reviewed": 1, "unlabeled": 2}
    assert summary["clip_counts"] == {"with_clip": 3}
    assert summary["labeling_candidate_count"] == 2
    assert summary["labeling_candidate_examples"][0]["event_id"] == "fall-high"
    assert summary["human_label_aux_evaluation"] == {
        "evaluated_count": 1,
        "tp": 0,
        "fp": 0,
        "tn": 1,
        "fn": 0,
        "precision": None,
        "recall": None,
        "specificity": 1.0,
    }


def test_summarize_records_compares_human_labels_with_aux_results() -> None:
    rows = [
        {
            "label": label,
            "falldata_aux": {"status": "ok", "confirmed": confirmed},
        }
        for label, confirmed in [
            ("fall", True),
            ("fall", False),
            ("non_fall", True),
            ("non_fall", False),
        ]
    ]

    evaluation = summarize_fall_shadow_review.summarize_records(rows)[
        "human_label_aux_evaluation"
    ]

    assert evaluation == {
        "evaluated_count": 4,
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.5,
    }


def test_read_jsonl_reports_parse_errors(tmp_path) -> None:
    review_log = tmp_path / "fall_shadow_review.jsonl"
    review_log.write_text(
        json.dumps({"event_id": "ok"}) + "\nnot-json\n",
        encoding="utf-8",
    )

    rows, errors = summarize_fall_shadow_review._read_jsonl(review_log)

    assert len(rows) == 1
    assert errors
