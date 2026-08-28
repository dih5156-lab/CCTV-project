import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ops"
    / "build_fall_error_review.py"
)
spec = importlib.util.spec_from_file_location("build_fall_error_review", SCRIPT_PATH)
reviewer = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["build_fall_error_review"] = reviewer
spec.loader.exec_module(reviewer)


def _manifest_row(scene_id: str, video_path: str, is_fall: bool) -> dict:
    return {
        "scene_id": scene_id,
        "scene_group": scene_id.rsplit("_", 1)[0],
        "video_path": video_path,
        "is_fall": is_fall,
        "fall_start_frame": 10 if is_fall else 0,
        "fall_end_frame": 20 if is_fall else 0,
    }


def test_build_error_candidates_joins_metrics_to_original_videos(tmp_path) -> None:
    fall_video = tmp_path / "fall.mp4"
    non_fall_video = tmp_path / "non-fall.mp4"
    fall_video.write_bytes(b"video")
    non_fall_video.write_bytes(b"video")
    train_rows = {
        "non-fall": _manifest_row("non-fall", str(non_fall_video), False),
    }
    validation_rows = {
        "fall": _manifest_row("fall", str(fall_video), True),
    }
    metrics = {
        "dataset_version": "full-14702",
        "model_params": {"decision_threshold": 0.7},
        "holdout_errors": {
            "false_positives": [
                {"scene_id": "non-fall", "true": 0, "predicted": 1,
                 "probability": [0.2, 0.8]}
            ],
            "false_negatives": [],
        },
        "validation": {
            "errors": {
                "false_positives": [],
                "false_negatives": [
                    {"scene_id": "fall", "true": 1, "predicted": 0,
                     "probability": [0.6, 0.4]}
                ],
            }
        },
    }

    candidates = reviewer.build_error_candidates(
        metrics, train_rows=train_rows, validation_rows=validation_rows
    )

    assert [row["review_id"] for row in candidates] == [
        "holdout:false_positive:non-fall",
        "validation:false_negative:fall",
    ]
    assert candidates[0]["original_label"] == "non_fall"
    assert candidates[0]["predicted_label"] == "fall"
    assert candidates[0]["fall_probability"] == 0.8
    assert candidates[0]["video_path"] == str(non_fall_video)
    assert candidates[1]["original_label"] == "fall"
    assert candidates[1]["fall_probability"] == 0.4


def test_build_review_document_renders_paged_video_review_and_all_labels(tmp_path) -> None:
    candidates = []
    for index in range(3):
        clip = tmp_path / f"clip-{index}.mp4"
        clip.write_bytes(b"video")
        candidates.append(
            {
                "review_id": f"holdout:false_negative:scene-{index}",
                "scene_id": f"scene-{index}",
                "evaluation_split": "holdout",
                "error_type": "false_negative",
                "original_label": "fall",
                "predicted_label": "non_fall",
                "fall_probability": 0.2,
                "video_path": str(clip),
            }
        )

    document = reviewer.build_review_document(
        candidates,
        base_dir=tmp_path,
        dataset_version="full-14702",
        page_size=2,
    )

    assert "오탐·미탐 원본 영상 재검수" in document
    assert "페이지당 2개" in document
    assert "data-label=\"fall\"" in document
    assert "data-label=\"non_fall\"" in document
    assert "data-label=\"ambiguous\"" in document
    assert "data-label=\"exclude\"" in document
    assert "fall_error_review_labels.json" in document
    assert "holdout:false_negative:scene-0" in document
    assert "clip-0.mp4" in document
    assert "const pageSize = 2" in document
    assert "function showPage({ scrollToTop = false } = {})" in document
    assert "refreshCard(card); showPage();" in document
    assert "showPage({scrollToTop: true});" in document


def test_write_candidates_jsonl_preserves_review_metadata(tmp_path) -> None:
    output = tmp_path / "errors.jsonl"
    candidates = [{"review_id": "holdout:false_positive:scene", "scene_id": "scene"}]

    reviewer.write_candidates_jsonl(output, candidates)

    assert json.loads(output.read_text().strip()) == candidates[0]


def test_filter_error_candidates_selects_error_type_and_scene_pattern() -> None:
    candidates = [
        {"scene_id": "fall_back_BY_C2", "error_type": "false_negative"},
        {"scene_id": "fall_front_FY_C2", "error_type": "false_negative"},
        {"scene_id": "normal_back_BY_C2", "error_type": "false_positive"},
    ]

    filtered = reviewer.filter_error_candidates(
        candidates,
        error_type="false_negative",
        scene_id_pattern=r"_BY_C2$",
    )

    assert filtered == [candidates[0]]


def test_select_evaluation_metrics_wraps_comparison_evaluation() -> None:
    comparison = {
        "threshold": 0.71,
        "evaluations": {
            "hard475": {
                "errors": {
                    "false_positives": [],
                    "false_negatives": [{"scene_id": "fall"}],
                }
            }
        },
    }

    selected = reviewer.select_evaluation_metrics(comparison, "hard475")

    assert selected["dataset_version"] == "hard475"
    assert selected["validation"] == comparison["evaluations"]["hard475"]
    assert selected["model_params"]["decision_threshold"] == 0.71
