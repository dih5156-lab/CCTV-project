
from scripts.ops import prepare_reviewed_fall_retraining
from scripts.ops.prepare_reviewed_fall_retraining import prepare_retraining_inputs


def test_prepare_retraining_inputs_uses_only_holdout_confirmed_errors() -> None:
    train_rows = [
        {"scene_id": "confirmed", "is_fall": True, "label": "fall"},
        {"scene_id": "excluded", "is_fall": True, "label": "fall"},
        {"scene_id": "ordinary", "is_fall": False, "label": "not_fall"},
    ]
    error_rows = [
        {
            "review_id": "holdout:false_negative:confirmed",
            "scene_id": "confirmed",
            "evaluation_split": "holdout",
            "original_label": "fall",
        },
        {
            "review_id": "holdout:false_negative:excluded",
            "scene_id": "excluded",
            "evaluation_split": "holdout",
            "original_label": "fall",
        },
        {
            "review_id": "validation:false_negative:validation-only",
            "scene_id": "validation-only",
            "evaluation_split": "validation",
            "original_label": "fall",
        },
    ]
    labels = {
        "schema_version": 1,
        "items": [
            {"review_id": "holdout:false_negative:confirmed", "label": "fall"},
            {"review_id": "holdout:false_negative:excluded", "label": "exclude"},
            {"review_id": "validation:false_negative:validation-only", "label": "fall"},
        ],
    }

    prepared = prepare_retraining_inputs(
        train_rows=train_rows,
        error_rows=error_rows,
        labels_payload=labels,
        reviewed_weight=3.0,
    )

    assert [row["scene_id"] for row in prepared["train_rows"]] == [
        "confirmed",
        "ordinary",
    ]
    assert prepared["reviewed_hard_cases"] == {
        "schema_version": 1,
        "items": [{"scene_id": "confirmed", "weight": 3.0}],
    }
    assert prepared["summary"]["excluded_from_train"] == 1
    assert prepared["summary"]["validation_feedback_preserved"] == 1


def test_prepare_retraining_inputs_applies_corrected_binary_label() -> None:
    train_rows = [{"scene_id": "corrected", "is_fall": True, "label": "fall"}]
    error_rows = [
        {
            "review_id": "holdout:false_negative:corrected",
            "scene_id": "corrected",
            "evaluation_split": "holdout",
            "original_label": "fall",
        }
    ]
    labels = {
        "schema_version": 1,
        "items": [
            {"review_id": "holdout:false_negative:corrected", "label": "non_fall"}
        ],
    }

    prepared = prepare_retraining_inputs(
        train_rows=train_rows,
        error_rows=error_rows,
        labels_payload=labels,
        reviewed_weight=2.0,
    )

    assert prepared["train_rows"][0]["is_fall"] is False
    assert prepared["train_rows"][0]["label"] == "not_fall"
    assert prepared["summary"]["corrected_labels"] == 1


def test_prepare_retraining_inputs_rejects_unknown_review_id() -> None:
    labels = {
        "schema_version": 1,
        "items": [{"review_id": "unknown", "label": "fall"}],
    }

    try:
        prepare_retraining_inputs(
            train_rows=[],
            error_rows=[],
            labels_payload=labels,
            reviewed_weight=3.0,
        )
    except ValueError as exc:
        assert "unknown review_id" in str(exc)
    else:
        raise AssertionError("unknown review ID should fail")


def test_merge_review_inputs_appends_rows_errors_and_labels() -> None:
    merged = prepare_reviewed_fall_retraining.merge_review_inputs(
        train_rows=[{"scene_id": "base"}],
        error_rows=[{"review_id": "base-review"}],
        labels_payload={
            "schema_version": 1,
            "items": [{"review_id": "base-review", "label": "fall"}],
        },
        additional_train_rows=[{"scene_id": "blind"}],
        additional_error_rows=[{"review_id": "blind-review"}],
        additional_labels_payload={
            "schema_version": 1,
            "items": [{"review_id": "blind-review", "label": "fall"}],
        },
    )

    assert [row["scene_id"] for row in merged["train_rows"]] == ["base", "blind"]
    assert [row["review_id"] for row in merged["error_rows"]] == [
        "base-review",
        "blind-review",
    ]
    assert len(merged["labels_payload"]["items"]) == 2


def test_append_reviewed_training_cases_adds_confirmed_rows_and_weights() -> None:
    appended = prepare_reviewed_fall_retraining.append_reviewed_training_cases(
        train_rows=[{"scene_id": "base", "is_fall": False}],
        reviewed_hard_cases={
            "schema_version": 1,
            "items": [{"scene_id": "base", "weight": 3.0}],
        },
        additional_error_rows=[
            {
                "review_id": "validation:false_negative:back-fall",
                "scene_id": "back-fall",
                "is_fall": True,
                "label": "fall",
            },
            {
                "review_id": "validation:false_negative:skip",
                "scene_id": "skip",
                "is_fall": True,
                "label": "fall",
            },
        ],
        additional_labels_payload={
            "schema_version": 1,
            "items": [
                {
                    "review_id": "validation:false_negative:back-fall",
                    "label": "fall",
                },
                {
                    "review_id": "validation:false_negative:skip",
                    "label": "exclude",
                },
            ],
        },
        reviewed_weight=3.0,
        additional_host_data_root="/host/data",
        additional_container_data_root="/app/data",
    )

    assert [row["scene_id"] for row in appended["train_rows"]] == [
        "base",
        "back-fall",
    ]
    assert appended["reviewed_hard_cases"]["items"] == [
        {"scene_id": "base", "weight": 3.0},
        {"scene_id": "back-fall", "weight": 3.0},
    ]
    assert appended["summary"] == {
        "source_train_rows": 1,
        "prepared_train_rows": 2,
        "source_reviewed_hard_cases": 1,
        "reviewed_training_hard_cases": 2,
        "additional_review_items": 2,
        "additional_training_rows": 1,
        "additional_excluded_or_ambiguous": 1,
        "additional_corrected_labels": 0,
    }


def test_append_reviewed_training_cases_rewrites_host_video_path() -> None:
    appended = prepare_reviewed_fall_retraining.append_reviewed_training_cases(
        train_rows=[],
        reviewed_hard_cases={"schema_version": 1, "items": []},
        additional_error_rows=[
            {
                "review_id": "validation:false_negative:fall",
                "scene_id": "fall",
                "video_path": "/host/data/validation/fall.mp4",
                "is_fall": True,
            }
        ],
        additional_labels_payload={
            "schema_version": 1,
            "items": [
                {"review_id": "validation:false_negative:fall", "label": "fall"}
            ],
        },
        reviewed_weight=3.0,
        additional_host_data_root="/host/data",
        additional_container_data_root="/app/data",
    )

    assert appended["train_rows"][0]["video_path"] == "/app/data/validation/fall.mp4"
