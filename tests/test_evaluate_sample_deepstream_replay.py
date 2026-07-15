from pathlib import Path

from scripts.ops.evaluate_sample_deepstream_replay import (
    DEFAULT_REVIEW_LOG,
    _host_path_from_container_path,
    _resolve_review_log_path,
    _summarize_shadow_records,
)


def test_host_path_from_container_path_maps_app_relative_paths():
    assert _host_path_from_container_path(
        "/app/data/fall_dataset/annotations/review.jsonl",
        Path("/app"),
    ) == Path("data/fall_dataset/annotations/review.jsonl")


def test_resolve_review_log_uses_env_file_when_default_requested():
    resolved = _resolve_review_log_path(
        DEFAULT_REVIEW_LOG,
        {"FALL_SHADOW_REVIEW_LOG_PATH": "/app/data/fall_dataset/annotations/review.jsonl"},
        Path("/app"),
    )

    assert resolved == Path("data/fall_dataset/annotations/review.jsonl")


def test_resolve_review_log_keeps_explicit_argument():
    explicit = Path("custom/review.jsonl")

    resolved = _resolve_review_log_path(
        explicit,
        {"FALL_SHADOW_REVIEW_LOG_PATH": "/app/data/fall_dataset/annotations/review.jsonl"},
        Path("/app"),
    )

    assert resolved == explicit


def test_summarize_shadow_records_counts_only_ok_confirmed_records():
    records = [
        {
            "camera_id": "sample_eval",
            "falldata_aux": {
                "status": "skipped_cooldown",
                "confirmed": True,
                "fall_probability": 0.91,
            },
        },
        {
            "camera_id": "sample_eval",
            "falldata_aux": {
                "status": "ok",
                "confirmed": False,
                "fall_probability": 0.2,
            },
        },
    ]

    summary = _summarize_shadow_records(records, "sample_eval")

    assert summary["detected"] is False
    assert summary["detected_by_event"] is False
    assert summary["detected_by_aux"] is False
    assert summary["shadow_record_count"] == 2
    assert summary["fall_event_count"] == 0
    assert summary["fall_candidate_count"] == 0
    assert summary["confirmed_shadow_record_count"] == 0
    assert summary["aux_published_shadow_record_count"] == 0
    assert summary["max_fall_probability"] is None


def test_summarize_shadow_records_reports_confirmed_probability():
    records = [
        {
            "camera_id": "sample_eval",
            "falldata_aux": {
                "status": "ok",
                "confirmed": True,
                "fall_probability": 0.88,
            },
        },
        {
            "camera_id": "other",
            "falldata_aux": {
                "status": "ok",
                "confirmed": True,
                "fall_probability": 0.99,
            },
        },
    ]

    summary = _summarize_shadow_records(records, "sample_eval")

    assert summary["detected"] is True
    assert summary["detected_by_event"] is False
    assert summary["detected_by_aux"] is True
    assert summary["confirmed_shadow_record_count"] == 1
    assert summary["aux_published_shadow_record_count"] == 1
    assert summary["max_fall_probability"] == 0.88


def test_summarize_shadow_records_reports_fall_event_even_when_aux_errors():
    records = [
        {
            "camera_id": "sample_eval",
            "event_type": "fall_detected",
            "fall_score": 4.5,
            "falldata_aux": {
                "status": "error",
                "confirmed": False,
            },
        }
    ]

    summary = _summarize_shadow_records(records, "sample_eval")

    assert summary["detected"] is True
    assert summary["detected_by_event"] is True
    assert summary["detected_by_aux"] is False
    assert summary["fall_event_count"] == 1
    assert summary["fall_candidate_count"] == 1
    assert summary["max_fall_score"] == 4.5


def test_summarize_shadow_records_reports_near_miss_details():
    records = [
        {
            "camera_id": "sample_eval",
            "event_type": "fall_near_miss",
            "near_miss": {
                "type": "folded_floor_pose",
                "score": 0.0,
                "reasons": ["folded_floor_pose:0.38"],
            },
            "falldata_aux": {
                "status": "not_run",
                "confirmed": None,
            },
        },
        {
            "camera_id": "sample_eval",
            "event_type": "fall_near_miss",
            "near_miss": {
                "type": "low_score_pose",
                "score": 2.5,
                "reasons": ["torso_horizontal:44.3"],
            },
            "falldata_aux": {
                "status": "not_run",
                "confirmed": None,
            },
        },
    ]

    summary = _summarize_shadow_records(records, "sample_eval")

    assert summary["detected"] is False
    assert summary["near_miss_record_count"] == 2
    assert summary["near_miss_types"] == ["folded_floor_pose", "low_score_pose"]
    assert summary["max_near_miss_score"] == 2.5


def test_summarize_shadow_records_pending_borderline_requires_aux_confirmation():
    records = [
        {
            "camera_id": "sample_eval",
            "event_type": "fall_detected",
            "fall_score": 3.0,
            "falldata_aux_publish_pending": True,
            "falldata_aux": {
                "status": "ok",
                "confirmed": False,
                "fall_probability": 0.91,
            },
        }
    ]

    summary = _summarize_shadow_records(records, "sample_eval")

    assert summary["detected"] is False
    assert summary["detected_by_event"] is False
    assert summary["detected_by_aux"] is False
    assert summary["fall_event_count"] == 0
    assert summary["fall_candidate_count"] == 1


def test_summarize_shadow_records_reports_compare_model_separately():
    records = [
        {
            "camera_id": "sample_eval",
            "falldata_aux": {
                "status": "ok",
                "confirmed": False,
                "fall_probability": 0.82,
                "compare_model": {
                    "status": "ok",
                    "confirmed": True,
                    "fall_probability": 0.93,
                },
            },
        },
        {
            "camera_id": "sample_eval",
            "falldata_aux": {
                "status": "ok",
                "confirmed": True,
                "fall_probability": 0.91,
                "compare_model": {
                    "status": "ok",
                    "confirmed": False,
                    "fall_probability": 0.34,
                },
            },
        },
    ]

    summary = _summarize_shadow_records(records, "sample_eval")

    assert summary["detected"] is True
    assert summary["detected_by_aux"] is True
    assert summary["detected_by_compare_aux"] is True
    assert summary["confirmed_shadow_record_count"] == 1
    assert summary["aux_published_shadow_record_count"] == 1
    assert summary["compare_model_record_count"] == 2
    assert summary["compare_confirmed_shadow_record_count"] == 1
    assert summary["max_fall_probability"] == 0.91
    assert summary["max_compare_fall_probability"] == 0.93
    assert summary["last_compare_status"] == "ok"


def test_summarize_shadow_records_compare_veto_marks_confirmed_aux_unpublished():
    records = [
        {
            "camera_id": "sample_eval",
            "event_type": "fall_detected",
            "fall_score": 6.0,
            "falldata_aux_publish_pending": True,
            "falldata_aux": {
                "status": "ok",
                "confirmed": True,
                "fall_probability": 0.92,
                "compare_model": {
                    "status": "ok",
                    "confirmed": False,
                    "fall_probability": 0.49,
                },
            },
        }
    ]

    summary = _summarize_shadow_records(
        records,
        "sample_eval",
        compare_veto_enabled=True,
        compare_veto_min_fall_score=5.0,
    )

    assert summary["detected"] is False
    assert summary["detected_by_aux"] is False
    assert summary["confirmed_shadow_record_count"] == 1
    assert summary["aux_published_shadow_record_count"] == 0
    assert summary["detected_by_compare_aux"] is False


def test_summarize_shadow_records_compare_veto_ignores_scores_below_minimum():
    records = [
        {
            "camera_id": "sample_eval",
            "event_type": "fall_detected",
            "fall_score": 3.0,
            "falldata_aux_publish_pending": True,
            "falldata_aux": {
                "status": "ok",
                "confirmed": True,
                "fall_probability": 0.92,
                "compare_model": {
                    "status": "ok",
                    "confirmed": False,
                    "fall_probability": 0.49,
                },
            },
        }
    ]

    summary = _summarize_shadow_records(
        records,
        "sample_eval",
        compare_veto_enabled=True,
        compare_veto_min_fall_score=5.0,
    )

    assert summary["detected"] is True
    assert summary["detected_by_aux"] is True
    assert summary["confirmed_shadow_record_count"] == 1
    assert summary["aux_published_shadow_record_count"] == 1


def test_summarize_shadow_records_pending_borderline_detects_when_aux_confirmed():
    records = [
        {
            "camera_id": "sample_eval",
            "event_type": "fall_detected",
            "fall_score": 3.0,
            "falldata_aux_publish_pending": True,
            "falldata_aux": {
                "status": "ok",
                "confirmed": True,
                "fall_probability": 0.91,
            },
        }
    ]

    summary = _summarize_shadow_records(records, "sample_eval")

    assert summary["detected"] is True
    assert summary["detected_by_event"] is False
    assert summary["detected_by_aux"] is True
    assert summary["fall_event_count"] == 0
    assert summary["fall_candidate_count"] == 1
