from scripts.ops.evaluate_sample_deepstream_replay import _summarize_shadow_records


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
