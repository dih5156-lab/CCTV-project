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
    assert summary["shadow_record_count"] == 2
    assert summary["confirmed_shadow_record_count"] == 0
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
    assert summary["confirmed_shadow_record_count"] == 1
    assert summary["max_fall_probability"] == 0.88
