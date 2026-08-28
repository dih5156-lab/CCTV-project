from datetime import datetime, timezone

from scripts.ops.check_fall_shadow_operation import (
    analyze_deepstream_stats,
    analyze_shadow_records,
)


def test_analyze_shadow_records_reports_fresh_successful_shadow_activity() -> None:
    now = datetime(2026, 8, 13, 1, 10, tzinfo=timezone.utc)
    rows = [
        {
            "created_at": "2026-08-13T01:09:30+00:00",
            "event_type": "fall_shadow_window",
            "falldata_aux_publish_pending": False,
            "falldata_aux": {
                "mode": "shadow",
                "status": "ok",
                "threshold": 0.78,
            },
        },
        {
            "created_at": "2026-08-13T01:10:00+00:00",
            "event_type": "fall_shadow_window",
            "falldata_aux_publish_pending": False,
            "falldata_aux": {
                "mode": "shadow",
                "status": "ok",
                "threshold": 0.78,
            },
        },
    ]

    result = analyze_shadow_records(
        rows,
        now=now,
        max_age_seconds=90,
        expected_threshold=0.78,
    )

    assert result["passed"] is True
    assert result["records"] == 2
    assert result["status_counts"] == {"ok": 2}
    assert result["latest_age_seconds"] == 0.0
    assert result["publish_pending_count"] == 0


def test_analyze_shadow_records_fails_on_stale_or_risky_records() -> None:
    now = datetime(2026, 8, 13, 1, 10, tzinfo=timezone.utc)
    rows = [
        {
            "created_at": "2026-08-13T01:00:00+00:00",
            "event_type": "fall_shadow_window",
            "falldata_aux_publish_pending": True,
            "falldata_aux": {
                "mode": "confirm",
                "status": "error",
                "threshold": 0.7,
            },
        }
    ]

    result = analyze_shadow_records(
        rows,
        now=now,
        max_age_seconds=90,
        expected_threshold=0.78,
    )

    assert result["passed"] is False
    assert result["runtime_failure_count"] == 1
    assert result["publish_pending_count"] == 1
    assert "stale_shadow_records" in result["failures"]
    assert "non_shadow_mode" in result["failures"]
    assert "threshold_mismatch" in result["failures"]


def test_analyze_deepstream_stats_requires_recent_frame_progress() -> None:
    logs = """
2026-08-13 01:09:40 [INFO] DeepStream stats: frames=100 frame_dropped=0 failed=0
2026-08-13 01:09:50 [INFO] DeepStream stats: frames=400 frame_dropped=0 failed=0
"""
    now = datetime(2026, 8, 13, 1, 10, tzinfo=timezone.utc)

    result = analyze_deepstream_stats(logs, now=now, max_age_seconds=30)

    assert result["passed"] is True
    assert result["frame_progress"] == 300
    assert result["latest_frames"] == 400


def test_analyze_deepstream_stats_fails_without_progress() -> None:
    logs = """
2026-08-13 01:09:40 [INFO] DeepStream stats: frames=100 frame_dropped=0 failed=0
2026-08-13 01:09:50 [INFO] DeepStream stats: frames=100 frame_dropped=0 failed=0
"""
    now = datetime(2026, 8, 13, 1, 10, tzinfo=timezone.utc)

    result = analyze_deepstream_stats(logs, now=now, max_age_seconds=30)

    assert result["passed"] is False
    assert "no_frame_progress" in result["failures"]
