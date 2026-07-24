import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ops import evaluate_sample_deepstream_replay as replay
from scripts.ops.evaluate_sample_deepstream_replay import (
    DEFAULT_CONTAINER_RTSP_URL,
    DEFAULT_HOST_RTSP_URL,
    DEFAULT_REVIEW_LOG,
    _host_path_from_container_path,
    _resolve_review_log_path,
    _summarize_shadow_records,
)


def test_default_rtsp_input_path_is_separate_from_sample_eval_output():
    assert DEFAULT_HOST_RTSP_URL == "rtsp://localhost:8554/sample_eval_input"
    assert (
        DEFAULT_CONTAINER_RTSP_URL
        == "rtsp://cctv-media-server:8554/sample_eval_input"
    )


def test_rtsp_replay_starts_publisher_before_restarting_ai_engine(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "scene_id": "scene-1",
                "video_path": str(video_path),
                "label": "fall",
                "is_fall": True,
                "scene_length": 30,
                "camera": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    call_order = []

    def fake_run_replay(
        video_path,
        rtsp_url,
        duration,
        timeout_grace,
        *,
        on_started=None,
    ):
        call_order.append("publisher_started")
        if on_started:
            on_started()
        call_order.append("publisher_waited")
        return 0

    monkeypatch.setattr(replay, "_run_ffmpeg_replay", fake_run_replay)
    monkeypatch.setattr(
        replay,
        "_restart_ai_engine",
        lambda *args: call_order.append("restart"),
    )
    monkeypatch.setattr(replay, "_video_duration_seconds", lambda *args: 1.0)
    monkeypatch.setattr(replay, "_read_new_jsonl_records", lambda *args: [])
    monkeypatch.setattr(replay, "_write_jsonl", lambda *args: None)
    monkeypatch.setattr(replay, "_write_csv", lambda *args: None)
    monkeypatch.setattr(replay, "_apply_camera_config", lambda *args: None)
    monkeypatch.setattr(replay.time, "sleep", lambda *args: None)

    args = SimpleNamespace(
        manifest=manifest_path,
        label=None,
        max_videos=1,
        source_mode="rtsp",
        container_project_root=Path("/app"),
        container_rtsp_url="rtsp://cctv-media-server:8554/sample_eval",
        host_rtsp_url="rtsp://localhost:8554/sample_eval",
        eval_cameras_json=tmp_path / "eval-cameras.json",
        camera_id="sample_eval",
        compose_env_file=tmp_path / ".env",
        review_log=tmp_path / "review.jsonl",
        prepare_only=False,
        apply_camera_config=True,
        cameras_json=tmp_path / "cameras.json",
        restart_ai_engine=True,
        compose_file=tmp_path / "compose.yml",
        restart_wait_seconds=0.0,
        assumed_fps=30.0,
        shadow_wait_seconds=0.0,
        timeout_grace_seconds=1.0,
        results_jsonl=tmp_path / "results.jsonl",
        results_csv=tmp_path / "results.csv",
        restore_camera_config=False,
    )

    replay.evaluate(args)

    assert call_order == ["publisher_started", "restart", "publisher_waited"]


def test_ffmpeg_replay_publishes_rtsp_over_tcp(tmp_path, monkeypatch):
    captured_command = []

    class CompletedProcess:
        def wait(self, timeout):
            return 0

    monkeypatch.setattr(replay.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        replay.subprocess,
        "Popen",
        lambda command: captured_command.extend(command) or CompletedProcess(),
    )

    replay._run_ffmpeg_replay(
        tmp_path / "sample.mp4",
        "rtsp://localhost:8554/sample_eval",
        1.0,
        1.0,
    )

    transport_index = captured_command.index("-rtsp_transport")
    assert captured_command[transport_index + 1] == "tcp"


def test_ffmpeg_replay_rejects_nonzero_publisher_exit(tmp_path, monkeypatch):
    class FailedProcess:
        def wait(self, timeout):
            return 1

    monkeypatch.setattr(replay.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(replay.subprocess, "Popen", lambda command: FailedProcess())

    with pytest.raises(RuntimeError, match="ffmpeg replay failed with exit code 1"):
        replay._run_ffmpeg_replay(
            tmp_path / "sample.mp4",
            "rtsp://localhost:8554/sample_eval_input",
            1.0,
            1.0,
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
