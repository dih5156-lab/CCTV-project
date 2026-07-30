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
from src.core.ai.fall_temporal_model import FRAME_FEATURE_NAMES


def test_label_feature_capture_records_adds_manifest_metadata() -> None:
    capture = {
        "schema_version": 1,
        "runtime": "deepstream_pose_inline",
        "camera_id": "sample_eval",
        "feature_names": ["torso_angle_mean", "hip_speed_max"],
        "feature_vector": [41.5, 0.82],
    }
    manifest_row = {
        "video_path": "/dataset/scene-001.mp4",
        "scene_id": "scene-001",
        "group_id": "subject-001",
        "is_fall": True,
    }

    labeled, errors = replay._label_feature_capture_records(
        [capture],
        manifest_row,
    )

    assert errors == []
    assert labeled == [
        {
            **capture,
            "label": 1,
            "is_fall": True,
            "scene_id": "scene-001",
            "group_id": "subject-001",
            "video_path": "/dataset/scene-001.mp4",
        }
    ]


def test_label_feature_capture_records_uses_scene_group_as_group_id() -> None:
    labeled, errors = replay._label_feature_capture_records(
        [
            {
                "schema_version": 1,
                "runtime": "deepstream_pose_inline",
                "feature_names": ["a"],
                "feature_vector": [1.0],
            }
        ],
        {
            "video_path": "/dataset/scene-001.mp4",
            "scene_id": "scene-001",
            "scene_group": "subject-001",
            "is_fall": False,
        },
    )

    assert errors == []
    assert labeled[0]["group_id"] == "subject-001"


def test_label_feature_capture_records_preserves_temporal_sequence_metadata() -> None:
    frame_records = [
        {"timestamp": float(index), "frame_index": 120 + index}
        for index in range(48)
    ]
    capture = {
        "schema_version": 2,
        "runtime": "deepstream_pose_inline",
        "feature_names": ["frames_seen"],
        "feature_vector": [48.0],
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "frame_records": frame_records,
    }
    manifest_row = {
        "video_path": "/dataset/scene-001.mp4",
        "scene_id": "scene-001",
        "scene_group": "subject-001",
        "is_fall": True,
        "fall_start_frame": 120,
        "fall_end_frame": 180,
        "scene_position": "복도",
        "scene_location": "병원",
        "age_group": "노인",
        "fall_direction": "뒤",
    }

    labeled, errors = replay._label_feature_capture_records(
        [capture],
        manifest_row,
    )

    assert errors == []
    assert labeled[0]["frame_records"] == frame_records
    assert labeled[0]["group_id"] == "subject-001"
    assert labeled[0]["fall_start_frame"] == 120
    assert labeled[0]["fall_end_frame"] == 180
    assert labeled[0]["scene_position"] == "복도"
    assert labeled[0]["scene_location"] == "병원"
    assert labeled[0]["age_group"] == "노인"
    assert labeled[0]["fall_direction"] == "뒤"


def test_label_feature_capture_records_drops_positive_window_before_fall() -> None:
    capture = {
        "schema_version": 2,
        "runtime": "deepstream_pose_inline",
        "feature_names": ["frames_seen"],
        "feature_vector": [48.0],
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "frame_records": [
            {"timestamp": float(index), "frame_index": index}
            for index in range(48)
        ],
    }

    labeled, errors = replay._label_feature_capture_records(
        [capture],
        {
            "video_path": "/dataset/fall.mp4",
            "scene_id": "fall-before-window",
            "scene_group": "subject-001",
            "is_fall": True,
            "fall_start_frame": 120,
            "fall_end_frame": 180,
        },
    )

    assert errors == []
    assert labeled == []


def test_label_feature_capture_records_drops_positive_window_without_fall_frame() -> None:
    capture = {
        "schema_version": 2,
        "runtime": "deepstream_pose_inline",
        "feature_names": ["frames_seen"],
        "feature_vector": [4.0],
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "frame_records": [
            {"timestamp": float(frame_index), "frame_index": frame_index}
            for frame_index in (0, 20, 100, 120)
        ],
    }

    labeled, errors = replay._label_feature_capture_records(
        [capture],
        {
            "video_path": "/dataset/fall.mp4",
            "scene_id": "fall-with-pose-gap",
            "scene_group": "subject-001",
            "is_fall": True,
            "fall_start_frame": 30,
            "fall_end_frame": 90,
        },
    )

    assert errors == []
    assert labeled == []


def test_label_feature_capture_records_requires_frame_index_for_positive_v2() -> None:
    capture = {
        "schema_version": 2,
        "runtime": "deepstream_pose_inline",
        "feature_names": ["frames_seen"],
        "feature_vector": [48.0],
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "frame_records": [{"timestamp": float(index)} for index in range(48)],
    }

    labeled, errors = replay._label_feature_capture_records(
        [capture],
        {
            "video_path": "/dataset/fall.mp4",
            "scene_id": "fall-missing-index",
            "scene_group": "subject-001",
            "is_fall": True,
            "fall_start_frame": 120,
            "fall_end_frame": 180,
        },
    )

    assert labeled == []
    assert errors == ["record 0: positive frame_records require frame_index"]


@pytest.mark.parametrize(
    "capture, expected_error",
    [
        (
            {
                "schema_version": 3,
                "runtime": "deepstream_pose_inline",
                "feature_names": ["a"],
                "feature_vector": [1.0],
            },
            "unsupported schema_version",
        ),
        (
            {
                "schema_version": 1,
                "runtime": "offline",
                "feature_names": ["a"],
                "feature_vector": [1.0],
            },
            "unexpected runtime",
        ),
        (
            {
                "schema_version": 1,
                "runtime": "deepstream_pose_inline",
                "feature_names": ["a", "b"],
                "feature_vector": [1.0],
            },
            "feature length mismatch",
        ),
        (
            {
                "schema_version": 2,
                "runtime": "deepstream_pose_inline",
                "feature_names": ["a"],
                "feature_vector": [1.0],
                "frame_feature_names": list(FRAME_FEATURE_NAMES),
                "frame_records": [],
            },
            "invalid frame_records",
        ),
    ],
)
def test_label_feature_capture_records_rejects_invalid_records(
    capture,
    expected_error,
) -> None:
    labeled, errors = replay._label_feature_capture_records(
        [capture],
        {
            "video_path": "/dataset/scene-001.mp4",
            "scene_id": "scene-001",
            "group_id": "subject-001",
            "is_fall": False,
        },
    )

    assert labeled == []
    assert expected_error in errors[0]


def test_default_rtsp_input_path_is_separate_from_sample_eval_output():
    assert DEFAULT_HOST_RTSP_URL == "rtsp://localhost:8554/sample_eval_input"
    assert (
        DEFAULT_CONTAINER_RTSP_URL
        == "rtsp://cctv-media-server:8554/sample_eval_input"
    )


def test_recreate_ai_engine_passes_scoped_environment(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setenv("UNCHANGED_ENV", "kept")
    monkeypatch.setattr(
        replay.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    replay._recreate_ai_engine(
        tmp_path / "docker-compose.jetson.yml",
        tmp_path / ".env",
        environment_overrides={
            "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH":
                "/app/data/fall_eval/capture.jsonl",
            "FALLDATA_AUX_COMPARE_MODEL_PATH":
                "/app/models/candidate.joblib",
        },
    )

    command, kwargs = calls[0]
    assert command[-4:] == [
        "up",
        "-d",
        "--force-recreate",
        "cctv-ai-engine",
    ]
    assert kwargs["check"] is True
    assert kwargs["env"]["UNCHANGED_ENV"] == "kept"
    assert (
        kwargs["env"]["FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH"]
        == "/app/data/fall_eval/capture.jsonl"
    )
    assert (
        kwargs["env"]["FALLDATA_AUX_COMPARE_MODEL_PATH"]
        == "/app/models/candidate.joblib"
    )


def test_resolve_project_container_path_maps_relative_path(tmp_path) -> None:
    host_path, container_path = replay._resolve_project_container_path(
        Path("data/fall_eval/capture.jsonl"),
        host_project_root=tmp_path,
        container_project_root=Path("/app"),
    )

    assert host_path == tmp_path / "data/fall_eval/capture.jsonl"
    assert container_path == Path("/app/data/fall_eval/capture.jsonl")


def test_resolve_project_container_path_rejects_outside_project(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="project root"):
        replay._resolve_project_container_path(
            tmp_path.parent / "outside.jsonl",
            host_project_root=tmp_path,
            container_project_root=Path("/app"),
        )


def test_main_parses_feature_capture_and_candidate_model_arguments(
    monkeypatch,
) -> None:
    captured_args = []
    monkeypatch.setattr(replay, "evaluate", lambda args: captured_args.append(args) or [])
    monkeypatch.setattr(
        replay.sys,
        "argv",
        [
            "evaluate_sample_deepstream_replay.py",
            "--feature-capture-log",
            "data/fall_eval/capture.jsonl",
            "--feature-dataset-jsonl",
            "data/fall_eval/dataset.jsonl",
            "--runtime-compare-model-path",
            "models/falldata/candidate.joblib",
            "--scene-position",
            "복도",
            "--prepare-only",
        ],
    )

    assert replay.main() == 0
    [args] = captured_args
    assert args.feature_capture_log == Path(
        "data/fall_eval/capture.jsonl"
    )
    assert args.feature_dataset_jsonl == Path(
        "data/fall_eval/dataset.jsonl"
    )
    assert args.runtime_compare_model_path == Path(
        "models/falldata/candidate.joblib"
    )
    assert args.scene_position == "복도"


def test_filter_manifest_rows_applies_label_position_and_limit() -> None:
    rows = [
        {
            "scene_id": "room-normal",
            "label": "not_fall",
            "scene_position": "병실",
        },
        {
            "scene_id": "corridor-normal-1",
            "label": "not_fall",
            "scene_position": "복도",
        },
        {
            "scene_id": "corridor-fall",
            "label": "fall",
            "scene_position": "복도",
        },
        {
            "scene_id": "corridor-normal-2",
            "label": "not_fall",
            "scene_position": "복도",
        },
    ]

    filtered = replay._filter_manifest_rows(
        rows,
        label="not_fall",
        scene_position="복도",
        max_videos=1,
    )

    assert [row["scene_id"] for row in filtered] == [
        "corridor-normal-1"
    ]


def test_filter_manifest_rows_selects_exact_scene_ids_in_requested_order() -> None:
    rows = [
        {"scene_id": "scene-a", "label": "not_fall"},
        {"scene_id": "scene-b", "label": "not_fall"},
        {"scene_id": "scene-c", "label": "not_fall"},
    ]

    filtered = replay._filter_manifest_rows(
        rows,
        label="not_fall",
        scene_position=None,
        max_videos=0,
        scene_ids=("scene-c", "scene-a"),
    )

    assert [row["scene_id"] for row in filtered] == ["scene-c", "scene-a"]


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


def test_file_replay_restores_camera_config_when_initial_restart_fails(
    tmp_path,
    monkeypatch,
):
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
    backup_path = tmp_path / "cameras.backup.json"
    backup_path.write_text("{}", encoding="utf-8")
    copied = []
    restart_count = 0

    monkeypatch.setattr(replay, "_apply_camera_config", lambda *args: backup_path)

    def fail_first_restart(*args):
        nonlocal restart_count
        restart_count += 1
        if restart_count == 1:
            raise RuntimeError("restart failed")

    monkeypatch.setattr(replay, "_restart_ai_engine", fail_first_restart)
    monkeypatch.setattr(
        replay.shutil,
        "copy2",
        lambda source, destination: copied.append((source, destination)),
    )

    args = SimpleNamespace(
        manifest=manifest_path,
        label=None,
        max_videos=1,
        source_mode="file",
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
        restore_camera_config=True,
    )

    with pytest.raises(RuntimeError, match="restart failed"):
        replay.evaluate(args)

    assert (backup_path, args.cameras_json) in copied


def test_feature_capture_labels_new_records_and_restores_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "scene_id": "scene-1",
                "group_id": "subject-1",
                "video_path": str(video_path),
                "label": "not_fall",
                "is_fall": False,
                "scene_length": 30,
                "camera": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backup_path = tmp_path / "cameras.backup.json"
    backup_path.write_text("{}", encoding="utf-8")
    capture_path = tmp_path / "data/fall_eval/capture.jsonl"
    dataset_path = tmp_path / "data/fall_eval/dataset.jsonl"
    recreate_calls = []

    def fake_recreate(*_args, **kwargs):
        environment_overrides = kwargs.get("environment_overrides")
        recreate_calls.append(environment_overrides)
        if environment_overrides:
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            capture_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "runtime": "deepstream_pose_inline",
                        "camera_id": "sample_eval",
                        "feature_names": ["a", "b"],
                        "feature_vector": [0.1, 0.2],
                        "frame_feature_names": list(FRAME_FEATURE_NAMES),
                        "frame_records": [
                            {"timestamp": float(index)}
                            for index in range(48)
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(replay, "_apply_camera_config", lambda *_args: backup_path)
    monkeypatch.setattr(replay, "_restart_ai_engine", lambda *_args: None)
    monkeypatch.setattr(
        replay,
        "_recreate_ai_engine",
        fake_recreate,
    )
    monkeypatch.setattr(replay, "_video_duration_seconds", lambda *_args: 0.1)
    monkeypatch.setattr(replay.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(replay.shutil, "copy2", lambda *_args: None)

    args = SimpleNamespace(
        manifest=manifest_path,
        label=None,
        max_videos=1,
        source_mode="file",
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
        compose_file=tmp_path / "docker-compose.jetson.yml",
        restart_wait_seconds=0.0,
        assumed_fps=30.0,
        shadow_wait_seconds=0.0,
        timeout_grace_seconds=1.0,
        score_source="inline_pose_rf",
        runtime_result_timeout_seconds=0.0,
        runtime_result_poll_seconds=0.1,
        results_jsonl=tmp_path / "results.jsonl",
        results_csv=tmp_path / "results.csv",
        restore_camera_config=True,
        feature_capture_log=Path("data/fall_eval/capture.jsonl"),
        feature_dataset_jsonl=Path("data/fall_eval/dataset.jsonl"),
        runtime_compare_model_path=None,
    )

    [result] = replay.evaluate(args)

    [dataset_record] = replay._read_jsonl(dataset_path)
    assert dataset_record["scene_id"] == "scene-1"
    assert dataset_record["group_id"] == "subject-1"
    assert dataset_record["label"] == 0
    assert result["feature_capture_record_count"] == 1
    assert result["feature_capture_errors"] == []
    assert recreate_calls == [
        {
            "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH":
                "/app/data/fall_eval/capture.jsonl",
        },
        None,
    ]


def test_feature_capture_restores_runtime_when_replay_fails(
    tmp_path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "scene_id": "scene-1",
                "group_id": "subject-1",
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
    backup_path = tmp_path / "cameras.backup.json"
    backup_path.write_text("{}", encoding="utf-8")
    recreate_calls = []

    monkeypatch.setattr(replay, "_apply_camera_config", lambda *_args: backup_path)
    monkeypatch.setattr(
        replay,
        "_restart_ai_engine",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("restart failed")),
    )
    monkeypatch.setattr(
        replay,
        "_recreate_ai_engine",
        lambda *_args, **kwargs: recreate_calls.append(
            kwargs.get("environment_overrides")
        ),
    )
    monkeypatch.setattr(replay, "_video_duration_seconds", lambda *_args: 0.1)
    monkeypatch.setattr(replay.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(replay.shutil, "copy2", lambda *_args: None)

    args = SimpleNamespace(
        manifest=manifest_path,
        label=None,
        max_videos=1,
        source_mode="file",
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
        compose_file=tmp_path / "docker-compose.jetson.yml",
        restart_wait_seconds=0.0,
        assumed_fps=30.0,
        shadow_wait_seconds=0.0,
        timeout_grace_seconds=1.0,
        score_source="inline_pose_rf",
        runtime_result_timeout_seconds=0.0,
        runtime_result_poll_seconds=0.1,
        results_jsonl=tmp_path / "results.jsonl",
        results_csv=tmp_path / "results.csv",
        restore_camera_config=True,
        feature_capture_log=Path("data/fall_eval/capture.jsonl"),
        feature_dataset_jsonl=Path("data/fall_eval/dataset.jsonl"),
        runtime_compare_model_path=None,
    )

    with pytest.raises(RuntimeError, match="restart failed"):
        replay.evaluate(args)

    assert recreate_calls == [
        {
            "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH":
                "/app/data/fall_eval/capture.jsonl",
        },
        None,
    ]


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


def test_summarize_shadow_records_reports_inline_pose_rf_normal_probability():
    records = [
        {
            "camera_id": "sample_eval",
            "falldata_aux": {
                "status": "ok",
                "runtime": "deepstream_pose_inline",
                "confirmed": False,
                "fall_probability": 0.42,
                "threshold": 0.7,
            },
        },
        {
            "camera_id": "sample_eval",
            "falldata_aux": {
                "status": "insufficient_pose_records",
                "runtime": "deepstream_pose_inline",
                "confirmed": False,
            },
        },
    ]

    summary = _summarize_shadow_records(records, "sample_eval")

    assert summary["inline_pose_rf_record_count"] == 1
    assert summary["inline_pose_rf_confirmed_record_count"] == 0
    assert summary["detected_by_inline_pose_rf"] is False
    assert summary["max_inline_pose_rf_probability"] == 0.42


def test_score_runtime_result_does_not_treat_missing_inline_result_as_negative():
    assert replay._score_runtime_result(True, False, evaluated=False) == "NO_RESULT"
    assert replay._score_runtime_result(False, False, evaluated=False) == "NO_RESULT"
    assert replay._score_runtime_result(True, True, evaluated=True) == "TP"
    assert replay._score_runtime_result(False, False, evaluated=True) == "TN"


def test_select_runtime_detection_requires_an_inline_pose_rf_record():
    no_result = replay._select_runtime_detection(
        {
            "detected": True,
            "detected_by_inline_pose_rf": False,
            "inline_pose_rf_record_count": 0,
        },
        "inline_pose_rf",
    )
    normal_result = replay._select_runtime_detection(
        {
            "detected": True,
            "detected_by_inline_pose_rf": False,
            "inline_pose_rf_record_count": 1,
        },
        "inline_pose_rf",
    )

    assert no_result == (False, False)
    assert normal_result == (False, True)


def test_read_runtime_records_waits_for_valid_inline_pose_rf_result(
    tmp_path,
    monkeypatch,
):
    insufficient = [
        {
            "camera_id": "sample_eval",
            "falldata_aux": {
                "status": "insufficient_pose_records",
                "runtime": "deepstream_pose_inline",
            },
        }
    ]
    valid = insufficient + [
        {
            "camera_id": "sample_eval",
            "falldata_aux": {
                "status": "ok",
                "runtime": "deepstream_pose_inline",
                "confirmed": False,
                "fall_probability": 0.3,
            },
        }
    ]
    reads = iter([insufficient, valid])
    monkeypatch.setattr(
        replay,
        "_read_new_jsonl_records",
        lambda *args: next(reads),
    )
    monotonic_values = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(
        replay.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(replay.time, "sleep", lambda *args: None)

    records = replay._read_runtime_records(
        tmp_path / "review.jsonl",
        0,
        "sample_eval",
        score_source="inline_pose_rf",
        timeout_seconds=5.0,
        poll_seconds=1.0,
    )

    assert records == valid


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
