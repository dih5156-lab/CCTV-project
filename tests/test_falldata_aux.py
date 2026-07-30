import json
import subprocess

import numpy as np

from scripts.datasets.train_yolo_pose_fall_rf import FEATURE_NAMES
from src.core.ai._falldata_aux import (
    PROJECT_ROOT,
    FallDataAuxConfig,
    FallDataAuxVerifier,
)
from src.core.ai.fall_temporal_model import FRAME_FEATURE_NAMES
from src.core.events import DetectionEvent, EventType


def _fall_event() -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.FALL_DETECTED,
        x=1,
        y=2,
        width=3,
        height=4,
        confidence=0.9,
        timestamp=1.0,
    )


def _pose_event(timestamp: float = 1.0) -> DetectionEvent:
    keypoints = [
        [float(10 + index), float(20 + index), 0.9]
        for index in range(17)
    ]
    return DetectionEvent(
        event_type=EventType.PERSON,
        x=10,
        y=20,
        width=80,
        height=160,
        confidence=0.92,
        timestamp=timestamp,
        keypoints=keypoints,
        metadata={
            "frame_num": 42,
            "frame_width": 640,
            "frame_height": 480,
            "fall_score": 3.5,
            "fall_reasons": ["torso_horizontal:0.80"],
        },
    )


class _FixedInlineClassifier:
    classes_ = np.asarray([0, 1])
    n_features_in_ = len(FEATURE_NAMES)

    def predict_proba(self, features):
        assert features.shape == (1, len(FEATURE_NAMES))
        return np.asarray([[0.08, 0.92]], dtype=np.float64)


def test_disabled_verifier_keeps_events_unchanged() -> None:
    verifier = FallDataAuxVerifier(FallDataAuxConfig(enabled=False))
    event = _fall_event()

    assert verifier.annotate_events([event]) == [event]
    verifier.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    assert verifier.annotate_events([event]) == [event]


def test_config_reads_optional_inline_feature_capture_path(
    monkeypatch,
    tmp_path,
) -> None:
    capture_path = tmp_path / "inline-features.jsonl"
    monkeypatch.setenv(
        "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH",
        str(capture_path),
    )

    config = FallDataAuxConfig.from_env()

    assert config.inline_feature_capture_path == capture_path


def test_config_disables_inline_feature_capture_when_env_is_blank(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH", "  ")

    config = FallDataAuxConfig.from_env()

    assert config.inline_feature_capture_path is None


def test_inline_feature_capture_writes_exact_summary_vector(tmp_path) -> None:
    capture_path = tmp_path / "inline-features.jsonl"
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            inline_pose_rf=True,
            inline_feature_capture_path=capture_path,
        )
    )
    frame_records = [
        {
            "timestamp": float(index),
            "fall_score": 3.5,
            "fall_reasons": ["torso_horizontal:0.80"],
            "detection_confidence": 0.92,
            "bbox_aspect": 0.5,
            "bbox_area_ratio": 0.04,
            "visible_keypoints": 17,
            "mean_keypoint_confidence": 0.9,
        }
        for index in range(48)
    ]
    summary = {
        "frames_seen": 12,
        "frames_with_pose": 10,
        "feature_names": ["torso_angle_mean", "hip_speed_max"],
        "feature_vector": [41.5, 0.82],
        "reason_counts": {},
        "frame_records": frame_records,
    }

    status = verifier._write_inline_feature_capture(
        "camera-1",
        summary,
        window_seconds=3.0,
    )

    record = json.loads(capture_path.read_text(encoding="utf-8"))
    assert status == "written"
    assert record["schema_version"] == 2
    assert record["runtime"] == "deepstream_pose_inline"
    assert record["camera_id"] == "camera-1"
    assert record["window_seconds"] == 3.0
    assert record["frames_seen"] == 12
    assert record["frames_with_pose"] == 10
    assert record["sampled_frames"] == 48
    assert record["frame_feature_names"] == list(FRAME_FEATURE_NAMES)
    assert record["frame_records"] == frame_records
    assert (
        record["frame_records"][0]["timestamp"]
        <= record["frame_records"][-1]["timestamp"]
    )
    assert record["feature_names"] == [
        "torso_angle_mean",
        "hip_speed_max",
    ]
    assert record["feature_vector"] == [41.5, 0.82]


def test_inline_feature_capture_is_noop_when_disabled(tmp_path) -> None:
    capture_path = tmp_path / "inline-features.jsonl"
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            inline_pose_rf=True,
            inline_feature_capture_path=None,
        )
    )

    status = verifier._write_inline_feature_capture(
        "camera-1",
        {
            "frames_seen": 1,
            "frames_with_pose": 1,
            "feature_names": ["feature"],
            "feature_vector": [1.0],
            "frame_records": [],
        },
        window_seconds=3.0,
    )

    assert status is None
    assert not capture_path.exists()


def test_inline_feature_capture_failure_is_fail_open(tmp_path) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            inline_pose_rf=True,
            inline_feature_capture_path=tmp_path,
        )
    )

    status = verifier._write_inline_feature_capture(
        "camera-1",
        {
            "frames_seen": 1,
            "frames_with_pose": 1,
            "feature_names": ["feature"],
            "feature_vector": [1.0],
            "frame_records": [],
        },
        window_seconds=3.0,
    )

    assert status == "error"


def test_add_pose_events_keeps_one_highest_fall_score_record_per_frame() -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, inline_pose_rf=True)
    )
    lower_score_event = _pose_event()
    higher_score_event = _pose_event()
    higher_score_event.metadata["fall_score"] = 4.5

    verifier.add_pose_events(
        "cam01",
        [lower_score_event, higher_score_event],
    )

    records = list(verifier._pose_records["cam01"])
    assert len(records) == 1
    assert records[0]["frame_index"] == 42
    assert records[0]["fall_score"] == 4.5


def test_add_pose_events_deduplicates_across_calls_and_resets_on_frame_rewind() -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, inline_pose_rf=True)
    )
    lower_score_event = _pose_event()
    higher_score_event = _pose_event()
    higher_score_event.metadata["fall_score"] = 4.5
    rewound_event = _pose_event()
    rewound_event.metadata["frame_num"] = 0

    verifier.add_pose_events("cam01", [lower_score_event])
    verifier.add_pose_events("cam01", [higher_score_event])

    records = list(verifier._pose_records["cam01"])
    assert len(records) == 1
    assert records[0]["fall_score"] == 4.5

    verifier.add_pose_events("cam01", [rewound_event])

    records = list(verifier._pose_records["cam01"])
    assert len(records) == 1
    assert records[0]["frame_index"] == 0


def test_inline_pose_rf_uses_camera_pose_records_without_subprocess(
    monkeypatch,
    tmp_path,
) -> None:
    capture_path = tmp_path / "inline-features.jsonl"
    bundle = {
        "model": _FixedInlineClassifier(),
        "feature_names": FEATURE_NAMES,
        "fall_class_label": 1,
        "inference_config": {
            "max_frames": 48,
            "candidate_window_seconds": 3.0,
        },
        "training_config": {
            "min_pose_frames": 1,
            "decision_threshold": 0.7,
        },
    }
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            inline_pose_rf=True,
            cooldown_seconds=0,
            inline_feature_capture_path=capture_path,
        ),
        inline_pose_rf_bundle=bundle,
    )
    verifier.add_pose_events("cam01", [_pose_event()])
    verifier.add_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
    monkeypatch.setattr(
        verifier,
        "_run",
        lambda _command: (_ for _ in ()).throw(
            AssertionError("inline pose RF must not launch subprocesses")
        ),
    )

    result = verifier.verify(camera_name="cam01")
    other_camera_result = verifier.verify(camera_name="cam02")

    assert result["status"] == "ok"
    assert result["confirmed"] is True
    assert result["fall_probability"] == 0.92
    assert result["runtime"] == "deepstream_pose_inline"
    assert result["feature_capture_status"] == "written"
    captured_record = json.loads(capture_path.read_text(encoding="utf-8"))
    assert captured_record["feature_names"] == FEATURE_NAMES
    assert len(captured_record["feature_vector"]) == len(FEATURE_NAMES)
    assert {
        frame_record["frame_index"]
        for frame_record in captured_record["frame_records"]
    } == {42}
    assert verifier.snapshot_frames() == []
    assert other_camera_result["status"] == "no_pose_records"


def test_shadow_mode_keeps_event_and_adds_metadata(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="shadow", cooldown_seconds=0)
    )
    monkeypatch.setattr(
        verifier,
        "verify",
        lambda: {
            "enabled": True,
            "mode": "shadow",
            "status": "ok",
            "confirmed": False,
            "fall_probability": 0.2,
        },
    )

    [event] = verifier.annotate_events([_fall_event()])

    assert event.metadata["falldata_aux"]["status"] == "ok"
    assert event.metadata["falldata_aux"]["confirmed"] is False


def test_confirm_mode_drops_unconfirmed_fall(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="confirm", cooldown_seconds=0)
    )
    monkeypatch.setattr(
        verifier,
        "verify",
        lambda: {
            "enabled": True,
            "mode": "confirm",
            "status": "ok",
            "confirmed": False,
        },
    )

    assert verifier.annotate_events([_fall_event()]) == []


def test_confirm_mode_fail_opens_when_aux_is_unavailable(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="confirm", cooldown_seconds=0)
    )
    monkeypatch.setattr(
        verifier,
        "verify",
        lambda: {
            "enabled": True,
            "mode": "confirm",
            "status": "missing_dependency",
            "confirmed": False,
            "missing": ".venv-falldata/bin/python",
        },
    )

    [event] = verifier.annotate_events([_fall_event()])

    assert event.metadata["falldata_aux"]["status"] == "missing_dependency"
    assert event.metadata["falldata_aux_confirm_fallback"] == "missing_dependency"


def test_confirm_mode_can_disable_fail_open(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            mode="confirm",
            cooldown_seconds=0,
            fail_open_on_unavailable=False,
        )
    )
    monkeypatch.setattr(
        verifier,
        "verify",
        lambda: {
            "enabled": True,
            "mode": "confirm",
            "status": "error",
            "confirmed": False,
        },
    )

    assert verifier.annotate_events([_fall_event()]) == []


def test_shadow_mode_keeps_event_but_error_is_not_confirmed(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="shadow", cooldown_seconds=0)
    )
    monkeypatch.setattr(
        verifier,
        "_verify_once",
        lambda: (_ for _ in ()).throw(RuntimeError("timeout")),
    )

    [event] = verifier.annotate_events([_fall_event()])

    assert event.metadata["falldata_aux"]["status"] == "error"
    assert event.metadata["falldata_aux"]["confirmed"] is False


def test_verify_passes_max_extract_frames_to_mediapipe(monkeypatch, tmp_path) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            max_extract_frames=42,
            sequence_transform="stretch",
            cooldown_seconds=0,
            mediapipe_python=tmp_path / "mediapipe-python",
            model_python=tmp_path / "model-python",
            model_path=tmp_path / "model.pkl",
        )
    )
    for path in (
        verifier.config.mediapipe_python,
        verifier.config.model_python,
        verifier.config.model_path,
    ):
        path.write_text("", encoding="utf-8")
    verifier.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    commands = []

    def fake_run(command):
        commands.append(command)
        stdout = (
            "nonzero_feature_frames: 42\n"
            if "--output-dir" in command
            else "prediction: [0]\npredict_proba: [[0.91, 0.09]]\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(verifier, "_run", fake_run)

    verifier.verify()

    extract_command = commands[0]
    max_frames_index = extract_command.index("--max-frames")
    assert extract_command[max_frames_index + 1] == "42"
    transform_index = extract_command.index("--sequence-transform")
    assert extract_command[transform_index + 1] == "stretch"


def test_run_isolates_subprocess_pythonpath(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, timeout_seconds=5)
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    verifier._run(["python", "script.py"])

    assert captured["env"]["PYTHONPATH"] == str(PROJECT_ROOT)


def test_verify_records_compare_model_result(monkeypatch, tmp_path) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            cooldown_seconds=0,
            threshold=0.7,
            mediapipe_python=tmp_path / "mediapipe-python",
            model_python=tmp_path / "model-python",
            model_path=tmp_path / "baseline.pkl",
            compare_model_path=tmp_path / "candidate.pkl",
        )
    )
    for path in (
        verifier.config.mediapipe_python,
        verifier.config.model_python,
        verifier.config.model_path,
        verifier.config.compare_model_path,
    ):
        path.write_text("", encoding="utf-8")
    verifier.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))

    def fake_run(command):
        if "--output-dir" in command:
            stdout = "nonzero_feature_frames: 42\n"
        elif str(verifier.config.compare_model_path) in command:
            stdout = "prediction: [1]\npredict_proba: [[0.12, 0.88]]\n"
        else:
            stdout = "prediction: [0]\npredict_proba: [[0.91, 0.09]]\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(verifier, "_run", fake_run)

    result = verifier.verify()

    assert result["status"] == "ok"
    assert result["confirmed"] is True
    assert result["compare_model"]["status"] == "ok"
    assert result["compare_model"]["confirmed"] is False
    assert result["compare_model"]["prediction"] == 1
    assert result["compare_model"]["fall_probability"] == 0.12


def test_compare_model_can_use_its_own_threshold(monkeypatch, tmp_path) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            cooldown_seconds=0,
            threshold=0.7,
            compare_threshold=0.5,
            compare_fall_class_index=1,
            mediapipe_python=tmp_path / "mediapipe-python",
            model_python=tmp_path / "model-python",
            model_path=tmp_path / "baseline.pkl",
            compare_model_path=tmp_path / "candidate.pkl",
        )
    )
    for path in (
        verifier.config.mediapipe_python,
        verifier.config.model_python,
        verifier.config.model_path,
        verifier.config.compare_model_path,
    ):
        path.write_text("", encoding="utf-8")
    verifier.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))

    def fake_run(command):
        if "--output-dir" in command:
            stdout = "nonzero_feature_frames: 42\n"
        elif str(verifier.config.compare_model_path) in command:
            stdout = "prediction: [1]\npredict_proba: [[0.4, 0.6]]\n"
        else:
            stdout = "prediction: [0]\npredict_proba: [[0.91, 0.09]]\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(verifier, "_run", fake_run)

    result = verifier.verify()

    assert result["compare_model"]["confirmed"] is True
    assert result["compare_model"]["threshold"] == 0.5


def test_yolo_pose_compare_uses_named_fall_probability(monkeypatch, tmp_path) -> None:
    compare_model = tmp_path / "candidate.pkl"
    compare_model.write_text("", encoding="utf-8")
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            compare_model_kind="yolo_pose_rf",
            compare_model_path=compare_model,
            compare_fall_class_index=0,
            compare_threshold=0.7,
        )
    )
    monkeypatch.setattr(
        verifier,
        "_run",
        lambda command: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "prediction: [1]\n"
                "predict_proba: [[0.95, 0.05]]\n"
                "fall_probability: 0.82\n"
                "frames_with_pose: 42\n"
            ),
            stderr="",
        ),
    )

    result = verifier._run_compare_model(
        tmp_path / "unused-features",
        nonzero_frames=42,
        video_path=tmp_path / "candidate.mp4",
    )

    assert result["fall_probability"] == 0.82
    assert result["confirmed"] is True


def test_missing_compare_model_does_not_block_primary_result(monkeypatch, tmp_path) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            cooldown_seconds=0,
            mediapipe_python=tmp_path / "mediapipe-python",
            model_python=tmp_path / "model-python",
            model_path=tmp_path / "baseline.pkl",
            compare_model_path=tmp_path / "missing-candidate.pkl",
        )
    )
    for path in (
        verifier.config.mediapipe_python,
        verifier.config.model_python,
        verifier.config.model_path,
    ):
        path.write_text("", encoding="utf-8")
    verifier.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))

    def fake_run(command):
        stdout = (
            "nonzero_feature_frames: 42\n"
            if "--output-dir" in command
            else "prediction: [0]\npredict_proba: [[0.91, 0.09]]\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(verifier, "_run", fake_run)

    result = verifier.verify()

    assert result["status"] == "ok"
    assert result["confirmed"] is True
    assert result["compare_model"]["status"] == "missing_dependency"
    assert result["compare_model"]["confirmed"] is False


def test_verify_records_temporal_compare_model_result(monkeypatch, tmp_path) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            cooldown_seconds=0,
            mediapipe_python=tmp_path / "mediapipe-python",
            model_python=tmp_path / "model-python",
            model_path=tmp_path / "baseline.pkl",
            temporal_python=tmp_path / "temporal-python",
            temporal_compare_model_path=tmp_path / "candidate.pt",
            temporal_pose_model_path=tmp_path / "pose.pt",
        )
    )
    for path in (
        verifier.config.mediapipe_python,
        verifier.config.model_python,
        verifier.config.model_path,
        verifier.config.temporal_python,
        verifier.config.temporal_compare_model_path,
        verifier.config.temporal_pose_model_path,
    ):
        path.write_text("", encoding="utf-8")
    verifier.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    commands = []

    def fake_run(command):
        commands.append(command)
        if "--output-dir" in command:
            stdout = "nonzero_feature_frames: 42\n"
        elif str(verifier.config.temporal_compare_model_path) in command:
            stdout = (
                "prediction: [1]\n"
                "fall_probability: 0.91\n"
                "threshold: 0.6\n"
                "frames_with_pose: 28\n"
            )
        else:
            stdout = "prediction: [0]\npredict_proba: [[0.91, 0.09]]\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(verifier, "_run", fake_run)

    result = verifier.verify()

    assert result["status"] == "ok"
    assert result["temporal_compare_model"] == {
        "status": "ok",
        "model_path": str(verifier.config.temporal_compare_model_path),
        "confirmed": True,
        "prediction": 1,
        "fall_probability": 0.91,
        "threshold": 0.6,
        "frames_with_pose": 28,
    }
    temporal_command = next(
        command
        for command in commands
        if str(verifier.config.temporal_compare_model_path) in command
    )
    assert "--video" in temporal_command
    assert str(verifier.config.temporal_pose_model_path) in temporal_command


def test_temporal_compare_model_can_use_sliding_windows(monkeypatch, tmp_path) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            cooldown_seconds=0,
            mediapipe_python=tmp_path / "mediapipe-python",
            model_python=tmp_path / "model-python",
            model_path=tmp_path / "baseline.pkl",
            temporal_python=tmp_path / "temporal-python",
            temporal_compare_model_path=tmp_path / "candidate.pt",
            temporal_pose_model_path=tmp_path / "pose.pt",
            temporal_sliding_window_size=12,
            temporal_sliding_window_stride=4,
            temporal_min_confirmed_windows=3,
        )
    )
    for path in (
        verifier.config.mediapipe_python,
        verifier.config.model_python,
        verifier.config.model_path,
        verifier.config.temporal_python,
        verifier.config.temporal_compare_model_path,
        verifier.config.temporal_pose_model_path,
    ):
        path.write_text("", encoding="utf-8")
    verifier.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    commands = []

    def fake_run(command):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="prediction: [1]\nfall_probability: 0.8\nthreshold: 0.6\nframes_with_pose: 28\n",
            stderr="",
        )

    monkeypatch.setattr(verifier, "_run", fake_run)
    verifier.verify()

    temporal_command = next(
        command
        for command in commands
        if str(verifier.config.temporal_compare_model_path) in command
    )
    assert temporal_command[-6:] == [
        "--sliding-window-size",
        "12",
        "--sliding-window-stride",
        "4",
        "--min-confirmed-windows",
        "3",
    ]


def test_cooldown_without_previous_result_is_not_confirmed(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="shadow", cooldown_seconds=60)
    )
    monkeypatch.setattr("src.core.ai._falldata_aux.time.time", lambda: 100.0)
    verifier._last_run_at = 99.0

    result = verifier.verify()

    assert result["status"] == "skipped_cooldown"
    assert result["confirmed"] is False


def test_cooldown_reuses_previous_result_but_marks_status(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="shadow", cooldown_seconds=60)
    )
    monkeypatch.setattr("src.core.ai._falldata_aux.time.time", lambda: 100.0)
    verifier._last_run_at = 99.0
    verifier._last_result = {
        "enabled": True,
        "mode": "shadow",
        "status": "ok",
        "confirmed": True,
        "fall_probability": 0.91,
    }

    result = verifier.verify()

    assert result["status"] == "skipped_cooldown"
    assert result["previous_status"] == "ok"
    assert result["confirmed"] is True
    assert result["fall_probability"] == 0.91


def test_parse_smoke_outputs() -> None:
    output = """
    nonzero_feature_frames: 40
    prediction: [0]
    predict_proba: [[0.9185, 0.0815]]
    """

    assert FallDataAuxVerifier._parse_nonzero_frames(output) == 40
    assert FallDataAuxVerifier._parse_prediction(output) == 0
    assert FallDataAuxVerifier._parse_probability(output) == [0.9185, 0.0815]
