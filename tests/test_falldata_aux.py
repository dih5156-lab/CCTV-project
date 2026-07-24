import subprocess

import numpy as np

from src.core.ai._falldata_aux import FallDataAuxConfig, FallDataAuxVerifier
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


def test_disabled_verifier_keeps_events_unchanged() -> None:
    verifier = FallDataAuxVerifier(FallDataAuxConfig(enabled=False))
    event = _fall_event()

    assert verifier.annotate_events([event]) == [event]
    verifier.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    assert verifier.annotate_events([event]) == [event]


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
