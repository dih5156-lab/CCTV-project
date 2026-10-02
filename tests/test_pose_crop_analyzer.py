"""AIAnalyzer crop-pose POC의 CPU 계약 테스트."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.core.ai.analyzer import AIAnalyzer
from src.core.events import DetectionEvent, EventType


class _Keypoints:
    def __init__(self, values: np.ndarray):
        self.data = [values]


class _PoseResult:
    def __init__(self, values: np.ndarray):
        self.keypoints = _Keypoints(values)


def _keypoints() -> np.ndarray:
    values = np.zeros((17, 3), dtype=np.float32)
    values[:, 2] = 0.9
    values[0, :2] = (10, 20)
    values[5, :2] = (8, 30)
    values[6, :2] = (12, 30)
    values[11, :2] = (8, 50)
    values[12, :2] = (12, 50)
    return values


def test_run_pose_crop_restores_coordinates_and_emits_fall_event():
    with patch.object(AIAnalyzer, "load_models"):
        analyzer = AIAnalyzer(confidence_threshold=0.5)
    analyzer._pose_diag_enabled = False
    analyzer._record_pose_diagnostic = MagicMock()

    analyzer.person_model = object()
    analyzer.pose_model = MagicMock()
    analyzer._run_single_model = MagicMock(
        return_value=[
            DetectionEvent(
                event_type=EventType.PERSON,
                x=20,
                y=30,
                width=40,
                height=80,
                confidence=0.9,
                timestamp=1.0,
                object_id=17,
            )
        ]
    )
    analyzer._fall.detect_keypoints = MagicMock(return_value=True)
    analyzer.pose_model.predict.return_value = [_PoseResult(_keypoints())]

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    people, falls = analyzer._run_pose_crop(frame)

    assert len(people) == 1
    assert len(falls) == 1
    assert people[0].object_id == 17
    assert falls[0].event_type is EventType.FALL_DETECTED
    # crop origin is clipped to (14, 18), so (10, 20) -> (24, 38)
    assert people[0].keypoints[0][:2] == [24.0, 38.0]
    analyzer.pose_model.predict.assert_called_once()
    analyzer._record_pose_diagnostic.assert_called_once()
    assert analyzer._record_pose_diagnostic.call_args.kwargs["stage"] == "crop"


def test_run_pose_crop_returns_empty_events_when_pose_inference_fails():
    with patch.object(AIAnalyzer, "load_models"):
        analyzer = AIAnalyzer(confidence_threshold=0.5)
    analyzer._pose_diag_enabled = False

    analyzer.person_model = object()
    analyzer.pose_model = MagicMock()
    analyzer._run_single_model = MagicMock(
        return_value=[
            DetectionEvent(
                event_type=EventType.PERSON,
                x=10,
                y=10,
                width=20,
                height=40,
                confidence=0.8,
                timestamp=1.0,
                object_id=3,
            )
        ]
    )
    analyzer.pose_model.predict.side_effect = RuntimeError("synthetic inference failure")

    people, falls = analyzer._run_pose_crop(np.zeros((80, 100, 3), dtype=np.uint8))

    assert people == []
    assert falls == []


def test_crop_pose_diagnostic_writes_expected_jsonl_fields(tmp_path):
    with patch.object(AIAnalyzer, "load_models"):
        analyzer = AIAnalyzer(confidence_threshold=0.5)
    analyzer._pose_diag_enabled = True
    analyzer._pose_diag_path = tmp_path / "pose.jsonl"
    event = DetectionEvent(
        event_type=EventType.PERSON,
        x=1,
        y=2,
        width=10,
        height=20,
        confidence=0.9,
        timestamp=1.0,
        keypoints=[[3.0, 4.0, 0.8]],
    )

    analyzer._record_pose_diagnostic(
        stage="crop",
        imgsz=640,
        conf_threshold=0.5,
        elapsed_ms=12.3,
        person_events=[event],
    )

    import json

    record = json.loads(analyzer._pose_diag_path.read_text(encoding="utf-8"))
    assert record["stage"] == "crop"
    assert record["person_count"] == 1
    assert record["keypoint_count"] == 1
    assert record["keypoint_confidence_avg"] == 0.8
    assert record["elapsed_ms"] == 12.3


def test_run_pose_crop_records_diagnostic_when_no_person_model():
    with patch.object(AIAnalyzer, "load_models"):
        analyzer = AIAnalyzer(confidence_threshold=0.5)
    analyzer._pose_diag_enabled = False
    analyzer._record_pose_diagnostic = MagicMock()

    people, falls = analyzer._run_pose_crop(np.zeros((40, 60, 3), dtype=np.uint8))

    assert people == []
    assert falls == []
    analyzer._record_pose_diagnostic.assert_called_once()
    assert analyzer._record_pose_diagnostic.call_args.kwargs["stage"] == "crop"


def test_run_pose_crop_deduplicates_overlapping_person_events():
    with patch.object(AIAnalyzer, "load_models"):
        analyzer = AIAnalyzer(confidence_threshold=0.5)
    analyzer._pose_diag_enabled = False
    analyzer.person_model = object()
    analyzer.pose_model = MagicMock()
    duplicate = DetectionEvent(
        event_type=EventType.PERSON,
        x=20,
        y=20,
        width=40,
        height=80,
        confidence=0.9,
        timestamp=1.0,
        object_id=9,
    )
    analyzer._run_single_model = MagicMock(return_value=[duplicate, duplicate])
    analyzer._fall.detect_keypoints = MagicMock(return_value=True)
    analyzer.pose_model.predict.return_value = [_PoseResult(_keypoints()), _PoseResult(_keypoints())]

    people, falls = analyzer._run_pose_crop(np.zeros((120, 160, 3), dtype=np.uint8))

    assert len(people) == 1
    assert len(falls) == 1


def test_load_models_loads_person_model_only_when_crop_mode_enabled(monkeypatch):
    with patch.object(AIAnalyzer, "load_models"):
        analyzer = AIAnalyzer(
            helmet_model_path="helmet.pt",
            person_model_path="person.pt",
            pose_model_path="pose.pt",
        )

    analyzer.helmet_model = object()
    analyzer.pose_model = object()
    calls = []

    def fake_try_load(name, path):
        calls.append(name)

    with patch.object(analyzer, "_try_load", side_effect=fake_try_load):
        monkeypatch.setenv("POSE_CROP_MODE", "1")
        analyzer.load_models()
    assert calls == ["helmet", "pose", "person"]

    calls.clear()
    with patch.object(analyzer, "_try_load", side_effect=fake_try_load):
        monkeypatch.setenv("POSE_CROP_MODE", "0")
        analyzer.load_models()
    assert calls == ["helmet", "pose"]
