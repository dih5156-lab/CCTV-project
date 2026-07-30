from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.datasets.train_deepstream_pose_fall_tcn import (
    TemporalCaptureDataset,
    TrainingConfig,
    _aggregate_scene_probabilities,
    assert_validation_disjoint,
    load_temporal_capture_datasets,
    select_threshold,
    train_candidate,
)
from src.core.ai.fall_temporal_model import FRAME_FEATURE_NAMES


def _frame_record(index: int) -> dict:
    return {
        "timestamp": float(index),
        "fall_score": 3.5,
        "fall_reasons": ["torso_horizontal:0.80"],
        "detection_confidence": 0.92,
        "bbox_aspect": 0.5,
        "bbox_area_ratio": 0.04,
        "visible_keypoints": 17,
        "mean_keypoint_confidence": 0.9,
    }


def _capture_row(
    *,
    scene_id: str,
    group_id: str,
    label: int,
) -> dict:
    return {
        "schema_version": 2,
        "runtime": "deepstream_pose_inline",
        "scene_id": scene_id,
        "group_id": group_id,
        "label": label,
        "is_fall": bool(label),
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "frame_records": [_frame_record(index) for index in range(48)],
        "scene_position": "복도",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_load_temporal_capture_datasets_encodes_fixed_sequences(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "captures.jsonl"
    _write_jsonl(
        dataset_path,
        [
            _capture_row(scene_id="fall-1", group_id="subject-1", label=1),
            _capture_row(scene_id="normal-1", group_id="subject-2", label=0),
        ],
    )

    dataset = load_temporal_capture_datasets([dataset_path])

    assert dataset.sequences.shape == (2, 48, len(FRAME_FEATURE_NAMES))
    assert dataset.sequences.dtype == np.float32
    np.testing.assert_array_equal(dataset.labels, [1, 0])
    assert dataset.scene_ids == ("fall-1", "normal-1")
    assert dataset.group_ids == ("subject-1", "subject-2")
    assert dataset.metadata[0]["scene_position"] == "복도"


def test_load_temporal_capture_datasets_rejects_feature_order_mismatch(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "captures.jsonl"
    fall_row = _capture_row(scene_id="fall-1", group_id="subject-1", label=1)
    fall_row["frame_feature_names"] = list(reversed(FRAME_FEATURE_NAMES))
    _write_jsonl(
        dataset_path,
        [
            fall_row,
            _capture_row(scene_id="normal-1", group_id="subject-2", label=0),
        ],
    )

    with pytest.raises(ValueError, match="frame feature order mismatch"):
        load_temporal_capture_datasets([dataset_path])


def test_load_temporal_capture_datasets_rejects_non_finite_sequence(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "captures.jsonl"
    fall_row = _capture_row(scene_id="fall-1", group_id="subject-1", label=1)
    fall_row["frame_records"][0]["detection_confidence"] = float("inf")
    _write_jsonl(
        dataset_path,
        [
            fall_row,
            _capture_row(scene_id="normal-1", group_id="subject-2", label=0),
        ],
    )

    with pytest.raises(ValueError, match="non-finite sequence"):
        load_temporal_capture_datasets([dataset_path])


def test_load_temporal_capture_datasets_rejects_empty_sequence(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "captures.jsonl"
    fall_row = _capture_row(scene_id="fall-1", group_id="subject-1", label=1)
    fall_row["frame_records"] = []
    _write_jsonl(
        dataset_path,
        [
            fall_row,
            _capture_row(scene_id="normal-1", group_id="subject-2", label=0),
        ],
    )

    with pytest.raises(ValueError, match="frame_records must be non-empty"):
        load_temporal_capture_datasets([dataset_path])


def test_load_temporal_capture_datasets_rejects_single_class(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "captures.jsonl"
    _write_jsonl(
        dataset_path,
        [
            _capture_row(scene_id="fall-1", group_id="subject-1", label=1),
            _capture_row(scene_id="fall-2", group_id="subject-2", label=1),
        ],
    )

    with pytest.raises(ValueError, match="both fall and non-fall"):
        load_temporal_capture_datasets([dataset_path])


@pytest.mark.parametrize(
    ("validation_scene", "validation_group", "expected_message"),
    [
        ("fall-1", "subject-3", "scene overlap.*fall-1"),
        ("fall-3", "subject-1", "group overlap.*subject-1"),
    ],
)
def test_assert_validation_disjoint_rejects_scene_or_group_overlap(
    tmp_path: Path,
    validation_scene: str,
    validation_group: str,
    expected_message: str,
) -> None:
    training_path = tmp_path / "training.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        training_path,
        [
            _capture_row(scene_id="fall-1", group_id="subject-1", label=1),
            _capture_row(scene_id="normal-1", group_id="subject-2", label=0),
        ],
    )
    _write_jsonl(
        validation_path,
        [
            _capture_row(
                scene_id=validation_scene,
                group_id=validation_group,
                label=1,
            ),
            _capture_row(scene_id="normal-3", group_id="subject-4", label=0),
        ],
    )

    training = load_temporal_capture_datasets([training_path])
    validation = load_temporal_capture_datasets([validation_path])

    with pytest.raises(ValueError, match=expected_message):
        assert_validation_disjoint(training, validation)


def test_select_threshold_uses_lowest_passing_value_at_or_above_point_seven() -> None:
    result = select_threshold(
        np.asarray([1, 1, 0, 0], dtype=np.int64),
        np.asarray([0.92, 0.71, 0.69, 0.20], dtype=np.float32),
    )

    assert result["passed"] is True
    assert result["decision_threshold"] == 0.70
    assert result["selected"]["fall_recall"] == 1.0
    assert result["selected"]["false_positive_rate"] == 0.0
    assert all(item["threshold"] >= 0.70 for item in result["sweep"])


def test_select_threshold_reports_failure_without_lowering_minimum() -> None:
    result = select_threshold(
        np.asarray([1, 1, 0, 0], dtype=np.int64),
        np.asarray([0.65, 0.60, 0.80, 0.10], dtype=np.float32),
    )

    assert result["passed"] is False
    assert result["decision_threshold"] == 0.70
    assert len(result["sweep"]) == 6


def test_aggregate_scene_probabilities_uses_max_window_per_scene() -> None:
    scene_ids, labels, probabilities = _aggregate_scene_probabilities(
        scene_ids=("fall-1", "fall-1", "normal-1"),
        labels=np.asarray([1, 1, 0], dtype=np.int64),
        probabilities=np.asarray([0.72, 0.91, 0.20], dtype=np.float32),
    )

    assert scene_ids == ("fall-1", "normal-1")
    np.testing.assert_array_equal(labels, [1, 0])
    np.testing.assert_allclose(probabilities, [0.91, 0.20])


def test_aggregate_scene_probabilities_rejects_mixed_scene_labels() -> None:
    with pytest.raises(ValueError, match="scene has mixed labels"):
        _aggregate_scene_probabilities(
            scene_ids=("scene-1", "scene-1"),
            labels=np.asarray([1, 0], dtype=np.int64),
            probabilities=np.asarray([0.8, 0.2], dtype=np.float32),
        )


def _synthetic_dataset(
    *,
    prefix: str,
    samples_per_class: int,
) -> TemporalCaptureDataset:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    scene_ids: list[str] = []
    group_ids: list[str] = []
    metadata: list[dict] = []
    for label in (1, 0):
        for index in range(samples_per_class):
            sequence = np.zeros(
                (48, len(FRAME_FEATURE_NAMES)),
                dtype=np.float32,
            )
            sequence[:, 0] = 0.9 if label else 0.1
            sequence[:, 2] = 0.8 if label else 0.2
            scene_id = f"{prefix}-{'fall' if label else 'normal'}-{index}"
            group_id = f"{prefix}-group-{'fall' if label else 'normal'}-{index}"
            sequences.append(sequence)
            labels.append(label)
            scene_ids.append(scene_id)
            group_ids.append(group_id)
            metadata.append({"scene_id": scene_id, "group_id": group_id})
    return TemporalCaptureDataset(
        sequences=np.stack(sequences),
        labels=np.asarray(labels, dtype=np.int64),
        scene_ids=tuple(scene_ids),
        group_ids=tuple(group_ids),
        metadata=tuple(metadata),
    )


def test_train_candidate_builds_temporal_only_checkpoint_contract() -> None:
    training = _synthetic_dataset(prefix="train", samples_per_class=4)
    validation = _synthetic_dataset(prefix="validation", samples_per_class=2)

    checkpoint, metrics = train_candidate(
        training,
        validation,
        TrainingConfig(
            epochs=2,
            patience=2,
            batch_size=4,
            channels=4,
            device="cpu",
            random_state=7,
        ),
    )

    assert checkpoint["format_version"] == 2
    assert checkpoint["model_type"] == "deepstream_pose_temporal_tcn"
    assert checkpoint["sequence_length"] == 48
    assert checkpoint["frame_feature_names"] == list(FRAME_FEATURE_NAMES)
    assert checkpoint["decision_threshold"] >= 0.70
    assert checkpoint["channels"] == 4
    assert "state_dict" in checkpoint
    assert "split_hash" in checkpoint
    assert metrics["training"]["epochs_completed"] == 2
    assert metrics["validation"]["fall_support"] == 2
    assert metrics["validation"]["normal_support"] == 2
