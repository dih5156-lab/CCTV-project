from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.datasets.train_deepstream_pose_fall_tcn import (
    assert_validation_disjoint,
    load_temporal_capture_datasets,
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
