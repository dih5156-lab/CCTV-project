"""falldata RF 학습 스크립트의 데이터 분리 헬퍼 테스트."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "datasets"
    / "train_falldata_video_rf.py"
)

spec = importlib.util.spec_from_file_location("train_falldata_video_rf", SCRIPT_PATH)
train_falldata_video_rf = importlib.util.module_from_spec(spec)
assert spec and spec.loader
try:
    spec.loader.exec_module(train_falldata_video_rf)
except (AttributeError, ImportError) as exc:
    pytest.skip(f"falldata RF training dependencies unavailable: {exc}", allow_module_level=True)


def test_scene_base_group_strips_camera_suffix() -> None:
    row = {"scene_id": "00047_H_A_N_C8", "video_path": "unused.mp4"}

    assert train_falldata_video_rf._group_for_row(row, "scene_base") == "00047_H_A_N"


def test_group_holdout_keeps_scene_variants_together() -> None:
    rows = [
        {"scene_id": "not_fall_C1", "video_path": "unused.mp4"},
        {"scene_id": "not_fall_C2", "video_path": "unused.mp4"},
        {"scene_id": "fall_a_C1", "video_path": "unused.mp4"},
        {"scene_id": "fall_a_C2", "video_path": "unused.mp4"},
        {"scene_id": "fall_b_C1", "video_path": "unused.mp4"},
        {"scene_id": "fall_b_C2", "video_path": "unused.mp4"},
    ]
    groups = [train_falldata_video_rf._group_for_row(row, "scene_base") for row in rows]
    x = np.arange(len(rows) * 2, dtype=np.float32).reshape(len(rows), 2)
    y = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int64)
    row_ids = [row["scene_id"] for row in rows]
    args = types.SimpleNamespace(test_size=0.34, random_state=4, cv_group_by="scene_base")

    _, _, _, _, train_ids, test_ids, split_info = train_falldata_video_rf._train_test_split(
        x,
        y,
        row_ids,
        groups,
        args,
    )

    train_bases = {scene_id.rsplit("_C", 1)[0] for scene_id in train_ids}
    test_bases = {scene_id.rsplit("_C", 1)[0] for scene_id in test_ids}
    assert train_bases.isdisjoint(test_bases)
    assert split_info["method"] == "group_shuffle"
    assert split_info["group_by"] == "scene_base"
    assert "train_class_counts" in split_info
    assert "test_class_counts" in split_info


def test_prediction_error_summary_lists_false_positive_and_false_negative() -> None:
    evaluation = {
        "predictions": [
            {"scene_id": "fp", "true": 1, "predicted": 0},
            {"scene_id": "fn", "true": 0, "predicted": 1},
            {"scene_id": "tp", "true": 0, "predicted": 0},
        ]
    }

    summary = train_falldata_video_rf._prediction_error_summary(evaluation)

    assert summary["false_positive_count"] == 1
    assert summary["false_negative_count"] == 1
    assert summary["false_positives"][0]["scene_id"] == "fp"
    assert summary["false_negatives"][0]["scene_id"] == "fn"
