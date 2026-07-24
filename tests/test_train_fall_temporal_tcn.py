from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "datasets"
    / "train_fall_temporal_tcn.py"
)

spec = importlib.util.spec_from_file_location("train_fall_temporal_tcn", SCRIPT_PATH)
train_fall_temporal_tcn = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["train_fall_temporal_tcn"] = train_fall_temporal_tcn
spec.loader.exec_module(train_fall_temporal_tcn)


def test_binary_metrics_reports_fall_first_confusion_and_errors() -> None:
    metrics = train_fall_temporal_tcn._binary_metrics(
        labels=np.asarray([1, 1, 0, 0]),
        probabilities=np.asarray([0.9, 0.4, 0.8, 0.2]),
        threshold=0.5,
        scene_ids=["fall_ok", "fall_missed", "normal_alarm", "normal_ok"],
    )

    assert metrics["confusion_matrix_labels"] == ["fall", "non_fall"]
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["classification_report"]["fall"]["precision"] == 0.5
    assert metrics["classification_report"]["fall"]["recall"] == 0.5
    assert metrics["errors"]["false_positive_count"] == 1
    assert metrics["errors"]["false_negative_count"] == 1
    assert metrics["errors"]["false_positives"][0]["scene_id"] == "normal_alarm"
    assert metrics["errors"]["false_negatives"][0]["scene_id"] == "fall_missed"


def test_fit_summary_standardizer_uses_safe_scale_for_constant_features() -> None:
    features = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 10.0],
        ],
        dtype=np.float32,
    )

    normalized, mean, scale = train_fall_temporal_tcn._fit_summary_standardizer(
        features
    )

    np.testing.assert_allclose(mean, [2.0, 10.0])
    np.testing.assert_allclose(scale, [1.0, 1.0])
    np.testing.assert_allclose(normalized[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(normalized[:, 1], [0.0, 0.0])


def test_select_cached_rows_keeps_only_rows_with_matching_feature_files(
    tmp_path: Path,
) -> None:
    rows = [
        {"scene_id": "scene_a", "video_path": "a.mp4"},
        {"scene_id": "scene_b", "video_path": "b.mp4"},
    ]
    (tmp_path / "scene_b_uniform_max30_stride6.json").write_text(
        "{}",
        encoding="utf-8",
    )

    selected = train_fall_temporal_tcn._select_cached_rows(
        rows,
        feature_cache=tmp_path,
        max_frames=30,
        frame_stride=6,
    )

    assert [row["scene_id"] for row in selected] == ["scene_b"]
