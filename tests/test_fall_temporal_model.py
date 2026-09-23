from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.inference.smoke_fall_temporal_model import (
    _effective_min_confirmed_windows,
    _required_frame_count,
    write_feature_summary,
)
from src.core.ai.fall_temporal_model import (
    FRAME_FEATURE_NAMES,
    FallTemporalHybrid,
    FallTemporalTCN,
    encode_frame_sequence,
)


def _frame(frame_index: int, fall_score: float, reasons: list[str]) -> dict:
    return {
        "frame_index": frame_index,
        "fall_score": fall_score,
        "fall_reasons": reasons,
        "detection_confidence": 0.8,
        "bbox_aspect": 1.2,
        "bbox_area_ratio": 0.25,
        "visible_keypoints": 12,
        "mean_keypoint_confidence": 0.7,
    }


def test_encode_frame_sequence_left_pads_and_preserves_temporal_order() -> None:
    sequence = encode_frame_sequence(
        [
            _frame(10, 1.0, []),
            _frame(20, 4.0, ["torso_horizontal"]),
        ],
        sequence_length=4,
    )

    assert sequence.shape == (4, len(FRAME_FEATURE_NAMES))
    np.testing.assert_array_equal(sequence[:2], 0.0)
    np.testing.assert_allclose(
        sequence[2:, FRAME_FEATURE_NAMES.index("fall_score")],
        [0.2, 0.8],
    )
    assert sequence[3, FRAME_FEATURE_NAMES.index("torso_horizontal")] == 1.0


def test_encode_frame_sequence_keeps_most_recent_frames_when_truncated() -> None:
    sequence = encode_frame_sequence(
        [_frame(index, float(index), []) for index in range(1, 6)],
        sequence_length=3,
    )

    np.testing.assert_allclose(
        sequence[:, FRAME_FEATURE_NAMES.index("fall_score")],
        [0.6, 0.8, 1.0],
    )


def test_required_frame_count_covers_consecutive_sliding_windows() -> None:
    assert _required_frame_count(60, 60, 20, 2) == 80
    assert _required_frame_count(60, 60, 20, 3) == 100
    assert _required_frame_count(60, 0, 20, 2) == 60


def test_short_sequence_does_not_reject_one_valid_window() -> None:
    assert _effective_min_confirmed_windows(2, 1) == 1


def test_write_feature_summary_persists_reusable_pose_features(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "scene.json"
    summary = {
        "frames_seen": 60,
        "frames_with_pose": 30,
        "frame_records": [{"frame_index": 10, "fall_score": 0.8}],
    }

    write_feature_summary(summary, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == summary


def test_fall_temporal_tcn_returns_one_logit_per_video() -> None:
    model = FallTemporalTCN(input_features=len(FRAME_FEATURE_NAMES), channels=8)
    batch = torch.randn(2, 30, len(FRAME_FEATURE_NAMES))

    logits = model(batch)

    assert logits.shape == (2,)


def test_fall_temporal_hybrid_combines_sequence_and_summary_features() -> None:
    model = FallTemporalHybrid(
        input_features=len(FRAME_FEATURE_NAMES),
        summary_features=37,
        channels=8,
    )
    sequence_batch = torch.randn(2, 30, len(FRAME_FEATURE_NAMES))
    summary_batch = torch.randn(2, 37)

    logits = model(sequence_batch, summary_batch)

    assert logits.shape == (2,)
