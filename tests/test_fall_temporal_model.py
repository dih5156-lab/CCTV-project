from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
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


def test_temporal_cli_pose_default_matches_recorded_evaluation_contract(monkeypatch):
    import sys

    from scripts.inference.smoke_fall_temporal_model import parse_args

    # 검증된 640 입력을 기본으로 사용하고, 대조 시험의 명시적 320 선택은 보존한다.
    base = ["temporal", "--model", "model.pt", "--pose-model", "pose.pt"]
    monkeypatch.setattr(sys, "argv", base)
    assert parse_args().imgsz == 640
    monkeypatch.setattr(sys, "argv", base + ["--imgsz", "320"])
    assert parse_args().imgsz == 320


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


def test_short_sequence_preserves_required_confirmation_count() -> None:
    assert _effective_min_confirmed_windows(2, 1) == 2


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


@pytest.mark.parametrize(
    "probabilities,sliding,required,status,prediction",
    [
        ([0.9], 2, 2, "insufficient_windows", 0),
        ([0.9, 0.9], 2, 2, "ok", 1),
        ([0.9, 0.1, 0.9], 2, 2, "ok", 0),
        ([], 0, 1, "insufficient_windows", 0),
        ([0.9], 0, 1, "ok", 1),
    ],
)
def test_temporal_cli_reports_completed_calls_and_preserves_minimum(
    monkeypatch, tmp_path, capsys, probabilities, sliding, required, status, prediction
):
    import sys

    from scripts.inference import smoke_fall_temporal_model as cli

    model = FallTemporalTCN(input_features=len(FRAME_FEATURE_NAMES), channels=4)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_type": "fall_temporal_temporal",
            "input_features": len(FRAME_FEATURE_NAMES),
            "channels": 4,
            "sequence_length": 2,
            "decision_threshold": 0.7,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    frame_count = (
        len(probabilities) + 1 if sliding and probabilities else len(probabilities)
    )
    summary = {
        "frame_records": [_frame(i, 1, []) for i in range(frame_count)],
        "frames_with_pose": frame_count,
        "frames_seen": frame_count,
    }
    features = tmp_path / "features.json"
    features.write_text(json.dumps(summary))
    exported = tmp_path / "exported.json"
    values = iter(probabilities)
    calls = []

    def predict(*args, **kwargs):
        calls.append(args[1])
        return next(values)

    monkeypatch.setattr(cli, "_predict_window", predict)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "temporal",
            "--model",
            str(checkpoint),
            "--pose-model",
            "unused.pt",
            "--feature-json",
            str(features),
            "--output-feature-json",
            str(exported),
            "--device",
            "cpu",
            "--sliding-window-size",
            str(sliding),
            "--sliding-window-stride",
            "1",
            "--min-confirmed-windows",
            str(required),
        ],
    )
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert f"window_inferences: {len(calls)}\n" in output
    assert len(calls) == len(probabilities)
    assert f"inference_status: {status}\n" in output
    assert f"prediction: [{prediction}]\n" in output
    if sliding:
        assert f"min_confirmed_windows: {required}\n" in output
    assert json.loads(exported.read_text()) == summary
