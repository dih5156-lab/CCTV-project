"""Lightweight temporal fall model shared by training and runtime inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

REASON_FEATURE_NAMES = (
    "torso_horizontal",
    "leg_above_head",
    "wide_bbox_low_head",
    "wide_bbox_candidate",
    "low_vertical_span",
    "torso_flattened",
    "missing_leg",
    "missing_shoulder",
    "folded_floor_pose",
)

FRAME_FEATURE_NAMES = (
    "fall_score",
    "detection_confidence",
    "bbox_aspect",
    "bbox_area_ratio",
    "visible_keypoints",
    "mean_keypoint_confidence",
    *REASON_FEATURE_NAMES,
)


def _reason_key(reason: str) -> str:
    return str(reason).split(":", 1)[0]


def encode_frame_sequence(
    frame_records: list[dict[str, Any]],
    *,
    sequence_length: int,
) -> np.ndarray:
    """Convert pose frame records to a fixed-length, left-padded sequence."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    sequence = np.zeros(
        (sequence_length, len(FRAME_FEATURE_NAMES)),
        dtype=np.float32,
    )
    selected_records = sorted(
        frame_records,
        key=lambda record: int(record.get("frame_index", 0)),
    )[-sequence_length:]
    start = sequence_length - len(selected_records)
    for offset, record in enumerate(selected_records):
        reasons = {_reason_key(reason) for reason in record.get("fall_reasons", [])}
        values = {
            "fall_score": min(max(float(record.get("fall_score", 0.0)) / 5.0, 0.0), 1.0),
            "detection_confidence": float(record.get("detection_confidence", 0.0)),
            "bbox_aspect": min(max(float(record.get("bbox_aspect", 0.0)) / 3.0, 0.0), 1.0),
            "bbox_area_ratio": min(max(float(record.get("bbox_area_ratio", 0.0)), 0.0), 1.0),
            "visible_keypoints": min(
                max(float(record.get("visible_keypoints", 0.0)) / 17.0, 0.0),
                1.0,
            ),
            "mean_keypoint_confidence": float(
                record.get("mean_keypoint_confidence", 0.0)
            ),
            **{name: float(name in reasons) for name in REASON_FEATURE_NAMES},
        }
        sequence[start + offset] = [values[name] for name in FRAME_FEATURE_NAMES]
    return sequence


class _TemporalEncoder(nn.Module):
    def __init__(self, *, input_features: int, channels: int = 32) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_features, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        features = self.temporal(sequence.transpose(1, 2))
        return torch.cat(
            [features.mean(dim=2), features.amax(dim=2)],
            dim=1,
        )


class FallTemporalTCN(nn.Module):
    """Small dilated Conv1D classifier suitable for ONNX/TensorRT export."""

    def __init__(self, *, input_features: int, channels: int = 32) -> None:
        super().__init__()
        self.encoder = _TemporalEncoder(
            input_features=input_features,
            channels=channels,
        )
        self.classifier = nn.Linear(channels * 2, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(sequence)).squeeze(1)


class FallTemporalHybrid(nn.Module):
    """Temporal encoder augmented with standardized video summary features."""

    def __init__(
        self,
        *,
        input_features: int,
        summary_features: int,
        channels: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = _TemporalEncoder(
            input_features=input_features,
            channels=channels,
        )
        self.summary_encoder = nn.Sequential(
            nn.Linear(summary_features, channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(channels * 3, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        summary: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            [self.encoder(sequence), self.summary_encoder(summary)],
            dim=1,
        )
        return self.classifier(combined).squeeze(1)
