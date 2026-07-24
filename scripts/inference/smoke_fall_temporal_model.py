#!/usr/bin/env python3
"""Run one fall temporal checkpoint against a buffered candidate video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.train_yolo_pose_fall_rf import (  # noqa: E402
    _extract_video_features,
    _load_pose_model,
)
from src.core.ai._fall_detector import FallDetector  # noqa: E402
from src.core.ai.fall_temporal_model import (  # noqa: E402
    FallTemporalHybrid,
    FallTemporalTCN,
    encode_frame_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model_type = str(checkpoint.get("model_type") or "")
    if model_type == "fall_temporal_hybrid":
        temporal_model = FallTemporalHybrid(
            input_features=int(checkpoint["input_features"]),
            summary_features=int(checkpoint["summary_features"]),
            channels=int(checkpoint["channels"]),
        )
    elif model_type == "fall_temporal_temporal":
        temporal_model = FallTemporalTCN(
            input_features=int(checkpoint["input_features"]),
            channels=int(checkpoint["channels"]),
        )
    else:
        raise SystemExit(f"unsupported model_type: {model_type}")
    temporal_model.load_state_dict(checkpoint["state_dict"])
    temporal_model.to(device).eval()

    summary = _extract_video_features(
        model=_load_pose_model(args.pose_model),
        detector=FallDetector(),
        video_path=args.video,
        max_frames=int(checkpoint["sequence_length"]),
        frame_stride=1,
        imgsz=args.imgsz,
        confidence_threshold=args.confidence_threshold,
    )
    sequence = encode_frame_sequence(
        list(summary.get("frame_records") or []),
        sequence_length=int(checkpoint["sequence_length"]),
    )
    sequence_tensor = torch.from_numpy(sequence[None]).to(device)
    with torch.inference_mode():
        if model_type == "fall_temporal_hybrid":
            summary_vector = np.asarray(summary["feature_vector"], dtype=np.float32)
            summary_mean = np.asarray(checkpoint["summary_mean"], dtype=np.float32)
            summary_scale = np.asarray(checkpoint["summary_scale"], dtype=np.float32)
            normalized_summary = (summary_vector - summary_mean) / summary_scale
            logits = temporal_model(
                sequence_tensor,
                torch.from_numpy(normalized_summary[None]).to(device),
            )
        else:
            logits = temporal_model(sequence_tensor)
        probability = float(torch.sigmoid(logits)[0].item())
    threshold = float(checkpoint["decision_threshold"])
    prediction = int(probability >= threshold)
    print(f"prediction: [{prediction}]")
    print(f"fall_probability: {probability:.8f}")
    print(f"threshold: {threshold:.8f}")
    print(f"frames_with_pose: {int(summary.get('frames_with_pose') or 0)}")
    print(f"device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
