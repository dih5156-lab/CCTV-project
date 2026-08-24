#!/usr/bin/env python3
"""Run one fall temporal checkpoint against a buffered candidate video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.train_yolo_pose_fall_rf import (  # noqa: E402
    _extract_video_features,
    _load_pose_model,
    _summarize_frames,
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
    parser.add_argument("--video", type=Path)
    parser.add_argument("--feature-json", type=Path, help="Use a cached feature JSON instead of decoding video.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=0,
        help="Evaluate overlapping pose windows; 0 keeps the original full-video inference.",
    )
    parser.add_argument("--sliding-window-stride", type=int, default=5)
    parser.add_argument(
        "--min-confirmed-windows",
        type=int,
        default=1,
        help="Require this many consecutive windows at/above threshold for a sliding prediction.",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def _predict_window(
    temporal_model: torch.nn.Module,
    frame_records: list[dict],
    *,
    frames_seen: int,
    checkpoint: dict,
    model_type: str,
    device: torch.device,
) -> float:
    sequence = encode_frame_sequence(
        frame_records,
        sequence_length=int(checkpoint["sequence_length"]),
    )
    sequence_tensor = torch.from_numpy(sequence[None]).to(device)
    with torch.inference_mode():
        if model_type == "fall_temporal_hybrid":
            summary = _summarize_frames(frame_records, frames_seen)
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
    return float(torch.sigmoid(logits)[0].item())


def _required_frame_count(
    sequence_length: int,
    window_size: int,
    stride: int,
    min_confirmed_windows: int,
) -> int:
    """Return enough frames to evaluate the requested consecutive windows."""
    required_windows = max(min_confirmed_windows, 1)
    if window_size <= 0:
        return sequence_length
    return max(
        sequence_length,
        window_size + max(stride, 1) * (required_windows - 1),
    )


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

    if args.feature_json is not None:
        summary = json.loads(args.feature_json.read_text(encoding="utf-8"))
    else:
        if args.video is None:
            raise SystemExit("--video is required unless --feature-json is provided")
        max_frames = _required_frame_count(
            int(checkpoint["sequence_length"]),
            int(args.sliding_window_size),
            int(args.sliding_window_stride),
            int(args.min_confirmed_windows),
        )
        summary = _extract_video_features(
            model=_load_pose_model(args.pose_model),
            detector=FallDetector(),
            video_path=args.video,
            max_frames=max_frames,
            frame_stride=1,
            imgsz=args.imgsz,
            confidence_threshold=args.confidence_threshold,
        )
    frame_records = list(summary.get("frame_records") or [])
    if args.sliding_window_size > 0:
        window_size = max(args.sliding_window_size, 1)
        stride = max(args.sliding_window_stride, 1)
        windows = [
            frame_records[start : start + window_size]
            for start in range(0, max(len(frame_records) - window_size + 1, 1), stride)
        ]
        probabilities = [
            _predict_window(
                temporal_model,
                window,
                frames_seen=len(window),
                checkpoint=checkpoint,
                model_type=model_type,
                device=device,
            )
            for window in windows
            if window
        ]
        probability = max(probabilities, default=0.0)
        print(f"sliding_windows: {len(probabilities)}")
        print("window_probabilities: " + ",".join(f"{value:.6f}" for value in probabilities))
        required_windows = max(args.min_confirmed_windows, 1)
        confirmed_windows = 0
        consecutive_windows = 0
        for value in probabilities:
            if value >= float(checkpoint["decision_threshold"]):
                consecutive_windows += 1
                confirmed_windows = max(confirmed_windows, consecutive_windows)
            else:
                consecutive_windows = 0
        prediction = int(confirmed_windows >= required_windows)
        print(f"confirmed_windows: {confirmed_windows}")
        print(f"min_confirmed_windows: {required_windows}")
    else:
        probability = _predict_window(
            temporal_model,
            frame_records,
            frames_seen=int(summary.get("frames_seen") or len(frame_records)),
            checkpoint=checkpoint,
            model_type=model_type,
            device=device,
        )
        prediction = int(probability >= float(checkpoint["decision_threshold"]))
    threshold = float(checkpoint["decision_threshold"])
    print(f"prediction: [{prediction}]")
    print(f"fall_probability: {probability:.8f}")
    print(f"threshold: {threshold:.8f}")
    print(f"frames_with_pose: {int(summary.get('frames_with_pose') or 0)}")
    print(f"device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
