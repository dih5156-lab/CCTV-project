#!/usr/bin/env python3
"""버퍼링된 후보 영상 하나에 넘어짐 시계열 체크포인트를 실행한다."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import select
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
    parser.add_argument(
        "--output-feature-json",
        type=Path,
        help="Persist extracted pose features for reuse by another model.",
    )
    parser.add_argument("--worker", action="store_true", help="Reuse models for bounded stdin requests.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    # 실시간 호출과 기존 640 기준 평가를 맞추며, 명시적 대조 시험 값은 허용한다.
    parser.add_argument("--imgsz", type=int, default=640)
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
    """요청한 연속 구간을 평가할 수 있도록 충분한 프레임을 반환한다."""
    required_windows = max(min_confirmed_windows, 1)
    if window_size <= 0:
        return sequence_length
    return max(
        sequence_length,
        window_size + max(stride, 1) * (required_windows - 1),
    )


def _effective_min_confirmed_windows(requested: int, available: int) -> int:
    """입력이 부족해도 사용자가 지정한 최소 연속 확정 수를 낮추지 않는다."""
    return max(requested, 1)


def write_feature_summary(summary: dict, output_path: Path) -> None:
    """추출된 pose 특징을 다른 temporal 모델이 재사용할 수 있게 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_resources(args):
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

    pose_model = None if args.feature_json is not None else _load_pose_model(args.pose_model)
    return device, checkpoint, model_type, temporal_model, pose_model


def _infer(args, resources) -> int:
    device, checkpoint, model_type, temporal_model, pose_model = resources
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
            model=pose_model,
            detector=FallDetector(),
            video_path=args.video,
            max_frames=max_frames,
            frame_stride=1,
            imgsz=args.imgsz,
            confidence_threshold=args.confidence_threshold,
        )
    if args.output_feature_json is not None:
        write_feature_summary(summary, args.output_feature_json)
    frame_records = list(summary.get("frame_records") or [])
    window_inferences = 0
    inference_status = "ok"
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
        required_windows = _effective_min_confirmed_windows(
            args.min_confirmed_windows,
            len(probabilities),
        )
        window_inferences = len(probabilities)
        if window_inferences < required_windows:
            inference_status = "insufficient_windows"
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
    elif frame_records:
        probability = _predict_window(
            temporal_model,
            frame_records,
            frames_seen=int(summary.get("frames_seen") or len(frame_records)),
            checkpoint=checkpoint,
            model_type=model_type,
            device=device,
        )
        window_inferences = 1
        prediction = int(probability >= float(checkpoint["decision_threshold"]))
    else:
        probability = 0.0
        prediction = 0
        inference_status = "insufficient_windows"
    threshold = float(checkpoint["decision_threshold"])
    print(f"window_inferences: {window_inferences}")
    print(f"inference_status: {inference_status}")
    print(f"prediction: [{prediction}]")
    print(f"fall_probability: {probability:.8f}")
    print(f"threshold: {threshold:.8f}")
    print(f"frames_with_pose: {int(summary.get('frames_with_pose') or 0)}")
    print(f"device: {device}")
    return 0


def _serve(args) -> int:
    # 로딩은 한 번만 수행하되 인물 선택·시간창·특징은 요청마다 새로 계산한다.
    if args.feature_json is not None or args.output_feature_json is not None:
        raise SystemExit("worker mode accepts video requests only")
    with contextlib.redirect_stdout(sys.stderr):
        resources = _load_resources(args)
    while select.select([sys.stdin], [], [], 60.0)[0]:
        line = sys.stdin.readline(4097)
        if not line:
            break
        if len(line) > 4096:
            return 1
        request_id = None
        try:
            request = json.loads(line)
            request_id = request["request_id"]
            args.video = Path(request["video"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                _infer(args, resources)
            result = {"request_id": request_id, "stdout": output.getvalue()}
        except Exception as exc:
            # CUDA 오류가 난 모델을 다음 요청에 넘기지 않고 부모가 새 프로세스를 만들게 한다.
            result = {"request_id": request_id, "error": type(exc).__name__ + ": " + str(exc)}
            print("TCN_RESULT " + json.dumps(result), flush=True)
            return 1
        print("TCN_RESULT " + json.dumps(result), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return _serve(args)
    if args.feature_json is None and args.video is None:
        raise SystemExit("--video is required unless --feature-json is provided")
    return _infer(args, _load_resources(args))


if __name__ == "__main__":
    raise SystemExit(main())
