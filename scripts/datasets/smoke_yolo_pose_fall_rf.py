#!/usr/bin/env python3
"""Run a YOLO-pose fall RF model on one video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.train_yolo_pose_fall_rf import (  # noqa: E402
    _extract_video_features,
    _load_pose_model,
)
from src.core.ai._fall_detector import FallDetector  # noqa: E402

DEFAULT_INFERENCE_CONFIG = {
    "max_frames": 30,
    "frame_stride": 6,
    "imgsz": 640,
    "confidence_threshold": 0.35,
    "candidate_window_frames": 0,
    "candidate_window_seconds": 0.0,
}


def _resolve_inference_config(
    bundle: dict,
    cli_overrides: dict,
) -> dict:
    config = dict(DEFAULT_INFERENCE_CONFIG)
    config.update(bundle.get("inference_config") or {})
    config.update(
        {
            key: value
            for key, value in cli_overrides.items()
            if value is not None
        }
    )
    return config


def _fall_probability_from_classifier(
    classifier: object,
    probabilities: list[list[float]],
) -> float | None:
    if not probabilities:
        return None
    classes = list(getattr(classifier, "classes_", []))
    if 1 not in classes:
        return None
    return float(probabilities[0][classes.index(1)])


def _tail_sampling_window(
    *,
    total_frames: int,
    window_frames: int,
    candidate_end_frame: int | None = None,
) -> tuple[int | None, int | None]:
    effective_end_frame = min(
        max(int(candidate_end_frame or total_frames), 0),
        total_frames,
    )
    if window_frames <= 0:
        return None, None
    if candidate_end_frame is None and effective_end_frame <= window_frames:
        return None, None
    return max(effective_end_frame - window_frames + 1, 1), effective_end_frame


def _candidate_window_frames(inference_config: dict, *, video_fps: float) -> int:
    window_seconds = float(inference_config.get("candidate_window_seconds") or 0.0)
    if window_seconds > 0.0 and video_fps > 0.0:
        return max(int(round(window_seconds * video_fps)) + 1, 1)
    return max(int(inference_config.get("candidate_window_frames") or 0), 0)


def _video_metadata(video_path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0, 0.0
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        )
    finally:
        capture.release()


def _select_model_features(
    *,
    summary: dict,
    model_feature_names: list[str] | None,
    expected_feature_count: int,
) -> np.ndarray:
    summary_feature_names = summary.get("feature_names")
    summary_feature_vector = summary["feature_vector"]
    if model_feature_names and summary_feature_names:
        values_by_name = dict(zip(summary_feature_names, summary_feature_vector))
        missing = [name for name in model_feature_names if name not in values_by_name]
        if missing:
            raise ValueError(f"missing model features: {missing}")
        selected_values = [values_by_name[name] for name in model_feature_names]
    else:
        selected_values = summary_feature_vector

    features = np.asarray(selected_values, dtype=np.float32).reshape(1, -1)
    if features.shape[1] != expected_feature_count:
        raise ValueError(
            f"model expects {expected_feature_count} features, extracted {features.shape[1]}"
        )
    return features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--candidate-window-frames", type=int)
    parser.add_argument("--candidate-window-seconds", type=float)
    parser.add_argument("--candidate-end-frame", type=int)
    parser.add_argument("--prediction-batch-size", type=int, default=0)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    classifier = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    feature_names = bundle.get("feature_names") if isinstance(bundle, dict) else None
    bundle_metadata = bundle if isinstance(bundle, dict) else {}
    model_kind = bundle_metadata.get("model_kind")
    if model_kind not in {None, "yolo_pose_summary_rf"}:
        raise ValueError(f"unsupported model kind: {model_kind}")
    inference_config = _resolve_inference_config(
        bundle_metadata,
        {
            "max_frames": args.max_frames,
            "frame_stride": args.frame_stride,
            "imgsz": args.imgsz,
            "confidence_threshold": args.confidence_threshold,
            "candidate_window_frames": args.candidate_window_frames,
            "candidate_window_seconds": args.candidate_window_seconds,
        },
    )
    total_frames, video_fps = _video_metadata(args.video)
    sampling_start_frame, sampling_end_frame = _tail_sampling_window(
        total_frames=total_frames,
        window_frames=_candidate_window_frames(
            inference_config,
            video_fps=video_fps,
        ),
        candidate_end_frame=args.candidate_end_frame,
    )
    summary = _extract_video_features(
        model=_load_pose_model(args.pose_model),
        detector=FallDetector(),
        video_path=args.video,
        max_frames=max(int(inference_config["max_frames"]), 1),
        frame_stride=max(int(inference_config["frame_stride"]), 1),
        imgsz=int(inference_config["imgsz"]),
        confidence_threshold=float(inference_config["confidence_threshold"]),
        prediction_batch_size=(
            1
            if args.pose_model.suffix.lower() == ".engine"
            and args.prediction_batch_size <= 0
            else args.prediction_batch_size
        ),
        start_frame=sampling_start_frame,
        end_frame=sampling_end_frame,
    )
    expected = getattr(classifier, "n_features_in_", len(summary["feature_vector"]))
    features = _select_model_features(
        summary=summary,
        model_feature_names=feature_names,
        expected_feature_count=expected,
    )
    prediction = classifier.predict(features)
    probability = classifier.predict_proba(features).tolist() if hasattr(classifier, "predict_proba") else None
    fall_probability = (
        _fall_probability_from_classifier(classifier, probability)
        if probability is not None
        else None
    )
    print(f"model: {args.model}")
    print(f"model_kind: {model_kind or 'legacy_yolo_pose_summary_rf'}")
    print(
        "bundle_schema_version: "
        f"{bundle_metadata.get('bundle_schema_version', 0)}"
    )
    print(f"feature_names: {len(feature_names or [])}")
    print(f"inference_config: {inference_config}")
    print(f"video_fps: {video_fps}")
    print(f"sampling_start_frame: {sampling_start_frame}")
    print(f"sampling_end_frame: {sampling_end_frame}")
    print(f"frames_seen: {summary['frames_seen']}")
    print(f"frames_with_pose: {summary['frames_with_pose']}")
    print(f"prediction: {prediction.tolist()}")
    if probability is not None:
        print(f"predict_proba: {probability}")
    if fall_probability is not None:
        print(f"fall_probability: {fall_probability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
