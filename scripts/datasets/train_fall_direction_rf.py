#!/usr/bin/env python3
"""학습 manifest의 낙상 방향 라벨로 방향 분류 RF를 학습한다.

기존 YOLO-pose 요약 feature cache를 재사용한다. 이 모델은 ``front/back/side``
등 상세 유형을 DB 메타데이터에 기록하기 위한 보조 모델이며, 경보 event_type은
항상 ``fall_detected``로 유지한다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_NAMES = [
    "frames_seen", "frames_with_pose_ratio", "max_fall_score", "mean_fall_score",
    "top5_mean_fall_score", "fall_score_std", "frames_score_ge_2_ratio",
    "frames_score_ge_3_ratio", "frames_score_ge_4_ratio", "longest_score_ge_3_run_ratio",
    "max_bbox_aspect", "mean_bbox_aspect", "max_bbox_area_ratio", "mean_bbox_area_ratio",
    "mean_visible_keypoints", "min_visible_keypoints", "mean_keypoint_confidence",
    "max_detection_confidence", "torso_horizontal_ratio", "leg_above_head_ratio",
    "wide_bbox_low_head_ratio", "wide_bbox_candidate_ratio", "low_vertical_span_ratio",
    "torso_flattened_ratio", "missing_leg_ratio", "missing_shoulder_ratio",
    "folded_floor_pose_ratio", "fall_score_slope", "fall_score_start_mean",
    "fall_score_end_mean", "fall_score_end_minus_start", "max_fall_score_rise",
    "fall_score_peak_position", "late_score_ge_3_ratio", "bbox_aspect_end_minus_start",
    "bbox_area_end_minus_start", "high_score_transition_ratio", "max_pose_width_height_ratio",
    "mean_pose_width_height_ratio", "max_torso_angle_from_vertical",
    "mean_torso_angle_from_vertical", "mean_torso_length_bbox_ratio",
    "torso_angle_end_minus_start", "hip_center_y_end_minus_start",
    "body_center_y_end_minus_start", "bbox_center_y_end_minus_start", "max_hip_center_y_rise",
    "max_body_center_y_rise", "max_bbox_center_y_rise", "mean_abs_hip_center_y_velocity",
    "vertical_to_horizontal_transition_ratio", "horizontal_posture_persistence_ratio",
    "max_torso_angle_rise",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_id(row: dict[str, Any]) -> str:
    value = str(row.get("scene_id") or Path(str(row["video_path"])).stem)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _feature_path(feature_cache: Path, row: dict[str, Any], max_frames: int, frame_stride: int, margin: int) -> Path:
    return feature_cache / f"{_safe_id(row)}_labeled_window_max{max_frames}_stride{frame_stride}_margin{margin}.json"


def normalize_direction(row: dict[str, Any]) -> str:
    """원천 라벨 표현을 운영 DB에서 사용할 안정적인 방향 코드로 변환한다."""
    value = " ".join(
        str(row.get(key) or "") for key in ("scene_category", "fall_type")
    ).lower()
    if any(token in value for token in ("측면", "옆", "side")):
        return "side"
    if any(token in value for token in ("후면", "뒤", "back")):
        return "back"
    if any(token in value for token in ("전면", "앞", "front")):
        return "front"
    return "other"


def _load_rows(rows: list[dict[str, Any]], cache: Path, *, max_frames: int, frame_stride: int, margin: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[list[float]] = []
    labels: list[str] = []
    scene_ids: list[str] = []
    for row in rows:
        if not bool(row.get("is_fall")):
            continue
        direction = normalize_direction(row)
        path = _feature_path(cache, row, max_frames, frame_stride, margin)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("frames_with_pose", 0)) < 1:
            continue
        vector = [float(payload.get(name, 0.0)) for name in FEATURE_NAMES]
        features.append(vector)
        labels.append(direction)
        scene_ids.append(str(row.get("scene_id") or payload.get("scene_id")))
    if not features:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,), dtype=object), []
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=object), scene_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--validation-feature-cache", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, default=PROJECT_ROOT / "models/experiments/fall_direction_rf.pkl")
    parser.add_argument("--metrics-json", type=Path, default=PROJECT_ROOT / "models/experiments/fall_direction_rf_metrics.json")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--fall-window-margin-frames", type=int, default=120)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    train_x, train_y, _ = _load_rows(
        _read_jsonl(args.manifest), args.feature_cache,
        max_frames=args.max_frames, frame_stride=args.frame_stride,
        margin=args.fall_window_margin_frames,
    )
    val_x, val_y, _ = _load_rows(
        _read_jsonl(args.validation_manifest), args.validation_feature_cache,
        max_frames=args.max_frames, frame_stride=args.frame_stride,
        margin=args.fall_window_margin_frames,
    )
    if len(set(train_y.tolist())) < 2:
        raise SystemExit(f"direction training needs at least two classes: {Counter(train_y.tolist())}")
    if len(train_y) == 0 or len(val_y) == 0:
        raise SystemExit("direction feature cache is empty; run pose feature extraction first")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    predictions = model.predict(val_x)
    labels = sorted(set(train_y.tolist()) | set(val_y.tolist()))
    metrics = {
        "model_kind": "fall_direction_rf",
        "feature_names": FEATURE_NAMES,
        "classes": labels,
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "train_class_counts": dict(Counter(train_y.tolist())),
        "validation_class_counts": dict(Counter(val_y.tolist())),
        "accuracy": float(accuracy_score(val_y, predictions)),
        "confusion_matrix": confusion_matrix(val_y, predictions, labels=labels).tolist(),
        "classification_report": classification_report(val_y, predictions, labels=labels, output_dict=True, zero_division=0),
    }
    bundle = {
        "bundle_schema_version": 1,
        "model_kind": "fall_direction_rf",
        "feature_schema_version": "yolo_pose_fall_summary_v2",
        "feature_names": FEATURE_NAMES,
        "classes": labels,
        "model": model,
        "inference_config": {"min_probability": 0.70},
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_model)
    args.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(args.output_model), "metrics": str(args.metrics_json), **metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
