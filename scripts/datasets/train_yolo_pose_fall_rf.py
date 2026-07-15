#!/usr/bin/env python3
"""Train a lightweight fall classifier from YOLO-pose fall features.

This pipeline matches the runtime shape better than the MediaPipe falldata RF:
it runs YOLO-pose on each manifest video, reuses the project's FallDetector
score logic, summarizes the per-frame signals, and trains a small classifier on
those summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ai._fall_detector import FallDetector  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "data/fall_eval/open_fall_train_manifest.jsonl"
DEFAULT_VALIDATION_MANIFEST = PROJECT_ROOT / "data/fall_eval/open_fall_val_manifest.jsonl"
DEFAULT_FEATURE_CACHE = PROJECT_ROOT / "data/fall_eval/yolo_pose_fall_feature_cache"
DEFAULT_VALIDATION_FEATURE_CACHE = (
    PROJECT_ROOT / "data/fall_eval/yolo_pose_fall_validation_feature_cache"
)
DEFAULT_POSE_MODEL = PROJECT_ROOT / "models/yolov8n-pose.pt"
DEFAULT_OUTPUT_MODEL = PROJECT_ROOT / "models/experiments/yolo_pose_fall_rf.pkl"
DEFAULT_METRICS = PROJECT_ROOT / "models/experiments/yolo_pose_fall_rf_metrics.json"
FALL_LABEL = 1
NON_FALL_LABEL = 0

FEATURE_NAMES = [
    "frames_seen",
    "frames_with_pose_ratio",
    "max_fall_score",
    "mean_fall_score",
    "top5_mean_fall_score",
    "fall_score_std",
    "frames_score_ge_2_ratio",
    "frames_score_ge_3_ratio",
    "frames_score_ge_4_ratio",
    "longest_score_ge_3_run_ratio",
    "max_bbox_aspect",
    "mean_bbox_aspect",
    "max_bbox_area_ratio",
    "mean_bbox_area_ratio",
    "mean_visible_keypoints",
    "min_visible_keypoints",
    "mean_keypoint_confidence",
    "max_detection_confidence",
    "torso_horizontal_ratio",
    "leg_above_head_ratio",
    "wide_bbox_low_head_ratio",
    "wide_bbox_candidate_ratio",
    "low_vertical_span_ratio",
    "torso_flattened_ratio",
    "missing_leg_ratio",
    "missing_shoulder_ratio",
    "folded_floor_pose_ratio",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_id(row: dict[str, Any]) -> str:
    value = str(row.get("scene_id") or Path(str(row["video_path"])).stem)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _label_for_row(row: dict[str, Any]) -> int:
    return FALL_LABEL if bool(row.get("is_fall")) else NON_FALL_LABEL


def _select_rows(rows: list[dict[str, Any]], max_videos: int) -> list[dict[str, Any]]:
    if max_videos <= 0 or len(rows) <= max_videos:
        return rows
    fall_rows = [row for row in rows if bool(row.get("is_fall"))]
    non_fall_rows = [row for row in rows if not bool(row.get("is_fall"))]
    primary_quota = max_videos // 2
    selected = non_fall_rows[:primary_quota] + fall_rows[:primary_quota]
    remaining = max_videos - len(selected)
    if remaining > 0:
        selected_ids = {id(row) for row in selected}
        selected.extend(row for row in rows if id(row) not in selected_ids)
    return selected[:max_videos]


def _feature_path(feature_cache: Path, row: dict[str, Any], max_frames: int, frame_stride: int) -> Path:
    return feature_cache / f"{_safe_id(row)}_uniform_max{max_frames}_stride{frame_stride}.json"


def _reason_key(reason: str) -> str:
    return reason.split(":", 1)[0]


def _longest_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _empty_summary(frames_seen: int) -> dict[str, Any]:
    return {
        "frames_seen": frames_seen,
        "frames_with_pose": 0,
        "feature_vector": [0.0 for _ in FEATURE_NAMES],
        "reason_counts": {},
        "frame_records": [],
    }


def _summarize_frames(frame_records: list[dict[str, Any]], frames_seen: int) -> dict[str, Any]:
    if not frame_records:
        return _empty_summary(frames_seen)

    fall_scores = np.asarray([record["fall_score"] for record in frame_records], dtype=np.float32)
    bbox_aspects = np.asarray([record["bbox_aspect"] for record in frame_records], dtype=np.float32)
    bbox_area_ratios = np.asarray(
        [record["bbox_area_ratio"] for record in frame_records],
        dtype=np.float32,
    )
    visible_counts = np.asarray(
        [record["visible_keypoints"] for record in frame_records],
        dtype=np.float32,
    )
    keypoint_confidences = np.asarray(
        [record["mean_keypoint_confidence"] for record in frame_records],
        dtype=np.float32,
    )
    detection_confidences = np.asarray(
        [record["detection_confidence"] for record in frame_records],
        dtype=np.float32,
    )
    reason_counts = Counter(
        _reason_key(reason)
        for record in frame_records
        for reason in record.get("fall_reasons", [])
    )
    pose_frames = len(frame_records)
    denominator = max(frames_seen, 1)
    top_count = min(5, len(fall_scores))
    top5_mean = float(np.sort(fall_scores)[-top_count:].mean()) if top_count else 0.0
    score_ge_3 = [float(score) >= 3.0 for score in fall_scores]

    def reason_ratio(key: str) -> float:
        return float(reason_counts.get(key, 0) / denominator)

    values = {
        "frames_seen": float(frames_seen),
        "frames_with_pose_ratio": float(pose_frames / denominator),
        "max_fall_score": float(fall_scores.max()),
        "mean_fall_score": float(fall_scores.mean()),
        "top5_mean_fall_score": top5_mean,
        "fall_score_std": float(fall_scores.std()),
        "frames_score_ge_2_ratio": float((fall_scores >= 2.0).sum() / denominator),
        "frames_score_ge_3_ratio": float((fall_scores >= 3.0).sum() / denominator),
        "frames_score_ge_4_ratio": float((fall_scores >= 4.0).sum() / denominator),
        "longest_score_ge_3_run_ratio": float(_longest_run(score_ge_3) / denominator),
        "max_bbox_aspect": float(bbox_aspects.max()),
        "mean_bbox_aspect": float(bbox_aspects.mean()),
        "max_bbox_area_ratio": float(bbox_area_ratios.max()),
        "mean_bbox_area_ratio": float(bbox_area_ratios.mean()),
        "mean_visible_keypoints": float(visible_counts.mean()),
        "min_visible_keypoints": float(visible_counts.min()),
        "mean_keypoint_confidence": float(keypoint_confidences.mean()),
        "max_detection_confidence": float(detection_confidences.max()),
        "torso_horizontal_ratio": reason_ratio("torso_horizontal"),
        "leg_above_head_ratio": reason_ratio("leg_above_head"),
        "wide_bbox_low_head_ratio": reason_ratio("wide_bbox_low_head"),
        "wide_bbox_candidate_ratio": reason_ratio("wide_bbox_candidate"),
        "low_vertical_span_ratio": reason_ratio("low_vertical_span"),
        "torso_flattened_ratio": reason_ratio("torso_flattened"),
        "missing_leg_ratio": reason_ratio("missing_leg"),
        "missing_shoulder_ratio": reason_ratio("missing_shoulder"),
        "folded_floor_pose_ratio": reason_ratio("folded_floor_pose"),
    }
    return {
        "frames_seen": frames_seen,
        "frames_with_pose": pose_frames,
        "feature_names": FEATURE_NAMES,
        "feature_vector": [values[name] for name in FEATURE_NAMES],
        "reason_counts": dict(sorted(reason_counts.items())),
        "frame_records": frame_records,
    }


def _load_pose_model(model_path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(model_path))


def _extract_video_features(
    *,
    model: Any,
    detector: FallDetector,
    video_path: Path,
    max_frames: int,
    frame_stride: int,
    imgsz: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        frame_indices = sorted(
            {
                int(value)
                for value in np.linspace(0, max(total_frames - 1, 0), num=max_frames)
            }
        )
    else:
        frame_indices = list(range(0, max_frames * frame_stride, frame_stride))

    frame_records: list[dict[str, Any]] = []
    sampled_frames = 0
    for zero_based_frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, zero_based_frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        sampled_frames += 1
        frame_index = zero_based_frame_index + 1
        results = model.predict(
            frame,
            imgsz=imgsz,
            conf=confidence_threshold,
            verbose=False,
        )
        if not results:
            continue
        result = results[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            continue
        confidences = result.boxes.conf.detach().cpu().numpy()
        best_index = int(np.argmax(confidences))
        xyxy = result.boxes.xyxy[best_index].detach().cpu().numpy().astype(float)
        keypoints_xy = result.keypoints.xy[best_index].detach().cpu().numpy().astype(float)
        keypoints_conf = result.keypoints.conf[best_index].detach().cpu().numpy().astype(float)
        keypoints = np.concatenate([keypoints_xy, keypoints_conf[:, None]], axis=1)
        bbox_width = max(float(xyxy[2] - xyxy[0]), 1.0)
        bbox_height = max(float(xyxy[3] - xyxy[1]), 1.0)
        score = detector._score_fall(keypoints.astype(np.float32), int(bbox_width), int(bbox_height))
        visible = keypoints_conf >= detector.min_keypoint_confidence
        frame_h, frame_w = frame.shape[:2]
        frame_records.append(
            {
                "frame_index": frame_index,
                "fall_score": float(score.score),
                "fall_reasons": list(score.reasons),
                "runtime_is_fall": bool(score.score >= detector.score_threshold),
                "detection_confidence": float(confidences[best_index]),
                "bbox_aspect": float(bbox_width / bbox_height),
                "bbox_area_ratio": float((bbox_width * bbox_height) / max(frame_w * frame_h, 1)),
                "visible_keypoints": int(visible.sum()),
                "mean_keypoint_confidence": float(keypoints_conf.mean()),
            }
        )
    capture.release()
    return _summarize_frames(frame_records, sampled_frames)


def _ensure_features(
    *,
    rows: list[dict[str, Any]],
    feature_cache: Path,
    model: Any,
    detector: FallDetector,
    max_frames: int,
    frame_stride: int,
    imgsz: int,
    confidence_threshold: float,
    force_extract: bool,
    label: str,
) -> None:
    feature_cache.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        output = _feature_path(feature_cache, row, max_frames, frame_stride)
        if output.exists() and not force_extract:
            print(f"[{label} {index}/{len(rows)}] cache {_safe_id(row)}", flush=True)
            continue
        print(f"[{label} {index}/{len(rows)}] extract {_safe_id(row)}", flush=True)
        summary = _extract_video_features(
            model=model,
            detector=detector,
            video_path=PROJECT_ROOT / str(row["video_path"]),
            max_frames=max_frames,
            frame_stride=frame_stride,
            imgsz=imgsz,
            confidence_threshold=confidence_threshold,
        )
        payload = {
            "scene_id": _safe_id(row),
            "video_path": row.get("video_path"),
            "is_fall": bool(row.get("is_fall")),
            "max_frames": max_frames,
            "frame_stride": frame_stride,
            **summary,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_dataset(
    rows: list[dict[str, Any]],
    *,
    feature_cache: Path,
    max_frames: int,
    frame_stride: int,
    min_pose_frames: int,
) -> dict[str, Any]:
    features: list[list[float]] = []
    labels: list[int] = []
    scene_ids: list[str] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        path = _feature_path(feature_cache, row, max_frames, frame_stride)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("frames_with_pose") or 0) < min_pose_frames:
            excluded.append(
                {
                    "scene_id": _safe_id(row),
                    "frames_with_pose": int(payload.get("frames_with_pose") or 0),
                    "reason": "frames_with_pose_below_minimum",
                }
            )
            continue
        features.append([float(value) for value in payload["feature_vector"]])
        labels.append(_label_for_row(row))
        scene_ids.append(_safe_id(row))
    if not features:
        raise SystemExit("no rows left after YOLO-pose feature filtering")
    return {
        "x": np.asarray(features, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "scene_ids": scene_ids,
        "excluded": excluded,
    }


def _class_counts(values: np.ndarray) -> dict[str, int]:
    return {str(label): int((values == label).sum()) for label in sorted(set(values.tolist()))}


def _predict_with_threshold(model: Any, x: np.ndarray, threshold: float) -> tuple[np.ndarray, list[list[float]]]:
    probabilities = model.predict_proba(x).tolist()
    classes = list(model.classes_)
    fall_index = classes.index(FALL_LABEL)
    predictions = np.asarray(
        [FALL_LABEL if row[fall_index] >= threshold else NON_FALL_LABEL for row in probabilities],
        dtype=np.int64,
    )
    return predictions, probabilities


def _evaluate(
    model: Any,
    dataset: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix

    predictions, probabilities = _predict_with_threshold(model, dataset["x"], threshold)
    matrix = confusion_matrix(dataset["y"], predictions, labels=[FALL_LABEL, NON_FALL_LABEL])
    false_positives = []
    false_negatives = []
    for index, (scene_id, true, predicted) in enumerate(
        zip(dataset["scene_ids"], dataset["y"], predictions)
    ):
        record = {
            "scene_id": scene_id,
            "true": int(true),
            "predicted": int(predicted),
            "probability": probabilities[index],
        }
        if true == NON_FALL_LABEL and predicted == FALL_LABEL:
            false_positives.append(record)
        if true == FALL_LABEL and predicted == NON_FALL_LABEL:
            false_negatives.append(record)
    return {
        "threshold": threshold,
        "confusion_matrix_labels": ["fall", "non_fall"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            dataset["y"],
            predictions,
            labels=[FALL_LABEL, NON_FALL_LABEL],
            target_names=["fall", "non_fall"],
            zero_division=0,
            output_dict=True,
        ),
        "errors": {
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
    }


def _threshold_sweep(model: Any, dataset: dict[str, Any], thresholds: list[float]) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        evaluation = _evaluate(model, dataset, threshold=threshold)
        fall_report = evaluation["classification_report"]["fall"]
        non_fall_report = evaluation["classification_report"]["non_fall"]
        rows.append(
            {
                "threshold": threshold,
                "false_positive_count": evaluation["errors"]["false_positive_count"],
                "false_negative_count": evaluation["errors"]["false_negative_count"],
                "fall_recall": fall_report["recall"],
                "fall_precision": fall_report["precision"],
                "non_fall_recall": non_fall_report["recall"],
                "non_fall_precision": non_fall_report["precision"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument(
        "--validation-feature-cache",
        type=Path,
        default=DEFAULT_VALIDATION_FEATURE_CACHE,
    )
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--max-videos", type=int, default=200)
    parser.add_argument("--validation-max-videos", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--min-pose-frames", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=0.6)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> int:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    args = parse_args()
    train_rows = _select_rows(_read_jsonl(args.manifest), args.max_videos)
    validation_rows = _select_rows(_read_jsonl(args.validation_manifest), args.validation_max_videos)
    if not train_rows:
        raise SystemExit("no train rows selected")
    if not validation_rows:
        raise SystemExit("no validation rows selected")

    pose_model = _load_pose_model(args.pose_model)
    detector = FallDetector()
    _ensure_features(
        rows=train_rows,
        feature_cache=args.feature_cache,
        model=pose_model,
        detector=detector,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        imgsz=args.imgsz,
        confidence_threshold=args.confidence_threshold,
        force_extract=args.force_extract,
        label="train",
    )
    _ensure_features(
        rows=validation_rows,
        feature_cache=args.validation_feature_cache,
        model=pose_model,
        detector=detector,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        imgsz=args.imgsz,
        confidence_threshold=args.confidence_threshold,
        force_extract=args.force_extract,
        label="validation",
    )

    train_dataset = _load_dataset(
        train_rows,
        feature_cache=args.feature_cache,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        min_pose_frames=args.min_pose_frames,
    )
    validation_dataset = _load_dataset(
        validation_rows,
        feature_cache=args.validation_feature_cache,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        min_pose_frames=args.min_pose_frames,
    )
    class_count = len(set(train_dataset["y"].tolist()))
    holdout_count = int(np.ceil(len(train_dataset["y"]) * 0.25))
    if class_count < 2:
        raise SystemExit(f"need both fall and non-fall classes, got {_class_counts(train_dataset['y'])}")
    if holdout_count < class_count:
        x_train = train_dataset["x"]
        x_holdout = train_dataset["x"]
        y_train = train_dataset["y"]
        y_holdout = train_dataset["y"]
        ids_train = train_dataset["scene_ids"]
        ids_holdout = train_dataset["scene_ids"]
        holdout_method = "train_resubstitution_small_sample"
    else:
        x_train, x_holdout, y_train, y_holdout, ids_train, ids_holdout = train_test_split(
            train_dataset["x"],
            train_dataset["y"],
            train_dataset["scene_ids"],
            test_size=0.25,
            random_state=args.random_state,
            stratify=train_dataset["y"],
        )
        holdout_method = "stratified_random"
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    holdout_dataset = {"x": x_holdout, "y": y_holdout, "scene_ids": ids_holdout}
    thresholds = [round(value, 2) for value in np.arange(0.35, 0.91, 0.05)]
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "validation_manifest": str(args.validation_manifest),
        "pose_model": str(args.pose_model),
        "output_model": str(args.output_model),
        "feature_names": FEATURE_NAMES,
        "rows": len(train_rows),
        "effective_rows": int(len(train_dataset["y"])),
        "validation_rows": len(validation_rows),
        "validation_effective_rows": int(len(validation_dataset["y"])),
        "class_counts": _class_counts(train_dataset["y"]),
        "validation_class_counts": _class_counts(validation_dataset["y"]),
        "excluded": train_dataset["excluded"],
        "validation_excluded": validation_dataset["excluded"],
        "model_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "decision_threshold": args.decision_threshold,
            "max_frames": args.max_frames,
            "frame_stride": args.frame_stride,
            "imgsz": args.imgsz,
            "confidence_threshold": args.confidence_threshold,
            "min_pose_frames": args.min_pose_frames,
        },
        "holdout_method": holdout_method,
        "holdout": _evaluate(model, holdout_dataset, threshold=args.decision_threshold),
        "validation": _evaluate(model, validation_dataset, threshold=args.decision_threshold),
        "validation_threshold_sweep": _threshold_sweep(model, validation_dataset, thresholds),
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, args.output_model)
    args.metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_errors = metrics["validation"]["errors"]
    print(f"features: {train_dataset['x'].shape}")
    print(f"class_counts: {metrics['class_counts']}")
    print(f"validation confusion_matrix labels={metrics['validation']['confusion_matrix_labels']}:")
    print(np.asarray(metrics["validation"]["confusion_matrix"]))
    print(
        "validation_errors: "
        f"FP={validation_errors['false_positive_count']} "
        f"FN={validation_errors['false_negative_count']}"
    )
    print(f"model: {args.output_model}")
    print(f"metrics: {args.metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
