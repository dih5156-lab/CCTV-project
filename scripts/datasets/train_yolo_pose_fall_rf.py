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
MODEL_BUNDLE_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = "yolo_pose_fall_summary_v2"

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
    "fall_score_slope",
    "fall_score_start_mean",
    "fall_score_end_mean",
    "fall_score_end_minus_start",
    "max_fall_score_rise",
    "fall_score_peak_position",
    "late_score_ge_3_ratio",
    "bbox_aspect_end_minus_start",
    "bbox_area_end_minus_start",
    "high_score_transition_ratio",
    "max_pose_width_height_ratio",
    "mean_pose_width_height_ratio",
    "max_torso_angle_from_vertical",
    "mean_torso_angle_from_vertical",
    "mean_torso_length_bbox_ratio",
    "torso_angle_end_minus_start",
    "hip_center_y_end_minus_start",
    "body_center_y_end_minus_start",
    "bbox_center_y_end_minus_start",
    "max_hip_center_y_rise",
    "max_body_center_y_rise",
    "max_bbox_center_y_rise",
    "mean_abs_hip_center_y_velocity",
    "vertical_to_horizontal_transition_ratio",
    "horizontal_posture_persistence_ratio",
    "max_torso_angle_rise",
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


def _scene_base(scene_id: str) -> str:
    parts = str(scene_id).rsplit("_C", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return str(scene_id)


def _group_holdout_indices(
    scene_ids: list[str],
    labels: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    groups = [_scene_base(scene_id) for scene_id in scene_ids]
    unique_groups = np.asarray(sorted(set(groups)), dtype=object)
    unique_group_count = len(unique_groups)
    effective_test_size = max(test_size, min(0.5, 2 / unique_group_count))
    holdout_group_count = min(
        max(int(np.ceil(unique_group_count * effective_test_size)), 1),
        max(unique_group_count - 1, 1),
    )
    random_generator = np.random.RandomState(random_state)
    first_split: tuple[np.ndarray, np.ndarray] | None = None
    selected_split: tuple[np.ndarray, np.ndarray] | None = None
    group_array = np.asarray(groups, dtype=object)
    for _ in range(100):
        shuffled_groups = random_generator.permutation(unique_groups)
        holdout_groups_for_split = set(shuffled_groups[:holdout_group_count])
        holdout_mask = np.asarray(
            [group in holdout_groups_for_split for group in group_array],
            dtype=bool,
        )
        holdout_indices = np.flatnonzero(holdout_mask)
        train_indices = np.flatnonzero(~holdout_mask)
        if first_split is None:
            first_split = (train_indices, holdout_indices)
        if len(set(labels[train_indices])) >= 2 and len(set(labels[holdout_indices])) >= 2:
            selected_split = (train_indices, holdout_indices)
            break
    train_indices, holdout_indices = selected_split or first_split or (
        np.arange(len(labels)),
        np.arange(len(labels)),
    )
    train_groups = sorted({groups[index] for index in train_indices})
    holdout_groups = sorted({groups[index] for index in holdout_indices})
    split_info = {
        "method": "group_shuffle",
        "group_by": "scene_base",
        "requested_test_size": test_size,
        "effective_group_test_size": effective_test_size,
        "train_groups": train_groups,
        "test_groups": holdout_groups,
        "group_overlap": sorted(set(train_groups) & set(holdout_groups)),
        "train_class_counts": _class_counts(labels[train_indices]),
        "test_class_counts": _class_counts(labels[holdout_indices]),
    }
    if selected_split is None:
        split_info["warning"] = "could not find a group holdout split containing both classes"
    return train_indices, holdout_indices, split_info


def _label_for_row(row: dict[str, Any]) -> int:
    return FALL_LABEL if bool(row.get("is_fall")) else NON_FALL_LABEL


def _select_rows(rows: list[dict[str, Any]], max_videos: int) -> list[dict[str, Any]]:
    if max_videos <= 0 or len(rows) <= max_videos:
        return rows

    def select_scene_diverse(
        class_rows: list[dict[str, Any]],
        limit: int,
        *,
        environment_offset: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in class_rows:
            scene_id = str(row.get("scene_id") or Path(str(row.get("video_path", ""))).stem)
            grouped.setdefault(_scene_base(scene_id), []).append(row)

        environment_groups: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
        for group_rows in grouped.values():
            first_row = group_rows[0]
            environment = (
                str(first_row.get("scene_location") or "unknown"),
                str(first_row.get("scene_position") or "unknown"),
            )
            environment_groups.setdefault(environment, []).append(group_rows)

        environment_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for environment, scene_groups in environment_groups.items():
            rows_for_environment: list[dict[str, Any]] = []
            max_cameras = max(len(group_rows) for group_rows in scene_groups)
            for camera_offset in range(max_cameras):
                rows_for_environment.extend(
                    group_rows[camera_offset]
                    for group_rows in scene_groups
                    if camera_offset < len(group_rows)
                )
            environment_rows[environment] = rows_for_environment

        selected_rows: list[dict[str, Any]] = []
        offsets = {environment: 0 for environment in environment_rows}
        environments = list(environment_rows)
        if environments:
            rotation = environment_offset % len(environments)
            environments = environments[rotation:] + environments[:rotation]
        while len(selected_rows) < limit:
            added = False
            for environment in environments:
                rows_for_environment = environment_rows[environment]
                offset = offsets[environment]
                if offset >= len(rows_for_environment):
                    continue
                selected_rows.append(rows_for_environment[offset])
                offsets[environment] = offset + 1
                added = True
                if len(selected_rows) >= limit:
                    break
            if not added:
                break
        return selected_rows

    fall_rows = [row for row in rows if bool(row.get("is_fall"))]
    non_fall_rows = [row for row in rows if not bool(row.get("is_fall"))]
    primary_quota = max_videos // 2
    selected = select_scene_diverse(
        non_fall_rows,
        primary_quota,
        environment_offset=0,
    ) + select_scene_diverse(
        fall_rows,
        primary_quota,
        environment_offset=primary_quota,
    )
    remaining = max_videos - len(selected)
    if remaining > 0:
        selected_ids = {id(row) for row in selected}
        selected.extend(row for row in rows if id(row) not in selected_ids)
    return selected[:max_videos]


def _feature_path(
    feature_cache: Path,
    row: dict[str, Any],
    max_frames: int,
    frame_stride: int,
    fall_window_margin_frames: int,
) -> Path:
    return feature_cache / (
        f"{_safe_id(row)}_labeled_window_max{max_frames}_stride{frame_stride}"
        f"_margin{fall_window_margin_frames}.json"
    )


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


def _pose_geometry(
    keypoints: np.ndarray,
    *,
    bbox: np.ndarray,
    frame_width: int,
    frame_height: int,
    min_keypoint_confidence: float,
) -> dict[str, float]:
    visible = keypoints[:, 2] >= min_keypoint_confidence
    visible_xy = keypoints[visible, :2]
    bbox_width = max(float(bbox[2] - bbox[0]), 1.0)
    bbox_height = max(float(bbox[3] - bbox[1]), 1.0)

    def mean_visible(indices: tuple[int, ...]) -> np.ndarray | None:
        selected = [keypoints[index, :2] for index in indices if visible[index]]
        if not selected:
            return None
        return np.asarray(selected, dtype=np.float32).mean(axis=0)

    shoulder_center = mean_visible((5, 6))
    hip_center = mean_visible((11, 12))
    if shoulder_center is not None and hip_center is not None:
        torso_dx = float(hip_center[0] - shoulder_center[0])
        torso_dy = float(hip_center[1] - shoulder_center[1])
        torso_angle = float(
            np.arctan2(abs(torso_dx), abs(torso_dy)) / (np.pi / 2)
        )
        torso_length_ratio = float(
            np.hypot(torso_dx, torso_dy) / bbox_height
        )
    else:
        torso_angle = 0.0
        torso_length_ratio = 0.0

    if len(visible_xy) >= 2:
        pose_width = float(np.ptp(visible_xy[:, 0]))
        pose_height = max(float(np.ptp(visible_xy[:, 1])), 1.0)
        pose_width_height_ratio = pose_width / pose_height
        body_center_y_ratio = float(visible_xy[:, 1].mean() / max(frame_height, 1))
    else:
        pose_width_height_ratio = 0.0
        body_center_y_ratio = 0.0

    return {
        "pose_width_height_ratio": float(pose_width_height_ratio),
        "torso_angle_from_vertical": torso_angle,
        "torso_length_bbox_ratio": torso_length_ratio,
        "hip_center_y_frame_ratio": (
            float(hip_center[1] / max(frame_height, 1))
            if hip_center is not None
            else 0.0
        ),
        "body_center_y_frame_ratio": body_center_y_ratio,
        "bbox_center_y_frame_ratio": float(
            ((bbox[1] + bbox[3]) / 2) / max(frame_height, 1)
        ),
        "bbox_width_frame_ratio": float(bbox_width / max(frame_width, 1)),
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
    pose_width_height_ratios = np.asarray(
        [record.get("pose_width_height_ratio", 0.0) for record in frame_records],
        dtype=np.float32,
    )
    torso_angles = np.asarray(
        [record.get("torso_angle_from_vertical", 0.0) for record in frame_records],
        dtype=np.float32,
    )
    torso_length_ratios = np.asarray(
        [record.get("torso_length_bbox_ratio", 0.0) for record in frame_records],
        dtype=np.float32,
    )
    hip_center_y_ratios = np.asarray(
        [record.get("hip_center_y_frame_ratio", 0.0) for record in frame_records],
        dtype=np.float32,
    )
    body_center_y_ratios = np.asarray(
        [record.get("body_center_y_frame_ratio", 0.0) for record in frame_records],
        dtype=np.float32,
    )
    bbox_center_y_ratios = np.asarray(
        [record.get("bbox_center_y_frame_ratio", 0.0) for record in frame_records],
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
    window = max(1, len(fall_scores) // 3)
    start_scores = fall_scores[:window]
    end_scores = fall_scores[-window:]
    score_deltas = np.diff(fall_scores)
    time_axis = np.arange(len(fall_scores), dtype=np.float32)
    score_slope = (
        float(np.polyfit(time_axis, fall_scores, 1)[0]) if len(fall_scores) >= 2 else 0.0
    )
    transition_count = sum(
        before < 3.0 <= after for before, after in zip(fall_scores[:-1], fall_scores[1:])
    )
    torso_transition_count = sum(
        before < 0.45 <= after
        for before, after in zip(torso_angles[:-1], torso_angles[1:])
    )
    horizontal_posture = torso_angles >= 0.55

    def reason_ratio(key: str) -> float:
        return float(reason_counts.get(key, 0) / denominator)

    def end_minus_start(values: np.ndarray) -> float:
        return float(values[-window:].mean() - values[:window].mean())

    def max_positive_rise(values: np.ndarray) -> float:
        deltas = np.diff(values)
        return max(float(deltas.max()), 0.0) if len(deltas) else 0.0

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
        "fall_score_slope": score_slope,
        "fall_score_start_mean": float(start_scores.mean()),
        "fall_score_end_mean": float(end_scores.mean()),
        "fall_score_end_minus_start": float(end_scores.mean() - start_scores.mean()),
        "max_fall_score_rise": float(score_deltas.max()) if len(score_deltas) else 0.0,
        "fall_score_peak_position": float(np.argmax(fall_scores) / max(len(fall_scores) - 1, 1)),
        "late_score_ge_3_ratio": float((end_scores >= 3.0).sum() / len(end_scores)),
        "bbox_aspect_end_minus_start": float(
            bbox_aspects[-window:].mean() - bbox_aspects[:window].mean()
        ),
        "bbox_area_end_minus_start": float(
            bbox_area_ratios[-window:].mean() - bbox_area_ratios[:window].mean()
        ),
        "high_score_transition_ratio": float(transition_count / max(len(fall_scores) - 1, 1)),
        "max_pose_width_height_ratio": float(pose_width_height_ratios.max()),
        "mean_pose_width_height_ratio": float(pose_width_height_ratios.mean()),
        "max_torso_angle_from_vertical": float(torso_angles.max()),
        "mean_torso_angle_from_vertical": float(torso_angles.mean()),
        "mean_torso_length_bbox_ratio": float(torso_length_ratios.mean()),
        "torso_angle_end_minus_start": end_minus_start(torso_angles),
        "hip_center_y_end_minus_start": end_minus_start(hip_center_y_ratios),
        "body_center_y_end_minus_start": end_minus_start(body_center_y_ratios),
        "bbox_center_y_end_minus_start": end_minus_start(bbox_center_y_ratios),
        "max_hip_center_y_rise": max_positive_rise(hip_center_y_ratios),
        "max_body_center_y_rise": max_positive_rise(body_center_y_ratios),
        "max_bbox_center_y_rise": max_positive_rise(bbox_center_y_ratios),
        "mean_abs_hip_center_y_velocity": float(
            np.abs(np.diff(hip_center_y_ratios)).mean()
            if len(hip_center_y_ratios) >= 2
            else 0.0
        ),
        "vertical_to_horizontal_transition_ratio": float(
            torso_transition_count / max(len(torso_angles) - 1, 1)
        ),
        "horizontal_posture_persistence_ratio": float(horizontal_posture.mean()),
        "max_torso_angle_rise": max_positive_rise(torso_angles),
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


def _predict_pose_results(
    *,
    model: Any,
    frames: list[np.ndarray],
    imgsz: int,
    confidence_threshold: float,
    prediction_batch_size: int,
) -> list[Any]:
    batch_size = len(frames) if prediction_batch_size <= 0 else prediction_batch_size
    results: list[Any] = []
    for start in range(0, len(frames), max(batch_size, 1)):
        results.extend(
            model.predict(
                frames[start : start + batch_size],
                imgsz=imgsz,
                conf=confidence_threshold,
                verbose=False,
            )
        )
    return results


def _select_tracked_pose_index(
    *,
    confidences: np.ndarray,
    centers: np.ndarray,
    previous_center: tuple[float, float] | None,
    frame_diagonal: float,
) -> int:
    if previous_center is None or len(centers) == 1:
        return int(np.argmax(confidences))

    distances = np.hypot(
        centers[:, 0] - previous_center[0],
        centers[:, 1] - previous_center[1],
    ) / max(frame_diagonal, 1.0)
    continuity_scores = confidences - 0.35 * distances
    return int(np.argmax(continuity_scores))


def _sample_video_frames(
    capture: cv2.VideoCapture,
    *,
    max_frames: int,
    frame_stride: int,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[tuple[int, np.ndarray]]:
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        start_index = max(int(start_frame or 1) - 1, 0)
        end_index = min(int(end_frame or total_frames), total_frames) - 1
        if end_index < start_index:
            return []
        frame_indices = sorted(
            {
                int(value)
                for value in np.linspace(start_index, end_index, num=max_frames)
            }
        )
    else:
        start_index = max(int(start_frame or 1) - 1, 0)
        frame_indices = list(
            range(start_index, start_index + max_frames * frame_stride, frame_stride)
        )

    sampled: list[tuple[int, np.ndarray]] = []
    current_frame_index = 0
    for target_frame_index in frame_indices:
        while current_frame_index < target_frame_index:
            if not capture.grab():
                return sampled
            current_frame_index += 1
        ok, frame = capture.read()
        if not ok:
            return sampled
        sampled.append((target_frame_index + 1, frame))
        current_frame_index += 1
    return sampled


def _sampling_window_for_row(
    row: dict[str, Any],
    *,
    margin_frames: int,
) -> tuple[int | None, int | None]:
    if not bool(row.get("is_fall")):
        return None, None
    fall_start = int(row.get("fall_start_frame") or 0)
    fall_end = int(row.get("fall_end_frame") or 0)
    if fall_start <= 0 or fall_end < fall_start:
        return None, None
    scene_length = int(row.get("scene_length") or 0)
    start_frame = max(1, fall_start - max(margin_frames, 0))
    end_frame = fall_end + max(margin_frames, 0)
    if scene_length > 0:
        end_frame = min(end_frame, scene_length)
    return start_frame, end_frame


def _extract_video_features(
    *,
    model: Any,
    detector: FallDetector,
    video_path: Path,
    max_frames: int,
    frame_stride: int,
    imgsz: int,
    confidence_threshold: float,
    prediction_batch_size: int = 0,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    sampled = _sample_video_frames(
        capture,
        max_frames=max_frames,
        frame_stride=frame_stride,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    capture.release()

    if not sampled:
        return _empty_summary(0)
    results = _predict_pose_results(
        model=model,
        frames=[frame for _, frame in sampled],
        imgsz=imgsz,
        confidence_threshold=confidence_threshold,
        prediction_batch_size=prediction_batch_size,
    )
    frame_records: list[dict[str, Any]] = []
    previous_center: tuple[float, float] | None = None
    for (frame_index, frame), result in zip(sampled, results):
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            continue
        confidences = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
        best_index = _select_tracked_pose_index(
            confidences=confidences,
            centers=centers,
            previous_center=previous_center,
            frame_diagonal=float(np.hypot(frame.shape[1], frame.shape[0])),
        )
        previous_center = (float(centers[best_index, 0]), float(centers[best_index, 1]))
        xyxy = result.boxes.xyxy[best_index].detach().cpu().numpy().astype(float)
        keypoints_xy = result.keypoints.xy[best_index].detach().cpu().numpy().astype(float)
        keypoints_conf = result.keypoints.conf[best_index].detach().cpu().numpy().astype(float)
        keypoints = np.concatenate([keypoints_xy, keypoints_conf[:, None]], axis=1)
        bbox_width = max(float(xyxy[2] - xyxy[0]), 1.0)
        bbox_height = max(float(xyxy[3] - xyxy[1]), 1.0)
        score = detector._score_fall(keypoints.astype(np.float32), int(bbox_width), int(bbox_height))
        visible = keypoints_conf >= detector.min_keypoint_confidence
        frame_h, frame_w = frame.shape[:2]
        pose_geometry = _pose_geometry(
            keypoints,
            bbox=xyxy,
            frame_width=frame_w,
            frame_height=frame_h,
            min_keypoint_confidence=detector.min_keypoint_confidence,
        )
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
                **pose_geometry,
            }
        )
    return _summarize_frames(frame_records, len(sampled))


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
    prediction_batch_size: int,
    fall_window_margin_frames: int,
    force_extract: bool,
    label: str,
) -> None:
    feature_cache.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        output = _feature_path(
            feature_cache,
            row,
            max_frames,
            frame_stride,
            fall_window_margin_frames,
        )
        if output.exists() and not force_extract:
            print(f"[{label} {index}/{len(rows)}] cache {_safe_id(row)}", flush=True)
            continue
        print(f"[{label} {index}/{len(rows)}] extract {_safe_id(row)}", flush=True)
        start_frame, end_frame = _sampling_window_for_row(
            row,
            margin_frames=fall_window_margin_frames,
        )
        summary = _extract_video_features(
            model=model,
            detector=detector,
            video_path=PROJECT_ROOT / str(row["video_path"]),
            max_frames=max_frames,
            frame_stride=frame_stride,
            imgsz=imgsz,
            confidence_threshold=confidence_threshold,
            prediction_batch_size=prediction_batch_size,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        payload = {
            "scene_id": _safe_id(row),
            "video_path": row.get("video_path"),
            "is_fall": bool(row.get("is_fall")),
            "max_frames": max_frames,
            "frame_stride": frame_stride,
            "sampling_start_frame": start_frame,
            "sampling_end_frame": end_frame,
            "fall_window_margin_frames": fall_window_margin_frames,
            **summary,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_dataset(
    rows: list[dict[str, Any]],
    *,
    feature_cache: Path,
    max_frames: int,
    frame_stride: int,
    fall_window_margin_frames: int,
    min_pose_frames: int,
) -> dict[str, Any]:
    features: list[list[float]] = []
    labels: list[int] = []
    scene_ids: list[str] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        path = _feature_path(
            feature_cache,
            row,
            max_frames,
            frame_stride,
            fall_window_margin_frames,
        )
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
        summary = _summarize_frames(
            list(payload.get("frame_records") or []),
            int(payload.get("frames_seen") or 0),
        )
        features.append([float(value) for value in summary["feature_vector"]])
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


def _dataset_summary(scene_ids: list[str], labels: np.ndarray) -> dict[str, Any]:
    group_labels: dict[str, set[int]] = {}
    for scene_id, label in zip(scene_ids, labels.tolist()):
        group_labels.setdefault(_scene_base(scene_id), set()).add(int(label))
    return {
        "scene_ids": scene_ids,
        "groups": len(group_labels),
        "group_class_counts": {
            "fall": sum(FALL_LABEL in values for values in group_labels.values()),
            "non_fall": sum(NON_FALL_LABEL in values for values in group_labels.values()),
        },
    }


def _hard_case_sample_weights(
    labels: np.ndarray,
    fall_probabilities: np.ndarray,
    *,
    hard_case_weight: float,
) -> np.ndarray:
    predicted_labels = (fall_probabilities >= 0.5).astype(np.int64)
    sample_weights = np.ones(len(labels), dtype=np.float64)
    sample_weights[predicted_labels != labels] = max(hard_case_weight, 1.0)
    return sample_weights


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


def _build_model_bundle(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bundle_schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "model_kind": "yolo_pose_summary_rf",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "fall_class_label": FALL_LABEL,
        "model": model,
        "inference_config": {
            "max_frames": args.max_frames,
            "frame_stride": args.frame_stride,
            "imgsz": args.imgsz,
            "confidence_threshold": args.confidence_threshold,
            "candidate_window_frames": args.candidate_window_frames,
            "candidate_window_seconds": args.candidate_window_seconds,
        },
        "training_config": {
            "prediction_batch_size": args.prediction_batch_size,
            "min_pose_frames": args.min_pose_frames,
            "decision_threshold": args.decision_threshold,
        },
    }


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
    parser.add_argument("--dataset-version", default="yolo_pose_fall_rf")
    parser.add_argument("--max-videos", type=int, default=200)
    parser.add_argument("--validation-max-videos", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--fall-window-margin-frames", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--candidate-window-frames", type=int, default=0)
    parser.add_argument("--candidate-window-seconds", type=float, default=0.0)
    parser.add_argument("--prediction-batch-size", type=int, default=0)
    parser.add_argument("--min-pose-frames", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--hard-case-weight", type=float, default=1.0)
    parser.add_argument("--hard-case-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=0.6)
    parser.add_argument(
        "--classifier",
        choices=("random_forest", "extra_trees"),
        default="random_forest",
    )
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> int:
    import joblib
    from sklearn.base import clone
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict

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
        prediction_batch_size=args.prediction_batch_size,
        fall_window_margin_frames=args.fall_window_margin_frames,
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
        prediction_batch_size=args.prediction_batch_size,
        fall_window_margin_frames=args.fall_window_margin_frames,
        force_extract=args.force_extract,
        label="validation",
    )

    train_dataset = _load_dataset(
        train_rows,
        feature_cache=args.feature_cache,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        fall_window_margin_frames=args.fall_window_margin_frames,
        min_pose_frames=args.min_pose_frames,
    )
    validation_dataset = _load_dataset(
        validation_rows,
        feature_cache=args.validation_feature_cache,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        fall_window_margin_frames=args.fall_window_margin_frames,
        min_pose_frames=args.min_pose_frames,
    )
    class_count = len(set(train_dataset["y"].tolist()))
    if class_count < 2:
        raise SystemExit(f"need both fall and non-fall classes, got {_class_counts(train_dataset['y'])}")
    train_indices, holdout_indices, holdout_split = _group_holdout_indices(
        train_dataset["scene_ids"],
        train_dataset["y"],
        test_size=0.25,
        random_state=args.random_state,
    )
    x_train = train_dataset["x"][train_indices]
    x_holdout = train_dataset["x"][holdout_indices]
    y_train = train_dataset["y"][train_indices]
    y_holdout = train_dataset["y"][holdout_indices]
    ids_holdout = [train_dataset["scene_ids"][index] for index in holdout_indices]
    classifier_type = (
        ExtraTreesClassifier if args.classifier == "extra_trees" else RandomForestClassifier
    )
    model = classifier_type(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    sample_weights = np.ones(len(y_train), dtype=np.float64)
    hard_case_mining: dict[str, Any] = {
        "enabled": False,
        "weight": args.hard_case_weight,
        "scene_ids": [],
    }
    if args.hard_case_weight > 1.0:
        train_groups = np.asarray(
            [_scene_base(train_dataset["scene_ids"][index]) for index in train_indices],
            dtype=object,
        )
        fold_count = min(args.hard_case_folds, len(set(train_groups.tolist())))
        if fold_count < 2:
            raise SystemExit("hard-case mining needs at least two scene groups")
        oof_probabilities = cross_val_predict(
            clone(model),
            x_train,
            y_train,
            groups=train_groups,
            cv=GroupKFold(n_splits=fold_count),
            method="predict_proba",
            n_jobs=1,
        )
        fold_class_labels = sorted(set(y_train.tolist()))
        fall_index = fold_class_labels.index(FALL_LABEL)
        fall_probabilities = oof_probabilities[:, fall_index]
        sample_weights = _hard_case_sample_weights(
            y_train,
            fall_probabilities,
            hard_case_weight=args.hard_case_weight,
        )
        hard_indices = np.flatnonzero(sample_weights > 1.0)
        hard_case_mining = {
            "enabled": True,
            "weight": args.hard_case_weight,
            "folds": fold_count,
            "count": int(len(hard_indices)),
            "scene_ids": [
                train_dataset["scene_ids"][train_indices[index]]
                for index in hard_indices
            ],
        }
    model.fit(x_train, y_train, sample_weight=sample_weights)
    holdout_dataset = {"x": x_holdout, "y": y_holdout, "scene_ids": ids_holdout}
    thresholds = [round(value, 2) for value in np.arange(0.35, 0.91, 0.05)]
    holdout_evaluation = _evaluate(model, holdout_dataset, threshold=args.decision_threshold)
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": args.dataset_version,
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
        "dataset_summary": _dataset_summary(
            train_dataset["scene_ids"], train_dataset["y"]
        ),
        "validation_class_counts": _class_counts(validation_dataset["y"]),
        "excluded": train_dataset["excluded"],
        "validation_excluded": validation_dataset["excluded"],
        "model_params": {
            "classifier": args.classifier,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "hard_case_weight": args.hard_case_weight,
            "hard_case_folds": args.hard_case_folds,
            "decision_threshold": args.decision_threshold,
            "max_frames": args.max_frames,
            "frame_stride": args.frame_stride,
            "imgsz": args.imgsz,
            "confidence_threshold": args.confidence_threshold,
            "prediction_batch_size": args.prediction_batch_size,
            "min_pose_frames": args.min_pose_frames,
        },
        "holdout_method": holdout_split["method"],
        "holdout_split": holdout_split,
        "hard_case_mining": hard_case_mining,
        "holdout": holdout_evaluation,
        "holdout_errors": holdout_evaluation["errors"],
        "validation": _evaluate(model, validation_dataset, threshold=args.decision_threshold),
        "validation_threshold_sweep": _threshold_sweep(model, validation_dataset, thresholds),
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_build_model_bundle(model, args), args.output_model)
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
