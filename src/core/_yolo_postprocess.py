"""YOLO tensor 후처리 유틸리티."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


def map_yolo_box_to_frame(
    box: Any,
    frame_width: int,
    frame_height: int,
    *,
    input_size: float,
) -> Tuple[int, int, int, int]:
    """letterbox 기준 YOLO bbox를 원본 프레임 좌표로 복원한다."""
    x_center, y_center, width, height = [float(value) for value in box]
    gain = min(input_size / max(frame_width, 1), input_size / max(frame_height, 1))
    pad_x = (input_size - frame_width * gain) / 2.0
    pad_y = (input_size - frame_height * gain) / 2.0

    left = (x_center - width / 2.0 - pad_x) / gain
    top = (y_center - height / 2.0 - pad_y) / gain
    right = (x_center + width / 2.0 - pad_x) / gain
    bottom = (y_center + height / 2.0 - pad_y) / gain

    left = max(0, min(frame_width - 1, int(round(left))))
    top = max(0, min(frame_height - 1, int(round(top))))
    right = max(left + 1, min(frame_width, int(round(right))))
    bottom = max(top + 1, min(frame_height, int(round(bottom))))
    return left, top, right - left, bottom - top


def map_yolo_keypoints_to_frame(
    values: Any,
    frame_width: int,
    frame_height: int,
    *,
    input_size: float,
) -> List[List[float]]:
    """letterbox 기준 YOLO pose keypoint를 원본 프레임 좌표로 복원한다."""
    keypoints = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    gain = min(input_size / max(frame_width, 1), input_size / max(frame_height, 1))
    pad_x = (input_size - frame_width * gain) / 2.0
    pad_y = (input_size - frame_height * gain) / 2.0

    mapped: List[List[float]] = []
    for x_value, y_value, confidence in keypoints:
        x = (float(x_value) - pad_x) / gain
        y = (float(y_value) - pad_y) / gain
        mapped.append(
            [
                float(max(0.0, min(frame_width - 1, x))),
                float(max(0.0, min(frame_height - 1, y))),
                float(confidence),
            ]
        )
    return mapped


def nms_detections(
    detections: List[Dict[str, Any]],
    *,
    iou_threshold: float,
    max_detections: int,
) -> List[Dict[str, Any]]:
    """class별 NMS를 수행하고 confidence 내림차순으로 반환한다."""
    if not detections:
        return []

    kept: List[Dict[str, Any]] = []
    by_class: Dict[int, List[Dict[str, Any]]] = {}
    for detection in detections:
        by_class.setdefault(int(detection["class_id"]), []).append(detection)

    for class_detections in by_class.values():
        class_detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
        boxes = np.array([item["box"] for item in class_detections], dtype=np.float32)
        scores = np.array([item["confidence"] for item in class_detections], dtype=np.float32)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        while order.size > 0 and len(kept) < max_detections:
            idx = int(order[0])
            kept.append(class_detections[idx])
            if order.size == 1:
                break
            xx1 = np.maximum(x1[idx], x1[order[1:]])
            yy1 = np.maximum(y1[idx], y1[order[1:]])
            xx2 = np.minimum(x2[idx], x2[order[1:]])
            yy2 = np.minimum(y2[idx], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[idx] + areas[order[1:]] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            order = order[1:][iou <= iou_threshold]

    kept.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return kept[:max_detections]


def detections_from_yolo_output(
    output: Any,
    *,
    task: str,
    gie_id: int,
    labels: List[str],
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
    class_ids_filter: Optional[Set[int]],
    input_size: float,
    iou_threshold: float,
    max_detections: int,
    fall_checker: Callable[[List[List[float]], int, int], bool],
    person_pose_validator: Callable[[List[List[float]]], bool],
) -> List[Dict[str, Any]]:
    """YOLO raw output 배열을 DetectionEvent 생성 전 dict 목록으로 변환한다."""
    rows = np.asarray(output, dtype=np.float32)
    if rows.ndim == 3:
        rows = rows[0]
    if rows.ndim != 2:
        return []
    if rows.shape[0] < rows.shape[1]:
        rows = rows.T
    if rows.shape[1] < 5:
        return []

    if task == "pose":
        class_ids = np.zeros(rows.shape[0], dtype=np.int32)
        confidences = rows[:, 4]
    else:
        class_scores = rows[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(rows.shape[0]), class_ids]

    detections: List[Dict[str, Any]] = []
    for row, class_id, confidence in zip(rows, class_ids, confidences):
        class_id = int(class_id)
        confidence = float(confidence)
        if confidence < confidence_threshold:
            continue
        if class_ids_filter and class_id not in class_ids_filter:
            continue

        x, y, width, height = map_yolo_box_to_frame(
            row[:4],
            frame_width,
            frame_height,
            input_size=input_size,
        )
        label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
        keypoints = None
        is_fall = False
        if task == "pose" and row.shape[0] >= 56:
            keypoints = map_yolo_keypoints_to_frame(
                row[5:56],
                frame_width,
                frame_height,
                input_size=input_size,
            )
            is_fall = fall_checker(keypoints, width, height)
            if not is_fall and not person_pose_validator(keypoints):
                continue

        detections.append(
            {
                "box": (x, y, width, height),
                "confidence": confidence,
                "class_id": class_id,
                "label": label,
                "keypoints": keypoints,
                "is_fall": is_fall,
                "gie_id": gie_id,
                "task": task,
            }
        )
    return nms_detections(
        detections,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
