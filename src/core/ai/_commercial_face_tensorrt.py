"""Commercially deployable YuNet/SFace TensorRT primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from ._attribute_runtimes import build_tensorrt_runtime


SFACE_MODEL_ID = "opencv-sface-tensorrt-v1"
SFACE_INPUT_SIZE = (112, 112)
SFACE_LANDMARK_TEMPLATE = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
YUNET_INPUT_SIZE = (640, 640)
YUNET_STRIDES = (8, 16, 32)
YUNET_COUNTS = {8: 6400, 16: 1600, 32: 400}


@dataclass(frozen=True)
class YuNetFace:
    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    score: float


def preprocess_yunet_bgr(image: np.ndarray) -> np.ndarray:
    """Resize a BGR ROI to the fixed YuNet TensorRT input."""
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("YuNet input must be a non-empty BGR image")
    resized = cv2.resize(image, YUNET_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(
        resized.astype(np.float32).transpose(2, 0, 1)[None, ...]
    )


def _validate_yunet_outputs(outputs: dict[str, np.ndarray]) -> None:
    expected_names = {
        f"{prefix}_{stride}"
        for prefix in ("cls", "obj", "bbox", "kps")
        for stride in YUNET_STRIDES
    }
    missing = expected_names - set(outputs)
    if missing:
        raise ValueError(f"missing YuNet outputs: {sorted(missing)}")
    unexpected = set(outputs) - expected_names
    if unexpected:
        raise ValueError(f"unexpected YuNet outputs: {sorted(unexpected)}")
    widths = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}
    for stride in YUNET_STRIDES:
        for prefix, width in widths.items():
            name = f"{prefix}_{stride}"
            expected_shape = (1, YUNET_COUNTS[stride], width)
            if tuple(np.asarray(outputs[name]).shape) != expected_shape:
                raise ValueError(
                    f"unexpected {name} shape: {np.asarray(outputs[name]).shape}, "
                    f"expected {expected_shape}"
                )


def decode_yunet_outputs(
    outputs: dict[str, np.ndarray],
    *,
    roi: tuple[float, float, float, float],
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> list[YuNetFace]:
    """Decode official YuNet tensors and restore ROI coordinates to the frame."""
    _validate_yunet_outputs(outputs)
    roi_x, roi_y, roi_width, roi_height = (float(value) for value in roi)
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("YuNet ROI width and height must be positive")
    scale_x = roi_width / YUNET_INPUT_SIZE[0]
    scale_y = roi_height / YUNET_INPUT_SIZE[1]

    candidates: list[YuNetFace] = []
    model_boxes: list[list[float]] = []
    scores: list[float] = []
    for stride in YUNET_STRIDES:
        columns = YUNET_INPUT_SIZE[0] // stride
        cls_values = np.clip(outputs[f"cls_{stride}"].reshape(-1), 0.0, 1.0)
        obj_values = np.clip(outputs[f"obj_{stride}"].reshape(-1), 0.0, 1.0)
        combined_scores = np.sqrt(cls_values * obj_values)
        selected = np.flatnonzero(combined_scores >= score_threshold)
        bbox_values = outputs[f"bbox_{stride}"].reshape(-1, 4)
        landmark_values = outputs[f"kps_{stride}"].reshape(-1, 10)
        for index in selected:
            row, column = divmod(int(index), columns)
            bbox = bbox_values[index]
            center_x = (column + float(bbox[0])) * stride
            center_y = (row + float(bbox[1])) * stride
            width = float(np.exp(bbox[2])) * stride
            height = float(np.exp(bbox[3])) * stride
            model_x = center_x - width / 2.0
            model_y = center_y - height / 2.0
            frame_x1 = max(roi_x, roi_x + model_x * scale_x)
            frame_y1 = max(roi_y, roi_y + model_y * scale_y)
            frame_x2 = min(roi_x + roi_width, roi_x + (model_x + width) * scale_x)
            frame_y2 = min(roi_y + roi_height, roi_y + (model_y + height) * scale_y)
            if frame_x2 <= frame_x1 or frame_y2 <= frame_y1:
                continue
            frame_bbox = (
                frame_x1,
                frame_y1,
                frame_x2 - frame_x1,
                frame_y2 - frame_y1,
            )
            raw_landmarks = landmark_values[index]
            landmarks = tuple(
                (
                    float(np.clip(
                        roi_x + (column + float(raw_landmarks[point * 2])) * stride * scale_x,
                        roi_x,
                        roi_x + roi_width,
                    )),
                    float(np.clip(
                        roi_y + (row + float(raw_landmarks[point * 2 + 1])) * stride * scale_y,
                        roi_y,
                        roi_y + roi_height,
                    )),
                )
                for point in range(5)
            )
            candidates.append(
                YuNetFace(
                    bbox=frame_bbox,
                    landmarks=landmarks,
                    score=float(combined_scores[index]),
                )
            )
            model_boxes.append([model_x, model_y, width, height])
            scores.append(float(combined_scores[index]))

    if not candidates:
        return []
    kept = cv2.dnn.NMSBoxes(
        model_boxes, scores, score_threshold, nms_threshold, top_k=top_k
    )
    return [candidates[int(index)] for index in np.asarray(kept).reshape(-1)]


def preprocess_sface_bgr(image: np.ndarray) -> np.ndarray:
    """Match OpenCV FaceRecognizerSF's RGB float32 112x112 input blob."""
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("SFace input must be a non-empty BGR image")

    resized = cv2.resize(image, SFACE_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])


def _similarity_transform(
    source_points: np.ndarray, destination_points: np.ndarray
) -> np.ndarray:
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("SFace landmarks form a degenerate transform")

    covariance = destination_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_transposed = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_transposed) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_transposed
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("SFace landmarks form a degenerate transform")
    translation = destination_mean - scale * (rotation @ source_mean)
    return np.column_stack((scale * rotation, translation)).astype(np.float32)


def align_sface_bgr(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Align YuNet's ordered five landmarks to OpenCV's SFace template."""
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("SFace alignment input must be a non-empty BGR image")
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2):
        raise ValueError("SFace alignment requires exactly five landmarks")
    if not np.all(np.isfinite(points)):
        raise ValueError("SFace landmarks must contain only finite values")
    transform = _similarity_transform(points, SFACE_LANDMARK_TEMPLATE)
    return cv2.warpAffine(
        image,
        transform,
        SFACE_INPUT_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def normalize_sface_embedding(vector: np.ndarray) -> np.ndarray:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(flattened)):
        raise ValueError("SFace embedding must contain only finite values")
    norm = float(np.linalg.norm(flattened))
    if norm <= 0.0:
        raise ValueError("SFace produced a zero-norm embedding")
    return np.ascontiguousarray(flattened / norm)


class TensorRTSFaceEmbedder:
    """Fixed-shape SFace TensorRT embedding adapter."""

    def __init__(
        self,
        model_path: Path,
        runtime_factory: Callable[[Path], object] = build_tensorrt_runtime,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_id = SFACE_MODEL_ID
        self._runtime = runtime_factory(self.model_path)

    def embed_aligned(self, image: np.ndarray) -> np.ndarray:
        outputs = self._runtime.run(preprocess_sface_bgr(image))
        if len(outputs) != 1:
            raise ValueError(f"SFace expected one output, received {len(outputs)}")
        raw_embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if raw_embedding.size != 128:
            raise ValueError(
                "SFace output must contain 128 values, "
                f"received {raw_embedding.size}"
            )
        return normalize_sface_embedding(raw_embedding)
