"""Commercially deployable YuNet/SFace TensorRT primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ._attribute_runtimes import build_tensorrt_runtime


SFACE_MODEL_ID = "opencv-sface-tensorrt-v1"
SFACE_INPUT_SIZE = (112, 112)


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
