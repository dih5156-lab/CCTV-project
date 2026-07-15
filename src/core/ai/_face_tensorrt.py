"""ONNX Runtime 없이 ArcFace TensorRT 임베딩을 실행한다."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ._attribute_runtimes import build_tensorrt_runtime

MODEL_ID = "arcface-w600k-r50-tensorrt-v1"
INPUT_SIZE = (112, 112)


def preprocess_arcface_bgr(image: np.ndarray) -> np.ndarray:
    """BGR 얼굴 이미지를 ArcFace RGB NCHW 입력으로 변환한다."""
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("ArcFace input must be a non-empty BGR image")

    resized = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - 127.5) / 127.5
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """얼굴 임베딩을 cosine similarity 비교용 단위 벡터로 만든다."""
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    if norm <= 0.0:
        raise ValueError("ArcFace produced a zero-norm embedding")
    return np.ascontiguousarray(flattened / norm)


class TensorRTFaceEmbedder:
    """고정 112x112 ArcFace TensorRT engine 실행기."""

    def __init__(
        self,
        model_path: Path,
        runtime_factory: Callable[[Path], object] = build_tensorrt_runtime,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_id = MODEL_ID
        self._runtime = runtime_factory(self.model_path)

    def embed_aligned(self, image: np.ndarray) -> np.ndarray:
        outputs = self._runtime.run(preprocess_arcface_bgr(image))
        if len(outputs) != 1:
            raise ValueError(f"ArcFace expected one output, received {len(outputs)}")

        raw_embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if raw_embedding.size != 512:
            raise ValueError(
                "ArcFace output must contain 512 values, "
                f"received {raw_embedding.size}"
            )
        return normalize_embedding(raw_embedding)
