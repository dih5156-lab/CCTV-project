from pathlib import Path

import numpy as np
import pytest

from src.core.ai._face_tensorrt import (
    TensorRTFaceEmbedder,
    normalize_embedding,
    preprocess_arcface_bgr,
)


class FakeRuntime:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    def run(self, tensor):
        self.inputs.append(tensor)
        return [self.output]


def test_preprocess_arcface_bgr_rejects_empty_image():
    with pytest.raises(ValueError, match="non-empty BGR image"):
        preprocess_arcface_bgr(np.empty((0, 0, 3), dtype=np.uint8))


def test_preprocess_arcface_bgr_returns_normalized_nchw_tensor():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    tensor = preprocess_arcface_bgr(image)

    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert np.allclose(tensor, -1.0)


def test_normalize_embedding_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_embedding(np.zeros(512, dtype=np.float32))


def test_embed_aligned_returns_l2_normalized_512_vector():
    runtime = FakeRuntime(np.ones((1, 512), dtype=np.float32))
    embedder = TensorRTFaceEmbedder(
        Path("face.engine"), runtime_factory=lambda _: runtime
    )

    embedding = embedder.embed_aligned(
        np.zeros((112, 112, 3), dtype=np.uint8)
    )

    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0)
    assert runtime.inputs[0].shape == (1, 3, 112, 112)


def test_embed_aligned_rejects_unexpected_output_shape():
    runtime = FakeRuntime(np.ones((1, 128), dtype=np.float32))
    embedder = TensorRTFaceEmbedder(
        Path("face.engine"), runtime_factory=lambda _: runtime
    )

    with pytest.raises(ValueError, match="512 values"):
        embedder.embed_aligned(np.zeros((112, 112, 3), dtype=np.uint8))
