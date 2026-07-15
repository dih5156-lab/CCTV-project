from pathlib import Path

import numpy as np
import pytest

from src.core.ai._commercial_face_tensorrt import (
    TensorRTSFaceEmbedder,
    normalize_sface_embedding,
    preprocess_sface_bgr,
)


class FakeRuntime:
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = []

    def run(self, tensor):
        self.inputs.append(tensor)
        return self.outputs


def test_preprocess_sface_rejects_empty_image():
    with pytest.raises(ValueError, match="non-empty BGR image"):
        preprocess_sface_bgr(np.empty((0, 0, 3), dtype=np.uint8))


def test_preprocess_sface_matches_official_rgb_float_blob():
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    tensor = preprocess_sface_bgr(image)

    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert tensor[0, :, 0, 0].tolist() == [30.0, 20.0, 10.0]


def test_normalize_sface_embedding_rejects_non_finite_values():
    vector = np.ones(128, dtype=np.float32)
    vector[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        normalize_sface_embedding(vector)


def test_normalize_sface_embedding_rejects_zero_norm():
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_sface_embedding(np.zeros(128, dtype=np.float32))


def test_embed_aligned_returns_normalized_128_vector():
    runtime = FakeRuntime([np.ones((1, 128), dtype=np.float32)])
    embedder = TensorRTSFaceEmbedder(
        Path("sface.engine"), runtime_factory=lambda _: runtime
    )

    embedding = embedder.embed_aligned(np.zeros((112, 112, 3), dtype=np.uint8))

    assert embedding.shape == (128,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0)
    assert runtime.inputs[0].shape == (1, 3, 112, 112)
    assert embedder.model_id == "opencv-sface-tensorrt-v1"


@pytest.mark.parametrize(
    "outputs, message",
    [
        ([], "one output"),
        ([np.ones((1, 128)), np.ones((1, 128))], "one output"),
        ([np.ones((1, 512))], "128 values"),
    ],
)
def test_embed_aligned_rejects_unexpected_outputs(outputs, message):
    embedder = TensorRTSFaceEmbedder(
        Path("sface.engine"), runtime_factory=lambda _: FakeRuntime(outputs)
    )

    with pytest.raises(ValueError, match=message):
        embedder.embed_aligned(np.zeros((112, 112, 3), dtype=np.uint8))
