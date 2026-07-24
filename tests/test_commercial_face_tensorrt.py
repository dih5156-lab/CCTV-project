from pathlib import Path

import numpy as np
import pytest

from src.core.ai._commercial_face_tensorrt import (
    SFACE_LANDMARK_TEMPLATE,
    CommercialFaceEmbeddingPipeline,
    TensorRTSFaceEmbedder,
    TensorRTYuNetDetector,
    YuNetFace,
    align_sface_bgr,
    decode_yunet_outputs,
    normalize_sface_embedding,
    preprocess_sface_bgr,
    preprocess_yunet_bgr,
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


def _empty_yunet_outputs():
    outputs = {}
    for stride, count in ((8, 6400), (16, 1600), (32, 400)):
        outputs[f"cls_{stride}"] = np.zeros((1, count, 1), dtype=np.float32)
        outputs[f"obj_{stride}"] = np.zeros((1, count, 1), dtype=np.float32)
        outputs[f"bbox_{stride}"] = np.zeros((1, count, 4), dtype=np.float32)
        outputs[f"kps_{stride}"] = np.zeros((1, count, 10), dtype=np.float32)
    return outputs


def test_preprocess_yunet_returns_fixed_bgr_float_blob():
    image = np.zeros((320, 160, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    tensor = preprocess_yunet_bgr(image)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert tensor[0, :, 0, 0].tolist() == [10.0, 20.0, 30.0]


def test_decode_yunet_restores_bbox_and_landmarks_to_frame_coordinates():
    outputs = _empty_yunet_outputs()
    # stride 8, grid row=2, col=3
    index = 2 * 80 + 3
    outputs["cls_8"][0, index, 0] = 0.81
    outputs["obj_8"][0, index, 0] = 1.0
    outputs["bbox_8"][0, index] = [0.5, 0.5, 0.0, 0.0]
    outputs["kps_8"][0, index] = [0, 0, 1, 0, 0.5, 0.5, 0, 1, 1, 1]

    faces = decode_yunet_outputs(
        outputs,
        roi=(100, 50, 320, 160),
        score_threshold=0.6,
        nms_threshold=0.3,
    )

    assert len(faces) == 1
    face = faces[0]
    assert face.score == pytest.approx(0.9)
    assert face.bbox == pytest.approx((112.0, 54.0, 4.0, 2.0))
    assert np.asarray(face.landmarks) == pytest.approx(
        np.asarray(((112.0, 54.0), (116.0, 54.0), (114.0, 55.0), (112.0, 56.0), (116.0, 56.0)))
    )


def test_decode_yunet_filters_low_score_after_clamping():
    outputs = _empty_yunet_outputs()
    outputs["cls_32"][0, 0, 0] = 2.0
    outputs["obj_32"][0, 0, 0] = 0.25

    assert decode_yunet_outputs(outputs, roi=(0, 0, 640, 640), score_threshold=0.6) == []


def test_decode_yunet_applies_nms():
    outputs = _empty_yunet_outputs()
    for index, score in ((0, 0.9), (1, 0.8)):
        outputs["cls_32"][0, index, 0] = score * score
        outputs["obj_32"][0, index, 0] = 1.0
        outputs["bbox_32"][0, index] = [0.5 - index, 0.5, 1.0, 1.0]

    faces = decode_yunet_outputs(
        outputs, roi=(0, 0, 640, 640), score_threshold=0.6, nms_threshold=0.3
    )

    assert len(faces) == 1
    assert faces[0].score == pytest.approx(0.9)


def test_decode_yunet_clamps_bbox_and_landmarks_to_roi():
    outputs = _empty_yunet_outputs()
    outputs["cls_32"][0, 0, 0] = 1.0
    outputs["obj_32"][0, 0, 0] = 1.0
    outputs["bbox_32"][0, 0] = [0.0, 0.0, 1.0, 1.0]
    outputs["kps_32"][0, 0] = [-1, -1, 30, 30, 0, 0, 0, 0, 0, 0]

    face = decode_yunet_outputs(outputs, roi=(10, 20, 100, 50))[0]

    x, y, width, height = face.bbox
    assert x == 10
    assert y == 20
    assert width > 0
    assert height > 0
    assert face.landmarks[0] == (10, 20)
    assert face.landmarks[1] == (110, 70)


def test_decode_yunet_rejects_missing_or_bad_output_shape():
    outputs = _empty_yunet_outputs()
    del outputs["kps_16"]
    with pytest.raises(ValueError, match="missing YuNet outputs"):
        decode_yunet_outputs(outputs, roi=(0, 0, 640, 640))

    outputs = _empty_yunet_outputs()
    outputs["bbox_8"] = np.zeros((1, 1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="bbox_8 shape"):
        decode_yunet_outputs(outputs, roi=(0, 0, 640, 640))


def test_align_sface_is_identity_for_official_template():
    image = np.arange(112 * 112 * 3, dtype=np.uint8).reshape(112, 112, 3)

    aligned = align_sface_bgr(image, SFACE_LANDMARK_TEMPLATE)

    assert aligned.shape == (112, 112, 3)
    assert np.max(np.abs(aligned.astype(np.int16) - image.astype(np.int16))) <= 1


def test_align_sface_maps_scaled_landmarks_to_official_template():
    landmarks = SFACE_LANDMARK_TEMPLATE * 2.0 + np.array([10.0, 20.0])
    source = np.zeros((260, 260, 3), dtype=np.uint8)

    aligned = align_sface_bgr(source, landmarks)

    assert aligned.shape == (112, 112, 3)


@pytest.mark.parametrize(
    "landmarks, message",
    [
        (np.zeros((4, 2), dtype=np.float32), "five landmarks"),
        (np.full((5, 2), np.nan, dtype=np.float32), "finite"),
        (np.ones((5, 2), dtype=np.float32), "degenerate"),
    ],
)
def test_align_sface_rejects_invalid_landmarks(landmarks, message):
    with pytest.raises(ValueError, match=message):
        align_sface_bgr(np.zeros((112, 112, 3), dtype=np.uint8), landmarks)


def test_yunet_detector_runs_named_runtime_on_clamped_roi():
    outputs = _empty_yunet_outputs()
    outputs["cls_32"][0, 0, 0] = 1.0
    outputs["obj_32"][0, 0, 0] = 1.0

    class FakeNamedRuntime:
        def __init__(self):
            self.inputs = []

        def run_named(self, tensor):
            self.inputs.append(tensor)
            return outputs

    runtime = FakeNamedRuntime()
    detector = TensorRTYuNetDetector(
        Path("yunet.engine"), runtime_factory=lambda _: runtime
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    faces = detector.detect(frame, roi=(-10, -20, 110, 80))

    assert len(faces) == 1
    assert runtime.inputs[0].shape == (1, 3, 640, 640)
    assert faces[0].bbox[0] >= 0
    assert faces[0].bbox[1] >= 0


def test_yunet_detector_rejects_roi_outside_frame():
    detector = TensorRTYuNetDetector(
        Path("yunet.engine"), runtime_factory=lambda _: FakeRuntime([])
    )

    with pytest.raises(ValueError, match="does not intersect"):
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), (200, 200, 10, 10))


def test_commercial_pipeline_aligns_and_embeds_each_detected_face():
    face = YuNetFace(
        bbox=(10.0, 20.0, 80.0, 90.0),
        landmarks=tuple(map(tuple, SFACE_LANDMARK_TEMPLATE)),
        score=0.95,
    )

    class FakeDetector:
        def detect(self, frame, roi):
            return [face]

    class FakeEmbedder:
        model_id = "opencv-sface-tensorrt-v1"

        def __init__(self):
            self.images = []

        def embed_aligned(self, image):
            self.images.append(image)
            return np.ones(128, dtype=np.float32) / np.sqrt(128)

    embedder = FakeEmbedder()
    pipeline = CommercialFaceEmbeddingPipeline(FakeDetector(), embedder)

    results = pipeline.extract_embeddings(
        np.zeros((112, 112, 3), dtype=np.uint8), (0, 0, 112, 112)
    )

    assert len(results) == 1
    assert results[0].face == face
    assert results[0].embedding.shape == (128,)
    assert results[0].model_id == "opencv-sface-tensorrt-v1"
    assert embedder.images[0].shape == (112, 112, 3)
