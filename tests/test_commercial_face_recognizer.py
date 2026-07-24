from types import SimpleNamespace

import numpy as np
import pytest

from src.core.ai._commercial_face_recognizer import CommercialFaceRecognizer


class FakeEmbeddingPipeline:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def extract_embeddings(self, frame, roi):
        self.calls.append(roi)
        return self.results


class FakeGallery:
    def __init__(self, search_result):
        self.search_result = search_result
        self.queries = []

    def search(self, embedding, **kwargs):
        self.queries.append((embedding, kwargs))
        return self.search_result


def _embedded_face(*, bbox=(10.2, 20.4, 30.6, 40.8), score=0.95):
    face = SimpleNamespace(
        bbox=bbox,
        score=score,
    )
    return SimpleNamespace(face=face, embedding=np.ones(128, dtype=np.float32))


def test_recognizer_returns_gallery_identity_in_existing_event_contract():
    candidate = SimpleNamespace(
        person_id="employee-1",
        name="tester",
        category="employee",
        similarity=0.82,
    )
    search_result = SimpleNamespace(
        matched=True,
        decision="matched",
        best=candidate,
        second_best_similarity=0.3,
        margin=0.52,
        model_id="opencv-sface-tensorrt-v1",
    )
    pipeline = FakeEmbeddingPipeline([_embedded_face()])
    gallery = FakeGallery(search_result)
    recognizer = CommercialFaceRecognizer(pipeline, gallery)

    results = recognizer.detect_and_recognize(
        np.zeros((200, 200, 3), dtype=np.uint8),
        {"x": 0, "y": 0, "width": 100, "height": 180},
    )

    assert recognizer.enabled is True
    assert recognizer.backend_name == "opencv_yunet_sface_tensorrt"
    assert len(results) == 1
    assert results[0].matched is True
    assert results[0].label == "tester"
    assert results[0].confidence == pytest.approx(0.82)
    assert results[0].person_id == "employee-1"
    assert results[0].decision == "matched"
    assert pipeline.calls[0] == (0, 0, 100, 108)


def test_recognizer_keeps_ambiguous_candidate_unknown():
    candidate = SimpleNamespace(
        person_id="a", name="alpha", category="employee", similarity=0.7
    )
    search_result = SimpleNamespace(
        matched=False,
        decision="ambiguous",
        best=candidate,
        second_best_similarity=0.68,
        margin=0.02,
        model_id="opencv-sface-tensorrt-v1",
    )
    recognizer = CommercialFaceRecognizer(
        FakeEmbeddingPipeline([_embedded_face()]), FakeGallery(search_result)
    )

    result = recognizer.detect_and_recognize(
        np.zeros((200, 200, 3), dtype=np.uint8),
        {"x": 0, "y": 0, "width": 100, "height": 100},
    )[0]

    assert result.matched is False
    assert result.label == "unknown"
    assert result.decision == "ambiguous"


def test_enrollment_requires_exactly_one_detected_face():
    recognizer = CommercialFaceRecognizer(
        FakeEmbeddingPipeline([]), FakeGallery(None)
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="exactly one face"):
        recognizer.extract_enrollment_embedding(image)

    recognizer.embedding_pipeline.results = [_embedded_face(), _embedded_face()]
    with pytest.raises(ValueError, match="exactly one face"):
        recognizer.extract_enrollment_embedding(image)


def test_enrollment_returns_single_embedding():
    recognizer = CommercialFaceRecognizer(
        FakeEmbeddingPipeline([_embedded_face()]), FakeGallery(None)
    )

    embedding = recognizer.extract_enrollment_embedding(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )

    assert embedding.shape == (128,)


def test_enrollment_ignores_small_low_confidence_false_positive():
    primary_face = _embedded_face(
        bbox=(20.0, 10.0, 60.0, 75.0),
        score=0.96,
    )
    false_positive = _embedded_face(
        bbox=(2.0, 2.0, 8.0, 8.0),
        score=0.63,
    )
    recognizer = CommercialFaceRecognizer(
        FakeEmbeddingPipeline([primary_face, false_positive]), FakeGallery(None)
    )

    embedding = recognizer.extract_enrollment_embedding(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )

    assert np.array_equal(embedding, primary_face.embedding)


def test_enrollment_still_rejects_two_credible_faces():
    recognizer = CommercialFaceRecognizer(
        FakeEmbeddingPipeline(
            [
                _embedded_face(bbox=(5.0, 10.0, 40.0, 50.0), score=0.95),
                _embedded_face(bbox=(52.0, 12.0, 38.0, 48.0), score=0.91),
            ]
        ),
        FakeGallery(None),
    )

    with pytest.raises(ValueError, match="exactly one face"):
        recognizer.extract_enrollment_embedding(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )


def test_recognizer_ignores_person_bbox_outside_frame():
    pipeline = FakeEmbeddingPipeline([])
    recognizer = CommercialFaceRecognizer(pipeline, FakeGallery(None))

    results = recognizer.detect_and_recognize(
        np.zeros((100, 100, 3), dtype=np.uint8),
        {"x": 150, "y": 150, "width": 20, "height": 20},
    )

    assert results == []
    assert pipeline.calls == []
