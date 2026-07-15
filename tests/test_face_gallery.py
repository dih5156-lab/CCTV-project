import numpy as np
import pytest

from src.core.ai._face_gallery import InMemoryFaceGallery


def _unit(index: int, dimensions: int = 128) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_gallery_enrolls_one_or_multiple_samples_per_person():
    gallery = InMemoryFaceGallery(model_id="opencv-sface-tensorrt-v1")

    enrolled = gallery.enroll(
        {"person_id": "employee-1", "name": "tester", "category": "employee"},
        [_unit(0), _unit(1)],
    )

    assert enrolled.sample_count == 2
    assert enrolled.enrollment_status == "multi_sample"
    assert gallery.size == 1


def test_gallery_marks_single_photo_registration():
    gallery = InMemoryFaceGallery(model_id="opencv-sface-tensorrt-v1")

    enrolled = gallery.enroll(
        {"person_id": "employee-1", "name": "tester"}, [_unit(0)]
    )

    assert enrolled.enrollment_status == "single_sample"


def test_gallery_search_returns_ranked_person_candidates():
    gallery = InMemoryFaceGallery(model_id="opencv-sface-tensorrt-v1")
    gallery.enroll({"person_id": "a", "name": "alpha"}, [_unit(0), _unit(1)])
    gallery.enroll({"person_id": "b", "name": "beta"}, [_unit(2)])

    result = gallery.search(_unit(0), top_k=2, threshold=0.5, margin=0.2)

    assert result.decision == "matched"
    assert result.matched is True
    assert result.best.person_id == "a"
    assert result.best.similarity == pytest.approx(1.0)
    assert [candidate.person_id for candidate in result.candidates] == ["a", "b"]


def test_gallery_search_returns_unknown_below_threshold():
    gallery = InMemoryFaceGallery(model_id="opencv-sface-tensorrt-v1")
    gallery.enroll({"person_id": "a", "name": "alpha"}, [_unit(0)])
    query = (_unit(0) + _unit(1)) / np.sqrt(2)

    result = gallery.search(query, threshold=0.8, margin=0.1)

    assert result.decision == "unknown"
    assert result.matched is False


def test_gallery_search_returns_ambiguous_when_margin_is_too_small():
    gallery = InMemoryFaceGallery(model_id="opencv-sface-tensorrt-v1")
    gallery.enroll({"person_id": "a", "name": "alpha"}, [_unit(0)])
    gallery.enroll({"person_id": "b", "name": "beta"}, [_unit(1)])
    query = (_unit(0) + _unit(1)) / np.sqrt(2)

    result = gallery.search(query, threshold=0.5, margin=0.1)

    assert result.decision == "ambiguous"
    assert result.matched is False


def test_gallery_deactivate_update_and_delete_are_immediately_searchable():
    gallery = InMemoryFaceGallery(model_id="opencv-sface-tensorrt-v1")
    gallery.enroll({"person_id": "a", "name": "alpha"}, [_unit(0)])
    gallery.deactivate("a")
    assert gallery.search(_unit(0)).decision == "unknown"

    gallery.update("a", [_unit(1)], active=True)
    assert gallery.search(_unit(1), threshold=0.5).best.person_id == "a"

    assert gallery.delete("a") is True
    assert gallery.delete("a") is False
    assert gallery.size == 0


@pytest.mark.parametrize(
    "embeddings, message",
    [
        ([], "at least one"),
        ([np.zeros(128, dtype=np.float32)], "zero-norm"),
        ([np.ones(127, dtype=np.float32)], "128 values"),
    ],
)
def test_gallery_rejects_invalid_enrollment_embeddings(embeddings, message):
    gallery = InMemoryFaceGallery(model_id="opencv-sface-tensorrt-v1")

    with pytest.raises(ValueError, match=message):
        gallery.enroll({"person_id": "a", "name": "alpha"}, embeddings)


def test_gallery_rejects_duplicate_person_without_explicit_update():
    gallery = InMemoryFaceGallery(model_id="opencv-sface-tensorrt-v1")
    gallery.enroll({"person_id": "a", "name": "alpha"}, [_unit(0)])

    with pytest.raises(ValueError, match="already enrolled"):
        gallery.enroll({"person_id": "a", "name": "alpha"}, [_unit(1)])
