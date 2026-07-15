import numpy as np

from scripts.smoke.smoke_test_commercial_face_tensorrt import summarize_results


class FakeResult:
    def __init__(self, embedding):
        self.embedding = embedding


def test_summarize_results_passes_for_finite_unit_embeddings():
    embedding = np.ones(128, dtype=np.float32) / np.sqrt(128)

    summary = summarize_results([FakeResult(embedding)], latency_ms=12.5)

    assert summary["passed"] is True
    assert summary["faces"] == 1
    assert summary["latency_ms"] == 12.5


def test_summarize_results_fails_without_faces_or_with_invalid_embedding():
    assert summarize_results([], latency_ms=1.0)["passed"] is False

    invalid = np.ones(128, dtype=np.float32)
    invalid[0] = np.nan
    summary = summarize_results([FakeResult(invalid)], latency_ms=1.0)

    assert summary["passed"] is False
    assert summary["finite"] is False
