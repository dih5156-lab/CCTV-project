import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.ops.evaluate_insightface_tensorrt import (
    cosine_similarity,
    run_evaluation,
    summarize_scores,
)


class FakeEmbedder:
    def __init__(self, _engine):
        self.index = 0

    def embed_aligned(self, _image):
        vectors = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.1], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        vector = vectors[min(self.index, len(vectors) - 1)]
        self.index += 1
        return vector


def _args(tmp_path: Path, entries: list[dict]):
    engine = tmp_path / "face.engine"
    engine.write_bytes(b"engine")
    for entry in entries:
        image_path = tmp_path / entry["image"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(image_path),
            np.zeros((112, 112, 3), dtype=np.uint8),
        )
    gallery = tmp_path / "known_faces.json"
    gallery.write_text(json.dumps(entries), encoding="utf-8")
    return argparse.Namespace(
        engine=engine,
        gallery=gallery,
        threshold=0.5,
        warmup=0,
        iterations=1,
    )


def test_cosine_similarity_uses_normalized_dot_product():
    assert cosine_similarity(np.array([2.0, 0.0]), np.array([3.0, 0.0])) == 1.0
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_summarize_scores_reports_far_frr_and_p95_latency():
    summary = summarize_scores(
        genuine_scores=[0.8, 0.4],
        impostor_scores=[0.6, 0.2],
        threshold=0.5,
        latencies_ms=[10.0, 20.0, 30.0],
    )

    assert summary["genuine_pairs"] == 2
    assert summary["impostor_pairs"] == 2
    assert summary["false_accept_rate"] == 0.5
    assert summary["false_reject_rate"] == 0.5
    assert summary["p95_latency_ms"] == 30.0


def test_run_evaluation_rejects_missing_engine(tmp_path):
    args = _args(tmp_path, [])
    args.engine.unlink()
    with pytest.raises(FileNotFoundError, match="TensorRT engine not found"):
        run_evaluation(args, FakeEmbedder)


def test_run_evaluation_requires_two_identities(tmp_path):
    args = _args(
        tmp_path,
        [
            {"name": "a", "image": "known_faces/a1.jpg"},
            {"name": "a", "image": "known_faces/a2.jpg"},
        ],
    )
    with pytest.raises(ValueError, match="서로 다른 등록 인물 2명"):
        run_evaluation(args, FakeEmbedder)


def test_run_evaluation_requires_genuine_pair(tmp_path):
    args = _args(
        tmp_path,
        [
            {"name": "a", "image": "known_faces/a.jpg"},
            {"name": "b", "image": "known_faces/b.jpg"},
        ],
    )
    with pytest.raises(ValueError, match="동일 인물의 등록 이미지가 2장"):
        run_evaluation(args, FakeEmbedder)


def test_run_evaluation_returns_report_fields(tmp_path):
    args = _args(
        tmp_path,
        [
            {"name": "a", "image": "known_faces/a1.jpg"},
            {"name": "a", "image": "known_faces/a2.jpg"},
            {"name": "b", "image": "known_faces/b.jpg"},
        ],
    )
    report = run_evaluation(args, FakeEmbedder)

    assert report["model_id"] == "arcface-w600k-r50-tensorrt-v1"
    assert report["gallery_images"] == 3
    assert report["identities"] == 2
    assert report["genuine_pairs"] == 1
    assert report["impostor_pairs"] == 2
