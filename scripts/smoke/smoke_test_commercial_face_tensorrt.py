"""Run YuNet -> alignment -> SFace TensorRT end-to-end smoke validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ai._commercial_face_tensorrt import (  # noqa: E402
    CommercialFaceEmbeddingPipeline,
    TensorRTSFaceEmbedder,
    TensorRTYuNetDetector,
)


def summarize_results(results: list[object], latency_ms: float) -> dict[str, object]:
    embeddings = [np.asarray(result.embedding) for result in results]
    correct_shapes = all(embedding.shape == (128,) for embedding in embeddings)
    finite = all(np.isfinite(embedding).all() for embedding in embeddings)
    unit_norm = all(
        abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-5
        for embedding in embeddings
    )
    passed = bool(embeddings and correct_shapes and finite and unit_norm)
    return {
        "passed": passed,
        "faces": len(embeddings),
        "embedding_shape": [128] if correct_shapes and embeddings else None,
        "finite": finite,
        "unit_norm": unit_norm,
        "latency_ms": round(float(latency_ms), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/commercial_face")
    )
    parser.add_argument("--score-threshold", type=float, default=0.6)
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise FileNotFoundError(f"smoke image could not be read: {args.image}")
    pipeline = CommercialFaceEmbeddingPipeline(
        TensorRTYuNetDetector(
            args.model_dir / "yunet_fp16.engine",
            score_threshold=args.score_threshold,
        ),
        TensorRTSFaceEmbedder(args.model_dir / "sface_fp16.engine"),
    )
    started = time.perf_counter()
    results = pipeline.extract_embeddings(
        frame, (0, 0, frame.shape[1], frame.shape[0])
    )
    summary = summarize_results(results, (time.perf_counter() - started) * 1000)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
