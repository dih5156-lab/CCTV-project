"""ArcFace TensorRT 임베딩의 등록 얼굴 분리도와 지연시간을 평가한다."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.core.ai._face_tensorrt import MODEL_ID, TensorRTFaceEmbedder


@dataclass(frozen=True)
class GallerySample:
    name: str
    image_path: Path


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = np.asarray(left, dtype=np.float32).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(
        np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    )
    if denominator <= 0.0:
        raise ValueError("cosine similarity requires non-zero vectors")
    return float(np.dot(left_vector, right_vector) / denominator)


def summarize_scores(
    genuine_scores: list[float],
    impostor_scores: list[float],
    threshold: float,
    latencies_ms: list[float],
) -> dict:
    false_accepts = sum(score >= threshold for score in impostor_scores)
    false_rejects = sum(score < threshold for score in genuine_scores)
    return {
        "genuine_pairs": len(genuine_scores),
        "impostor_pairs": len(impostor_scores),
        "false_accept_rate": (
            false_accepts / len(impostor_scores) if impostor_scores else None
        ),
        "false_reject_rate": (
            false_rejects / len(genuine_scores) if genuine_scores else None
        ),
        "average_latency_ms": (
            float(np.mean(latencies_ms)) if latencies_ms else None
        ),
        "p95_latency_ms": (
            float(np.percentile(latencies_ms, 95, method="higher"))
            if latencies_ms
            else None
        ),
    }


def load_gallery_samples(gallery_path: Path) -> list[GallerySample]:
    payload = json.loads(gallery_path.read_text(encoding="utf-8"))
    samples = []
    for entry in payload:
        name = str(entry.get("name", "")).strip()
        relative_image = str(entry.get("image", "")).strip()
        image_path = gallery_path.parent / relative_image
        if name and image_path.is_file():
            samples.append(GallerySample(name=name, image_path=image_path))
    return samples


def evaluate_samples(
    embedder,
    samples: list[GallerySample],
    threshold: float,
    warmup: int,
    iterations: int,
) -> dict:
    identities = {sample.name for sample in samples}
    if len(identities) < 2:
        raise ValueError("평가에는 서로 다른 등록 인물 2명 이상이 필요합니다")

    images = []
    for sample in samples:
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise ValueError(
                f"등록 얼굴 이미지를 읽을 수 없습니다: {sample.image_path}"
            )
        images.append(image)

    for _ in range(max(warmup, 0)):
        embedder.embed_aligned(images[0])

    embeddings = []
    latencies_ms = []
    for sample, image in zip(samples, images):
        measured = []
        embedding = None
        for _ in range(max(iterations, 1)):
            started = time.perf_counter()
            embedding = embedder.embed_aligned(image)
            measured.append((time.perf_counter() - started) * 1000.0)
        embeddings.append((sample, embedding))
        latencies_ms.extend(measured)

    genuine_scores = []
    impostor_scores = []
    for index, (left_sample, left_embedding) in enumerate(embeddings):
        for right_sample, right_embedding in embeddings[index + 1 :]:
            score = cosine_similarity(left_embedding, right_embedding)
            target = (
                genuine_scores
                if left_sample.name == right_sample.name
                else impostor_scores
            )
            target.append(score)
    if not genuine_scores:
        raise ValueError("동일 인물의 등록 이미지가 2장 이상 필요합니다")
    return summarize_scores(
        genuine_scores,
        impostor_scores,
        threshold,
        latencies_ms,
    )


def run_evaluation(
    args,
    embedder_factory=TensorRTFaceEmbedder,
) -> dict:
    if not args.engine.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {args.engine}")
    samples = load_gallery_samples(args.gallery)
    embedder = embedder_factory(args.engine)
    summary = evaluate_samples(
        embedder,
        samples,
        args.threshold,
        args.warmup,
        args.iterations,
    )
    return {
        "model_id": MODEL_ID,
        "engine_path": str(args.engine),
        "gallery_images": len(samples),
        "identities": len({sample.name for sample in samples}),
        "threshold": args.threshold,
        **summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--gallery",
        type=Path,
        default=Path("known_faces.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/models/insightface_tensorrt_poc.json"),
    )
    args = parser.parse_args()

    try:
        report = run_evaluation(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
