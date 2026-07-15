# InsightFace TensorRT POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Jetson에서 ONNX Runtime 없이 InsightFace `w600k_r50` ArcFace 모델을 TensorRT로 변환·실행하고, 기존 등록 얼굴로 정확도와 지연시간을 측정하는 독립 POC를 만든다.

**Architecture:** 기존 `buffalo_l/w600k_r50.onnx`를 고정 batch 1 FP16 TensorRT 엔진으로 변환한다. POC 런타임은 정렬된 112×112 얼굴 입력만 받아 512차원 L2 정규화 임베딩을 반환하며, 모델 변환·전처리·유사도 평가를 메인 DeepStream 프로세스와 분리한다. 이 계획은 ArcFace 경로의 안정성과 품질을 먼저 증명하며 `det_10g` 얼굴 검출/랜드마크, 로컬 서비스 API, 중앙 서버 연동은 후속 계획으로 둔다.

**Tech Stack:** Python 3.10, NumPy 1.x, OpenCV, TensorRT 10.3, CUDA 12.6, pytest, `trtexec`

## Global Constraints

- 실시간 카메라 영상 추론은 Jetson에서 수행한다.
- 운영 경로에서는 네이티브 abort가 재현된 ONNX Runtime을 import하거나 실행하지 않는다.
- 현재 정상 운영 중인 OpenCV 백엔드와 DeepStream 컨테이너 설정은 POC 검증 전 변경하지 않는다.
- 모델 ID는 `arcface-w600k-r50-tensorrt-v1`, 입력은 RGB `1x3x112x112`, 출력은 512차원 float 임베딩으로 고정한다.
- TensorRT 엔진은 Jetson 대상 장치에서 생성하며 다른 장치에서 생성한 engine을 재사용하지 않는다.
- 얼굴 원본, crop, 임베딩을 로그에 출력하지 않는다.
- 새 Python 라이브러리는 추가하지 않는다. 저장소에 이미 설치된 NumPy, OpenCV, TensorRT만 사용한다.

---

## File Structure

- Create `src/core/ai/_face_tensorrt.py`: ArcFace 전처리, TensorRT 런타임 어댑터, 임베딩 정규화 책임.
- Create `scripts/convert/convert_insightface_arcface_to_engine.py`: `trtexec` 명령 생성과 ArcFace engine 변환 책임.
- Create `scripts/ops/evaluate_insightface_tensorrt.py`: 등록 얼굴을 이용한 임베딩 품질·지연시간 평가와 JSON report 생성 책임.
- Create `tests/test_face_tensorrt.py`: 전처리, shape, 정규화, 런타임 위임 단위 테스트.
- Create `tests/test_convert_insightface_arcface.py`: 변환 명령과 artifact 검증 단위 테스트.
- Create `tests/test_evaluate_insightface_tensorrt.py`: 평가 pair 생성, threshold metric, report 직렬화 테스트.
- Modify `scripts/health/check_model_report.py`: 새 POC report의 모델 ID, sample 수, 지연시간 필드 검증.
- Modify `tests/test_check_model_report.py`: POC report 검증 회귀 테스트.

### Task 1: ArcFace 전처리와 TensorRT 임베딩 런타임

**Files:**
- Create: `src/core/ai/_face_tensorrt.py`
- Create: `tests/test_face_tensorrt.py`

**Interfaces:**
- Consumes: `src.core.ai._attribute_runtimes.build_tensorrt_runtime(model_path: Path)`가 반환하는 `.run(tensor) -> list[np.ndarray]` 런타임.
- Produces: `preprocess_arcface_bgr(image: np.ndarray) -> np.ndarray`, `normalize_embedding(vector: np.ndarray) -> np.ndarray`, `TensorRTFaceEmbedder(model_path: Path, runtime_factory=build_tensorrt_runtime)`, `TensorRTFaceEmbedder.embed_aligned(image: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: 전처리 실패 테스트 작성**

```python
import numpy as np
import pytest

from src.core.ai._face_tensorrt import preprocess_arcface_bgr


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
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_face_tensorrt.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.ai._face_tensorrt'`.

- [ ] **Step 3: 최소 ArcFace 전처리 구현**

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ._attribute_runtimes import build_tensorrt_runtime

MODEL_ID = "arcface-w600k-r50-tensorrt-v1"
INPUT_SIZE = (112, 112)


def preprocess_arcface_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ArcFace input must be a non-empty BGR image")
    resized = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - 127.5) / 127.5
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])
```

- [ ] **Step 4: 전처리 테스트 GREEN 확인**

Run: `rtk pytest tests/test_face_tensorrt.py -q`

Expected: `2 passed`.

- [ ] **Step 5: 임베딩 런타임 실패 테스트 추가**

```python
from pathlib import Path

from src.core.ai._face_tensorrt import TensorRTFaceEmbedder, normalize_embedding


class FakeRuntime:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    def run(self, tensor):
        self.inputs.append(tensor)
        return [self.output]


def test_normalize_embedding_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_embedding(np.zeros(512, dtype=np.float32))


def test_embed_aligned_returns_l2_normalized_512_vector():
    runtime = FakeRuntime(np.ones((1, 512), dtype=np.float32))
    embedder = TensorRTFaceEmbedder(
        Path("face.engine"), runtime_factory=lambda _: runtime
    )

    embedding = embedder.embed_aligned(np.zeros((112, 112, 3), dtype=np.uint8))

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
```

- [ ] **Step 6: RED 확인**

Run: `rtk pytest tests/test_face_tensorrt.py -q`

Expected: FAIL because `TensorRTFaceEmbedder` and `normalize_embedding` are not defined.

- [ ] **Step 7: 최소 임베딩 런타임 구현**

```python
def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    if norm <= 0.0:
        raise ValueError("ArcFace produced a zero-norm embedding")
    return np.ascontiguousarray(flattened / norm)


class TensorRTFaceEmbedder:
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
        raw = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if raw.size != 512:
            raise ValueError(f"ArcFace output must contain 512 values, received {raw.size}")
        return normalize_embedding(raw)
```

- [ ] **Step 8: Task 1 전체 테스트 및 커밋**

Run: `rtk pytest tests/test_face_tensorrt.py -q`

Expected: `5 passed`.

Run: `rtk git add src/core/ai/_face_tensorrt.py tests/test_face_tensorrt.py`

Run: `rtk git commit -m "Add TensorRT ArcFace embedding runtime"`

### Task 2: InsightFace ArcFace TensorRT 변환 CLI

**Files:**
- Create: `scripts/convert/convert_insightface_arcface_to_engine.py`
- Create: `tests/test_convert_insightface_arcface.py`

**Interfaces:**
- Consumes: `data/insightface/models/buffalo_l/w600k_r50.onnx`, `trtexec` executable.
- Produces: `build_trtexec_command(onnx_path: Path, engine_path: Path, trtexec: Path) -> list[str]`, `validate_arcface_artifact(onnx_path: Path) -> None`, CLI artifact `models/insightface/w600k_r50_fp16.engine`.

- [ ] **Step 1: 변환 명령 실패 테스트 작성**

```python
from pathlib import Path

import pytest

from scripts.convert.convert_insightface_arcface_to_engine import (
    build_trtexec_command,
    validate_arcface_artifact,
)


def test_build_trtexec_command_uses_fixed_arcface_shape():
    command = build_trtexec_command(
        Path("w600k_r50.onnx"), Path("w600k_r50_fp16.engine"), Path("trtexec")
    )

    assert "--minShapes=input.1:1x3x112x112" in command
    assert "--optShapes=input.1:1x3x112x112" in command
    assert "--maxShapes=input.1:1x3x112x112" in command
    assert "--fp16" in command
    assert "--skipInference" in command


def test_validate_arcface_artifact_rejects_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="ArcFace ONNX model not found"):
        validate_arcface_artifact(tmp_path / "missing.onnx")
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_convert_insightface_arcface.py -q`

Expected: FAIL with missing module.

- [ ] **Step 3: 최소 변환 CLI 구현**

```python
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def validate_arcface_artifact(onnx_path: Path) -> None:
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ArcFace ONNX model not found: {onnx_path}")


def build_trtexec_command(onnx_path: Path, engine_path: Path, trtexec: Path) -> list[str]:
    shape = "input.1:1x3x112x112"
    return [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes={shape}",
        f"--optShapes={shape}",
        f"--maxShapes={shape}",
        "--skipInference",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=Path("data/insightface/models/buffalo_l/w600k_r50.onnx"))
    parser.add_argument("--engine", type=Path, default=Path("models/insightface/w600k_r50_fp16.engine"))
    parser.add_argument("--trtexec", type=Path)
    args = parser.parse_args()
    validate_arcface_artifact(args.onnx)
    trtexec = args.trtexec or Path(shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec")
    if not trtexec.is_file():
        raise FileNotFoundError(f"trtexec not found: {trtexec}")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_trtexec_command(args.onnx, args.engine, trtexec), check=True)
    if not args.engine.is_file() or args.engine.stat().st_size == 0:
        raise RuntimeError(f"TensorRT engine was not created: {args.engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: GREEN 확인과 커밋**

Run: `rtk pytest tests/test_convert_insightface_arcface.py -q`

Expected: `2 passed`.

Run: `rtk git add scripts/convert/convert_insightface_arcface_to_engine.py tests/test_convert_insightface_arcface.py`

Run: `rtk git commit -m "Add ArcFace TensorRT conversion CLI"`

### Task 3: ArcFace TensorRT 품질·지연시간 평가기

**Files:**
- Create: `scripts/ops/evaluate_insightface_tensorrt.py`
- Create: `tests/test_evaluate_insightface_tensorrt.py`

**Interfaces:**
- Consumes: `TensorRTFaceEmbedder.embed_aligned(image)`, `known_faces.json`, `known_faces/` 이미지.
- Produces: `cosine_similarity(left: np.ndarray, right: np.ndarray) -> float`, `summarize_scores(genuine_scores: list[float], impostor_scores: list[float], threshold: float, latencies_ms: list[float]) -> dict`, JSON report at `reports/models/insightface_tensorrt_poc.json`.

- [ ] **Step 1: metric 실패 테스트 작성**

```python
import numpy as np

from scripts.ops.evaluate_insightface_tensorrt import cosine_similarity, summarize_scores


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
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_evaluate_insightface_tensorrt.py -q`

Expected: FAIL with missing module.

- [ ] **Step 3: metric 최소 구현**

```python
def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        raise ValueError("cosine similarity requires non-zero vectors")
    return float(np.dot(left, right) / denominator)


def summarize_scores(genuine_scores, impostor_scores, threshold, latencies_ms):
    false_accepts = sum(score >= threshold for score in impostor_scores)
    false_rejects = sum(score < threshold for score in genuine_scores)
    return {
        "genuine_pairs": len(genuine_scores),
        "impostor_pairs": len(impostor_scores),
        "false_accept_rate": false_accepts / len(impostor_scores) if impostor_scores else None,
        "false_reject_rate": false_rejects / len(genuine_scores) if genuine_scores else None,
        "average_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else None,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95, method="higher")) if latencies_ms else None,
    }
```

- [ ] **Step 4: GREEN 확인**

Run: `rtk pytest tests/test_evaluate_insightface_tensorrt.py -q`

Expected: `2 passed`.

- [ ] **Step 5: CLI 평가 흐름 구현**

Add the following data loading and evaluation functions below the metric helpers:

```python
@dataclass(frozen=True)
class GallerySample:
    name: str
    image_path: Path


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


def evaluate_samples(embedder, samples, threshold, warmup, iterations):
    identities = {sample.name for sample in samples}
    if len(identities) < 2:
        raise ValueError("평가에는 서로 다른 등록 인물 2명 이상이 필요합니다")

    images = []
    for sample in samples:
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise ValueError(f"등록 얼굴 이미지를 읽을 수 없습니다: {sample.image_path}")
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
        for right_sample, right_embedding in embeddings[index + 1:]:
            score = cosine_similarity(left_embedding, right_embedding)
            target = genuine_scores if left_sample.name == right_sample.name else impostor_scores
            target.append(score)
    if not genuine_scores:
        raise ValueError("동일 인물의 등록 이미지가 2장 이상 필요합니다")
    return summarize_scores(genuine_scores, impostor_scores, threshold, latencies_ms)


def run_evaluation(args, embedder_factory=TensorRTFaceEmbedder) -> dict:
    if not args.engine.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {args.engine}")
    samples = load_gallery_samples(args.gallery)
    embedder = embedder_factory(args.engine)
    summary = evaluate_samples(
        embedder, samples, args.threshold, args.warmup, args.iterations
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
    parser.add_argument("--gallery", type=Path, default=Path("known_faces.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("reports/models/insightface_tensorrt_poc.json"))
    args = parser.parse_args()
    try:
        report = run_evaluation(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add these imports at the top of the evaluator:

```python
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.core.ai._face_tensorrt import MODEL_ID, TensorRTFaceEmbedder
```

- [ ] **Step 6: CLI behavior 테스트 작성**

```python
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.ops.evaluate_insightface_tensorrt import run_evaluation


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
        cv2.imwrite(str(image_path), np.zeros((112, 112, 3), dtype=np.uint8))
    gallery = tmp_path / "known_faces.json"
    gallery.write_text(json.dumps(entries), encoding="utf-8")
    return argparse.Namespace(
        engine=engine,
        gallery=gallery,
        threshold=0.5,
        warmup=0,
        iterations=1,
    )


def test_run_evaluation_rejects_missing_engine(tmp_path):
    args = _args(tmp_path, [])
    args.engine.unlink()
    with pytest.raises(FileNotFoundError, match="TensorRT engine not found"):
        run_evaluation(args, FakeEmbedder)


def test_run_evaluation_requires_two_identities(tmp_path):
    args = _args(tmp_path, [
        {"name": "a", "image": "known_faces/a1.jpg"},
        {"name": "a", "image": "known_faces/a2.jpg"},
    ])
    with pytest.raises(ValueError, match="서로 다른 등록 인물 2명"):
        run_evaluation(args, FakeEmbedder)


def test_run_evaluation_requires_genuine_pair(tmp_path):
    args = _args(tmp_path, [
        {"name": "a", "image": "known_faces/a.jpg"},
        {"name": "b", "image": "known_faces/b.jpg"},
    ])
    with pytest.raises(ValueError, match="동일 인물의 등록 이미지가 2장"):
        run_evaluation(args, FakeEmbedder)


def test_run_evaluation_returns_report_fields(tmp_path):
    args = _args(tmp_path, [
        {"name": "a", "image": "known_faces/a1.jpg"},
        {"name": "a", "image": "known_faces/a2.jpg"},
        {"name": "b", "image": "known_faces/b.jpg"},
    ])
    report = run_evaluation(args, FakeEmbedder)

    assert report["model_id"] == "arcface-w600k-r50-tensorrt-v1"
    assert report["gallery_images"] == 3
    assert report["identities"] == 2
    assert report["genuine_pairs"] == 1
    assert report["impostor_pairs"] == 2
```

- [ ] **Step 7: CLI behavior 테스트 통과 확인**

Run: `rtk pytest tests/test_evaluate_insightface_tensorrt.py -q`

Expected: `6 passed`.

- [ ] **Step 8: Task 3 커밋**

Run: `rtk git add scripts/ops/evaluate_insightface_tensorrt.py tests/test_evaluate_insightface_tensorrt.py`

Run: `rtk git commit -m "Add ArcFace TensorRT quality evaluator"`

### Task 4: POC report 운영 검증과 Jetson 실측

**Files:**
- Modify: `scripts/health/check_model_report.py`
- Modify: `tests/test_check_model_report.py`
- Runtime artifact, do not commit: `models/insightface/w600k_r50_fp16.engine`
- Runtime report, commit only if repository policy already tracks reports: `reports/models/insightface_tensorrt_poc.json`

**Interfaces:**
- Consumes: Task 3 JSON report.
- Produces: `check_insightface_tensorrt_report(report: dict) -> list[str]`, where an empty list means valid and each string is a validation error.

- [ ] **Step 1: report validation 실패 테스트 작성**

```python
from scripts.health.check_model_report import check_insightface_tensorrt_report


def test_insightface_report_requires_model_id_and_measured_samples():
    errors = check_insightface_tensorrt_report({"model_id": "wrong", "gallery_images": 0})

    assert "unexpected InsightFace model_id: wrong" in errors
    assert "InsightFace gallery_images must be at least 2" in errors


def test_insightface_report_accepts_complete_poc_result():
    errors = check_insightface_tensorrt_report({
        "model_id": "arcface-w600k-r50-tensorrt-v1",
        "gallery_images": 4,
        "identities": 2,
        "genuine_pairs": 2,
        "impostor_pairs": 4,
        "false_accept_rate": 0.0,
        "false_reject_rate": 0.0,
        "p95_latency_ms": 40.0,
    })

    assert errors == []
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_check_model_report.py -q`

Expected: FAIL because `check_insightface_tensorrt_report` is not defined.

- [ ] **Step 3: 최소 report validator 구현**

```python
def check_insightface_tensorrt_report(report: dict) -> list[str]:
    errors = []
    model_id = report.get("model_id")
    if model_id != "arcface-w600k-r50-tensorrt-v1":
        errors.append(f"unexpected InsightFace model_id: {model_id}")
    for key, minimum in (
        ("gallery_images", 2),
        ("identities", 2),
        ("genuine_pairs", 1),
        ("impostor_pairs", 1),
    ):
        if int(report.get(key, 0)) < minimum:
            errors.append(f"InsightFace {key} must be at least {minimum}")
    if report.get("p95_latency_ms") is None:
        errors.append("InsightFace p95_latency_ms is required")
    return errors
```

- [ ] **Step 4: validator 테스트 GREEN 확인**

Run: `rtk pytest tests/test_check_model_report.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Jetson에서 engine 생성**

Run:

```bash
rtk docker exec cctv-ai-engine python scripts/convert/convert_insightface_arcface_to_engine.py \
  --onnx /root/.insightface/models/buffalo_l/w600k_r50.onnx \
  --engine /app/models/insightface/w600k_r50_fp16.engine \
  --trtexec /usr/src/tensorrt/bin/trtexec
```

Expected: exit 0 and a non-empty `/app/models/insightface/w600k_r50_fp16.engine`. The current Jetson Compose configuration mounts host `data/insightface` at `/root/.insightface`; if that mount changes, use a read-only one-off bind mount rather than changing the running service before the POC passes.

- [ ] **Step 6: ONNX Runtime 비사용 smoke 확인**

Run:

```bash
rtk docker exec cctv-ai-engine python -c "from pathlib import Path; import numpy as np; from src.core.ai._face_tensorrt import TensorRTFaceEmbedder; e=TensorRTFaceEmbedder(Path('/app/models/insightface/w600k_r50_fp16.engine')); v=e.embed_aligned(np.zeros((112,112,3),dtype=np.uint8)); print(v.shape, float(np.linalg.norm(v)))"
```

Expected: exit 0, output `(512,) 1.0` within floating-point tolerance, and no `onnxruntime` warning, abort, or container restart.

- [ ] **Step 7: 실제 갤러리 평가 실행**

Run:

```bash
rtk docker exec cctv-ai-engine python scripts/ops/evaluate_insightface_tensorrt.py \
  --engine /app/models/insightface/w600k_r50_fp16.engine \
  --gallery /app/known_faces.json \
  --output /app/data/reports/models/insightface_tensorrt_poc.json
```

Expected for POC acceptance: exit 0, at least 2 identities, at least 1 genuine pair, at least 1 impostor pair, and `p95_latency_ms <= 1000`. FAR/FRR are measured and reported but no production threshold is approved from this small gallery alone.

- [ ] **Step 8: 회귀 및 운영 안정성 확인**

Run: `rtk pytest tests/test_face_tensorrt.py tests/test_convert_insightface_arcface.py tests/test_evaluate_insightface_tensorrt.py tests/test_check_model_report.py -q`

Expected: all tests PASS.

Run: `rtk pytest tests/test_face_recognition_engine.py tests/test_deepstream_face_context.py tests/test_deepstream_processor.py -q`

Expected: all tests PASS.

Run: `rtk ./scripts/ops/run_deepstream_stability_watch.sh 1 10`

Expected: 5 samples, 5 PASS, 0 FAIL, AI engine restart count unchanged.

- [ ] **Step 9: Task 4 커밋**

Run: `rtk git add scripts/health/check_model_report.py tests/test_check_model_report.py`

Run: `rtk git commit -m "Validate ArcFace TensorRT POC reports"`

## POC Exit Gate

Proceed to the face detector/alignment and local service plan only when all conditions are true:

- TensorRT engine conversion exits 0 on the target Jetson.
- The direct embedding smoke process exits normally without ONNX Runtime import or native abort.
- Output is a finite, L2-normalized 512-value vector.
- At least two identities and one same-person pair are evaluated.
- Measured p95 embedding latency is at most 1 second.
- DeepStream remains healthy with zero additional container restarts during the stability watch.

If the gallery lacks two images for any identity, stop at the quality evaluation gate and collect one additional consented registration image per test identity. Do not infer FAR/FRR from fabricated or duplicated images.
