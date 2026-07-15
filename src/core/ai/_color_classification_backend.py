"""의류 ROI용 색상 분류 ONNX 백엔드."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

import cv2
import numpy as np

from ...config.config import PROJECT_ROOT
from ._attribute_runtimes import AttributeRuntime, build_tensorrt_runtime

logger = logging.getLogger(__name__)

class ColorClassificationBackend(Protocol):
    backend_name: str

    def predict(self, region: np.ndarray) -> Dict[str, object]:
        """BGR 의류 ROI에서 색상과 신뢰도를 반환한다."""


class NullColorClassificationBackend:
    backend_name = "disabled"

    def predict(self, region: np.ndarray) -> Dict[str, object]:
        return {}


class OnnxColorClassificationBackend:
    """YOLO Classification ONNX 색상 모델 실행기."""

    backend_name = "color_yolov8n"

    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        *,
        input_size: int = 160,
        score_threshold: float = 0.75,
        device: str = "cpu",
        session_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        self._input_size = max(32, int(input_size))
        self._score_threshold = min(1.0, max(0.0, float(score_threshold)))
        self._labels = self._load_labels(label_map_path)
        self._session = None
        self._runtime: Optional[AttributeRuntime] = None
        self._input_name = ""
        self._warned = False

        resolved_model_path = self._resolve_path(model_path)
        if resolved_model_path is None:
            logger.warning("색상 분류 모델 파일을 찾지 못했습니다: %s", model_path)
            return
        try:
            if resolved_model_path.suffix.lower() == ".engine":
                self._runtime = build_tensorrt_runtime(resolved_model_path)
                self._input_name = "runtime"
                logger.info("색상 TensorRT 모델 로드 완료: %s", resolved_model_path)
                return
            if session_factory is None:
                if not self._preflight(resolved_model_path):
                    return
                import onnxruntime as ort  # type: ignore

                session_factory = ort.InferenceSession
                available = set(ort.get_available_providers())
            else:
                available = {"CPUExecutionProvider"}
            providers = self._providers(device, available)
            self._session = session_factory(str(resolved_model_path), providers=providers)
            inputs = self._session.get_inputs()
            if not inputs:
                raise ValueError("ONNX input node not found")
            self._input_name = str(inputs[0].name)
            shape = getattr(inputs[0], "shape", None)
            if isinstance(shape, (list, tuple)) and len(shape) >= 4:
                if isinstance(shape[2], int) and shape[2] > 0:
                    self._input_size = int(shape[2])
            logger.info("색상 분류 모델 로드 완료: %s (providers=%s)", resolved_model_path, providers)
        except Exception as exc:
            logger.warning("색상 분류 모델 로드 실패, HSV로 폴백합니다: %s", exc)
            self._session = None

    @staticmethod
    def _resolve_path(raw_path: str) -> Optional[Path]:
        candidate = Path(raw_path).expanduser()
        if candidate.exists():
            return candidate
        project_candidate = (PROJECT_ROOT / raw_path).resolve()
        return project_candidate if project_candidate.exists() else None

    def _load_labels(self, label_map_path: str) -> Dict[int, str]:
        resolved = self._resolve_path(label_map_path)
        if resolved is None:
            logger.warning("색상 라벨 파일을 찾지 못했습니다: %s", label_map_path)
            return {}
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            return {
                int(entry["index"]): str(entry["value"])
                for entry in payload.get("labels", [])
                if isinstance(entry, dict) and "index" in entry and "value" in entry
            }
        except Exception as exc:
            logger.warning("색상 라벨 파일 로드 실패: %s", exc)
            return {}

    @staticmethod
    def _providers(device: str, available: set[str]) -> list[object]:
        providers: list[object] = []
        if str(device).startswith("cuda"):
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    @staticmethod
    def _preflight(model_path: Path) -> bool:
        code = (
            "import onnxruntime as ort; "
            f"s=ort.InferenceSession({str(model_path)!r}, providers=['CPUExecutionProvider']); "
            "print(s.get_inputs()[0].name)"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            logger.warning("색상 ONNX Runtime 사전 점검 실패: %s", exc)
            return False
        if completed.returncode == 0:
            return True
        error_lines = (completed.stderr or "").strip().splitlines()
        logger.warning(
            "색상 ONNX Runtime 사전 점검 실패(returncode=%s): %s",
            completed.returncode,
            error_lines[-1] if error_lines else "no stderr",
        )
        return False

    def _preprocess(self, region: np.ndarray) -> np.ndarray:
        height, width = region.shape[:2]
        scale = self._input_size / max(min(height, width), 1)
        resized_width = max(self._input_size, int(round(width * scale)))
        resized_height = max(self._input_size, int(round(height * scale)))
        resized = cv2.resize(region, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        x1 = max(0, (resized_width - self._input_size) // 2)
        y1 = max(0, (resized_height - self._input_size) // 2)
        cropped = resized[y1:y1 + self._input_size, x1:x1 + self._input_size]
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0).astype(np.float32)

    def predict(self, region: np.ndarray) -> Dict[str, object]:
        if (self._session is None and self._runtime is None) or not self._input_name or not self._labels or region.size == 0:
            return {}
        try:
            tensor = self._preprocess(region)
            outputs = (
                self._runtime.run(tensor)
                if self._runtime is not None
                else self._session.run(None, {self._input_name: tensor})
            )
            logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            if logits.size == 0:
                return {}
            if (
                float(np.min(logits)) >= 0.0
                and float(np.max(logits)) <= 1.0
                and np.isclose(float(np.sum(logits)), 1.0, atol=1e-3)
            ):
                probabilities = logits
            else:
                shifted = logits - float(np.max(logits))
                probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
            index = int(np.argmax(probabilities))
            confidence = float(probabilities[index])
            color = self._labels.get(index)
            if color is None or confidence < self._score_threshold:
                return {"confidence": round(confidence, 4)}
            return {"color": color, "confidence": round(confidence, 4)}
        except Exception as exc:
            if not self._warned:
                logger.warning("색상 분류 추론 실패, HSV로 폴백합니다: %s", exc)
                self._warned = True
            return {}


def build_color_classification_backend(
    model_path: Optional[str],
    label_map_path: Optional[str],
    *,
    input_size: int = 160,
    score_threshold: float = 0.75,
    device: str = "cpu",
) -> ColorClassificationBackend:
    if not model_path or not label_map_path:
        return NullColorClassificationBackend()
    return OnnxColorClassificationBackend(
        model_path,
        label_map_path,
        input_size=input_size,
        score_threshold=score_threshold,
        device=device,
    )
