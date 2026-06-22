"""외형 속성 모델 백엔드 구현."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from ...config.config import PROJECT_ROOT
from ._attribute_backend import AttributeBackend, AttributeCrop
from ._attribute_runtimes import (
    AttributeRuntime,
    build_onnx_runtime,
    build_paddle_runtime,
    build_tensorrt_runtime,
    resolve_paddle_model_prefix,
)

logger = logging.getLogger(__name__)

Predictor = Callable[[AttributeCrop], Dict[str, object]]
SessionFactory = Callable[..., object]

_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
_DEFAULT_GENDER_FEMALE_MIN_SCORE = 0.75
_DEFAULT_GENDER_MALE_MAX_SCORE = 0.25


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        logger.warning("잘못된 %s 값입니다: %r, 기본값 %.2f 사용", name, raw_value, default)
        return default


def decode_pphuman_scores(
    scores: object,
    label_map: Dict[str, object],
    *,
    default_threshold: float = 0.5,
    gender_female_min_score: Optional[float] = None,
    gender_male_max_score: Optional[float] = None,
) -> Dict[str, object]:
    """PP-Human multi-label score 벡터를 속성 딕셔너리로 변환한다."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {}
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        values = 1.0 / (1.0 + np.exp(-values))

    labels = label_map.get("labels", [])
    if not isinstance(labels, list):
        return {}

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for entry in labels:
        if not isinstance(entry, dict):
            continue
        index = int(entry.get("index", -1))
        field = str(entry.get("field", "")).strip()
        if index < 0 or index >= len(values) or not field:
            continue
        threshold = float(entry.get("threshold", default_threshold))
        grouped.setdefault(field, []).append({
            "score": float(values[index]),
            "value": entry.get("value"),
            "threshold": threshold,
        })

    attrs: Dict[str, object] = {}
    female_min_score = (
        _read_float_env("APPEARANCE_GENDER_FEMALE_MIN_SCORE", _DEFAULT_GENDER_FEMALE_MIN_SCORE)
        if gender_female_min_score is None
        else float(gender_female_min_score)
    )
    male_max_score = (
        _read_float_env("APPEARANCE_GENDER_MALE_MAX_SCORE", _DEFAULT_GENDER_MALE_MAX_SCORE)
        if gender_male_max_score is None
        else float(gender_male_max_score)
    )
    for field, candidates in grouped.items():
        best = max(candidates, key=lambda item: float(item["score"]))
        if field.startswith("has_"):
            attrs[field] = float(best["score"]) >= float(best["threshold"])
            continue
        # gender는 단일 female score라 애매한 구간은 unknown으로 둔다.
        if field == "gender" and len(candidates) == 1:
            female_score = float(best["score"])
            if female_score >= female_min_score:
                attrs[field] = "female"
            elif female_score <= male_max_score:
                attrs[field] = "male"
            else:
                attrs[field] = "unknown"
            continue
        if float(best["score"]) >= float(best["threshold"]):
            attrs[field] = best["value"]

    attrs["attribute_scores"] = {
        field: round(max(float(item["score"]) for item in candidates), 4)
        for field, candidates in grouped.items()
    }
    return attrs


class NullAttributeBackend:
    """기본값: 추가 속성 모델 없이 동작."""

    backend_name = "hsv"

    def predict(self, crop: AttributeCrop) -> Dict[str, object]:
        return {}


class PPHumanAttributeBackend:
    """PP-Human 계열 속성 모델 연결 지점."""

    backend_name = "pphuman"

    def __init__(
        self,
        model_path: Optional[str] = None,
        label_map_path: Optional[str] = None,
        predictor: Optional[Predictor] = None,
        runtime: str = "auto",
        device: str = "cpu",
        input_size: int = 224,
        score_threshold: float = 0.5,
        session_factory: Optional[SessionFactory] = None,
    ) -> None:
        self._model_path = model_path
        self._predictor = predictor
        self._runtime = str(runtime or "auto").lower()
        self._device = device
        resolved_input_size = max(32, int(input_size))
        self._input_height = resolved_input_size
        self._input_width = resolved_input_size
        self._score_threshold = float(score_threshold)
        self._warned = False
        self._session = None
        self._input_name: Optional[str] = None
        self._runtime_session: Optional[AttributeRuntime] = None
        self._label_map = self._load_label_map(label_map_path)
        if predictor is None and model_path:
            self._runtime_session = self._build_runtime(session_factory)

    def predict(self, crop: AttributeCrop) -> Dict[str, object]:
        if self._predictor is not None:
            return dict(self._predictor(crop))
        if self._runtime_session is not None:
            outputs = self._runtime_session.run(self._preprocess(crop))
            return self._decode(outputs)
        # 테스트와 기존 내부 사용 호환: 직접 주입된 ONNX 세션도 실행한다.
        if self._session is not None and self._input_name:
            tensor = self._preprocess(crop)
            outputs = self._session.run(None, {self._input_name: tensor})
            return self._decode(outputs)
        if self._model_path and not self._warned:
            logger.warning(
                "PP-Human 속성 모델을 초기화하지 못해 HSV로 폴백합니다: %s",
                self._model_path,
            )
            self._warned = True
        return {}

    def _build_runtime(self, session_factory: Optional[SessionFactory]) -> Optional[AttributeRuntime]:
        """설정된 런타임에 맞는 속성 모델 세션을 생성한다."""
        model_path = self._resolve_model_path(self._model_path)
        if model_path is None:
            logger.warning("PP-Human 속성 모델 파일을 찾지 못했습니다: %s", self._model_path)
            return None

        if self._should_use_tensorrt(model_path):
            return self._build_tensorrt_runtime(model_path)

        if self._should_use_paddle(model_path):
            return self._build_paddle_runtime(model_path)

        return self._build_onnx_runtime(model_path, session_factory)

    def _should_use_tensorrt(self, model_path: Path) -> bool:
        """TensorRT engine을 직접 실행할지 판단한다."""
        if self._runtime in {"tensorrt", "trt", "engine"}:
            return True
        if self._runtime not in {"auto", ""}:
            return False
        return model_path.suffix.lower() == ".engine"

    def _build_tensorrt_runtime(self, model_path: Path) -> Optional[AttributeRuntime]:
        """TensorRT engine 세션을 생성한다."""
        try:
            runtime = build_tensorrt_runtime(model_path)
            self._set_input_shape_hint(runtime.input_shape)
            logger.info("PP-Human TensorRT 속성 모델 로드 완료: %s", model_path)
            return runtime
        except Exception as exc:
            logger.warning("PP-Human TensorRT 모델 로드 실패: %s", exc)
            return None

    def _build_onnx_runtime(
        self,
        model_path: Path,
        session_factory: Optional[SessionFactory],
    ) -> Optional[AttributeRuntime]:
        """ONNX Runtime 세션을 생성한다."""
        try:
            if session_factory is None and not self._onnx_runtime_preflight(model_path):
                return None
            ort_module = self._import_onnxruntime() if session_factory is None else None
            providers = self._select_providers(ort_module)
            runtime = build_onnx_runtime(
                model_path,
                providers,
                session_factory=session_factory,
            )
            self._input_name = runtime.input_name  # type: ignore[attr-defined]
            self._set_input_shape_hint(runtime.input_shape)
            self._session = runtime.session  # type: ignore[attr-defined]
            logger.info(
                "PP-Human 속성 모델 로드 완료: %s (providers=%s)",
                model_path,
                providers,
            )
            return runtime
        except Exception as exc:
            logger.warning("PP-Human 속성 세션 생성 실패: %s", exc)
            return None

    def _onnx_runtime_preflight(self, model_path: Path) -> bool:
        """ONNX Runtime 네이티브 크래시를 메인 AI 프로세스 밖에서 먼저 확인한다."""
        timeout = float(os.environ.get("APPEARANCE_ONNX_PREFLIGHT_TIMEOUT_SEC", "20"))
        code = (
            "import onnxruntime as ort; "
            "s=ort.InferenceSession("
            f"{str(model_path)!r}, providers=['CPUExecutionProvider']"
            "); "
            "print(s.get_inputs()[0].name)"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning("PP-Human ONNX Runtime 사전 점검 실패: %s", exc)
            return False

        if completed.returncode == 0:
            return True

        stderr = (completed.stderr or "").strip().splitlines()
        last_error = stderr[-1] if stderr else "no stderr"
        logger.warning(
            "PP-Human ONNX Runtime 사전 점검 실패(returncode=%s): %s. "
            "AI 엔진 보호를 위해 ONNX 속성 백엔드는 비활성화합니다.",
            completed.returncode,
            last_error,
        )
        return False

    @staticmethod
    def _import_onnxruntime():
        import onnxruntime as ort  # type: ignore

        return ort

    def _should_use_paddle(self, model_path: Path) -> bool:
        """Paddle inference artifact를 직접 사용할지 판단한다."""
        if self._runtime == "paddle":
            return True
        if self._runtime not in {"auto", ""}:
            return False
        if model_path.is_dir():
            return (model_path / "inference.json").exists()
        return model_path.name == "inference.json"

    def _build_paddle_runtime(self, model_path: Path) -> Optional[AttributeRuntime]:
        """Paddle inference 모델을 로드한다."""
        try:
            runtime = build_paddle_runtime(model_path)
            self._set_input_shape_hint(runtime.input_shape)
            logger.info(
                "PP-Human Paddle 속성 모델 로드 완료: %s",
                resolve_paddle_model_prefix(model_path),
            )
            return runtime
        except Exception as exc:
            logger.warning("PP-Human Paddle 모델 로드 실패: %s", exc)
            return None

    def _set_input_shape_hint(self, shape: object) -> None:
        """ONNX 입력 shape가 고정이면 전처리 resize 크기에 반영한다."""
        if not isinstance(shape, (list, tuple)) or len(shape) < 4:
            return
        height = shape[2]
        width = shape[3]
        if isinstance(height, int) and height > 0:
            self._input_height = height
        if isinstance(width, int) and width > 0:
            self._input_width = width

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[Path]:
        """모델 경로를 절대 경로로 해석한다."""
        if not model_path:
            return None
        candidate = Path(model_path).expanduser()
        if candidate.exists():
            return candidate
        project_candidate = (PROJECT_ROOT / model_path).resolve()
        if project_candidate.exists():
            return project_candidate
        return None

    def _resolve_label_map_path(self, label_map_path: Optional[str]) -> Optional[Path]:
        """라벨 맵 경로를 절대 경로로 해석한다."""
        if not label_map_path:
            return None
        candidate = Path(label_map_path).expanduser()
        if candidate.exists():
            return candidate
        project_candidate = (PROJECT_ROOT / label_map_path).resolve()
        if project_candidate.exists():
            return project_candidate
        return None

    def _load_label_map(self, label_map_path: Optional[str]) -> Dict[str, object]:
        """속성 라벨 맵 JSON을 로드한다."""
        path = self._resolve_label_map_path(label_map_path)
        if path is None:
            return {"labels": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("PP-Human 라벨 맵 로드 실패: %s (%s)", path, exc)
            return {"labels": []}

    def _select_providers(self, ort_module) -> List[object]:
        """장치 환경에 맞는 ONNX Runtime provider 우선순위를 구성한다."""
        if ort_module is None:
            return ["CPUExecutionProvider"]

        available = set(ort_module.get_available_providers())
        if str(self._device).startswith("cuda"):
            device_id = 0
            if ":" in str(self._device):
                try:
                    device_id = int(str(self._device).split(":", 1)[1])
                except ValueError:
                    device_id = 0
            providers: List[object] = []
            if "TensorrtExecutionProvider" in available:
                providers.append((
                    "TensorrtExecutionProvider",
                    {"device_id": device_id},
                ))
            if "CUDAExecutionProvider" in available:
                providers.append((
                    "CUDAExecutionProvider",
                    {"device_id": device_id},
                ))
            providers.append("CPUExecutionProvider")
            return providers
        return ["CPUExecutionProvider"]

    def _preprocess(self, crop: AttributeCrop) -> np.ndarray:
        """Paddle 계열 분류 입력 형식으로 전처리한다."""
        frame = crop.frame
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, int(crop.x))
        y1 = max(0, int(crop.y))
        x2 = min(frame_w, int(crop.x + crop.width))
        y2 = min(frame_h, int(crop.y + crop.height))
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            person_crop = np.zeros(
                (self._input_height, self._input_width, 3),
                dtype=np.uint8,
            )
        resized = cv2.resize(
            person_crop,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _DEFAULT_MEAN) / _DEFAULT_STD
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def _decode(self, outputs: List[object]) -> Dict[str, object]:
        """다중 라벨 출력 벡터를 속성 딕셔너리로 변환한다."""
        if not outputs:
            return {}
        return decode_pphuman_scores(
            outputs[0],
            self._label_map,
            default_threshold=self._score_threshold,
        )


def build_attribute_backend(
    backend: str,
    *,
    model_path: Optional[str] = None,
    label_map_path: Optional[str] = None,
    predictor: Optional[Predictor] = None,
    runtime: str = "auto",
    device: str = "cpu",
    input_size: int = 224,
    score_threshold: float = 0.5,
    session_factory: Optional[SessionFactory] = None,
) -> AttributeBackend:
    """설정값에 맞는 속성 백엔드를 생성한다."""
    normalized = str(backend or "hsv").strip().lower()
    if normalized == "pphuman":
        return PPHumanAttributeBackend(
            model_path=model_path,
            label_map_path=label_map_path,
            predictor=predictor,
            runtime=runtime,
            device=device,
            input_size=input_size,
            score_threshold=score_threshold,
            session_factory=session_factory,
        )
    return NullAttributeBackend()
