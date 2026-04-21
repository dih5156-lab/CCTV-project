"""외형 속성 모델 백엔드 구현."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from ...config.config import PROJECT_ROOT
from ._attribute_backend import AttributeBackend, AttributeCrop

logger = logging.getLogger(__name__)

Predictor = Callable[[AttributeCrop], Dict[str, object]]
SessionFactory = Callable[..., object]

_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


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
        self._input_size = max(32, int(input_size))
        self._score_threshold = float(score_threshold)
        self._warned = False
        self._session = None
        self._input_name: Optional[str] = None
        self._label_map = self._load_label_map(label_map_path)
        if predictor is None and model_path:
            self._session = self._build_session(session_factory)

    def predict(self, crop: AttributeCrop) -> Dict[str, object]:
        if self._predictor is not None:
            return dict(self._predictor(crop))
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

    def _build_session(self, session_factory: Optional[SessionFactory]) -> Optional[object]:
        """ONNX Runtime 세션을 생성한다."""
        model_path = self._resolve_model_path(self._model_path)
        if model_path is None:
            logger.warning("PP-Human 속성 모델 파일을 찾지 못했습니다: %s", self._model_path)
            return None

        try:
            ort = None
            if session_factory is None:
                import onnxruntime as ort  # type: ignore

                session_factory = ort.InferenceSession
            providers = self._select_providers(ort)
            session = session_factory(str(model_path), providers=providers)
            inputs = session.get_inputs()
            if not inputs:
                logger.warning("PP-Human 속성 모델 입력 노드를 찾지 못했습니다: %s", model_path)
                return None
            self._input_name = str(inputs[0].name)
            logger.info(
                "PP-Human 속성 모델 로드 완료: %s (providers=%s)",
                model_path,
                providers,
            )
            return session
        except Exception as exc:
            logger.warning("PP-Human 속성 세션 생성 실패: %s", exc)
            return None

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
            person_crop = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        resized = cv2.resize(
            person_crop,
            (self._input_size, self._input_size),
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
        scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if scores.size == 0:
            return {}
        if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
            scores = 1.0 / (1.0 + np.exp(-scores))

        labels = self._label_map.get("labels", [])
        if not isinstance(labels, list):
            return {}

        grouped: Dict[str, List[Dict[str, object]]] = {}
        for entry in labels:
            if not isinstance(entry, dict):
                continue
            index = int(entry.get("index", -1))
            field = str(entry.get("field", "")).strip()
            if index < 0 or index >= len(scores) or not field:
                continue
            threshold = float(entry.get("threshold", self._score_threshold))
            grouped.setdefault(field, []).append({
                "score": float(scores[index]),
                "value": entry.get("value"),
                "threshold": threshold,
            })

        attrs: Dict[str, object] = {}
        for field, candidates in grouped.items():
            best = max(candidates, key=lambda item: float(item["score"]))
            if field.startswith("has_"):
                attrs[field] = float(best["score"]) >= float(best["threshold"])
                continue
            if float(best["score"]) >= float(best["threshold"]):
                attrs[field] = best["value"]

        attrs["attribute_scores"] = {
            field: round(max(float(item["score"]) for item in candidates), 4)
            for field, candidates in grouped.items()
        }
        return attrs


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
