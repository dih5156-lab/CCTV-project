"""외형 속성 모델 런타임 어댑터."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np


class AttributeRuntime(Protocol):
    """속성 모델 런타임 공통 인터페이스."""

    input_shape: object

    def run(self, tensor: np.ndarray) -> List[object]:
        """전처리된 NCHW tensor를 받아 모델 출력을 반환한다."""


class OnnxAttributeRuntime:
    """ONNX Runtime 세션 실행 어댑터."""

    def __init__(
        self,
        session: object,
        input_name: str,
        input_shape: object,
        providers: List[object],
    ) -> None:
        self.session = session
        self.input_name = input_name
        self.input_shape = input_shape
        self.providers = providers

    def run(self, tensor: np.ndarray) -> List[object]:
        return self.session.run(None, {self.input_name: tensor})


class PaddleAttributeRuntime:
    """Paddle inference 모델 실행 어댑터."""

    input_shape = [None, 3, 256, 192]

    def __init__(self, model: object) -> None:
        self.model = model

    def run(self, tensor: np.ndarray) -> List[object]:
        import paddle  # type: ignore

        output = self.model(paddle.to_tensor(tensor))
        if isinstance(output, (list, tuple)):
            output = output[0]
        if hasattr(output, "numpy"):
            output = output.numpy()
        return [output]


def resolve_paddle_model_prefix(model_path: Path) -> Path:
    """paddle.jit.load에 전달할 inference prefix를 계산한다."""
    if model_path.is_dir():
        return model_path / "inference"
    return model_path.with_suffix("")


def build_paddle_runtime(model_path: Path) -> PaddleAttributeRuntime:
    """Paddle inference 모델을 로드한다."""
    import paddle  # type: ignore

    model_prefix = resolve_paddle_model_prefix(model_path)
    model = paddle.jit.load(str(model_prefix))
    model.eval()
    return PaddleAttributeRuntime(model)


def build_onnx_runtime(
    model_path: Path,
    providers: List[object],
    session_factory: Optional[object] = None,
) -> OnnxAttributeRuntime:
    """ONNX Runtime 세션을 생성한다."""
    if session_factory is None:
        import onnxruntime as ort  # type: ignore

        session_factory = ort.InferenceSession

    session = session_factory(str(model_path), providers=providers)  # type: ignore[misc]
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError(f"ONNX input node not found: {model_path}")

    input_name = str(inputs[0].name)
    input_shape = getattr(inputs[0], "shape", None)
    return OnnxAttributeRuntime(session, input_name, input_shape, providers)
