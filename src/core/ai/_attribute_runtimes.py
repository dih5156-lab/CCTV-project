"""외형 속성 모델 런타임 어댑터."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import re
from importlib import metadata
from pathlib import Path
from threading import RLock
from typing import List, Optional, Protocol

import numpy as np

DEFAULT_TENSORRT_MAJOR_MINOR = "10.3"
DEFAULT_TENSORRT_EXPECTED_CUDA_MAJOR = "12"


def _normalize_machine(machine: Optional[str] = None) -> str:
    value = (machine or platform.machine()).strip().lower()
    if value in {"aarch64", "arm64"}:
        return "arm64"
    if value in {"x86_64", "amd64"}:
        return "amd64"
    return value or "unknown"


def _installed_tensorrt_cuda_variant_majors(
    distribution_names: Optional[List[str]] = None,
) -> set[int]:
    """Collect CUDA major versions from pip TensorRT package names (e.g. tensorrt_cu13)."""
    names = distribution_names
    if names is None:
        names = []
        for dist in metadata.distributions():
            name = str(dist.metadata.get("Name") or "").strip()
            if name:
                names.append(name)

    majors: set[int] = set()
    for raw_name in names:
        match = re.match(r"^tensorrt_cu(\d+)$", str(raw_name).strip().lower())
        if match:
            majors.add(int(match.group(1)))
    return majors


def validate_tensorrt_cuda_variant_compatibility(
    *,
    expected_cuda_major: Optional[str] = None,
    machine: Optional[str] = None,
    distribution_names: Optional[List[str]] = None,
) -> None:
    """Fail early on arm64 when pip TensorRT CUDA major mismatches the host baseline."""
    if _normalize_machine(machine) != "arm64":
        return

    raw_expected = expected_cuda_major or os.getenv(
        "TENSORRT_EXPECTED_CUDA_MAJOR",
        DEFAULT_TENSORRT_EXPECTED_CUDA_MAJOR,
    )
    try:
        expected_major = int(str(raw_expected).strip())
    except (TypeError, ValueError):
        return

    variant_majors = _installed_tensorrt_cuda_variant_majors(distribution_names)
    if not variant_majors:
        return
    if expected_major in variant_majors:
        return

    found = ", ".join(f"cu{major}" for major in sorted(variant_majors))
    raise RuntimeError(
        "TensorRT CUDA major mismatch: "
        f"found pip TensorRT variants [{found}], expected cu{expected_major}. "
        "On Jetson, remove pip tensorrt_cu* packages and use the system/container TensorRT binding. "
        "If this host intentionally uses another CUDA baseline, override "
        "TENSORRT_EXPECTED_CUDA_MAJOR."
    )


def validate_tensorrt_version(
    actual: str,
    *,
    expected: Optional[str] = None,
) -> None:
    """Stop before native runtime creation when Python TensorRT is incompatible."""
    required = expected or os.getenv(
        "TENSORRT_EXPECTED_VERSION", DEFAULT_TENSORRT_MAJOR_MINOR
    )
    actual_parts = str(actual).split(".")
    required_parts = str(required).split(".")
    actual_major_minor = ".".join(actual_parts[:2])
    required_major_minor = ".".join(required_parts[:2])
    if actual_major_minor != required_major_minor:
        raise RuntimeError(
            "TensorRT version mismatch: "
            f"Python binding is {actual}, expected {required_major_minor}.x. "
            "Use the Jetson system/container TensorRT binding."
        )


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

    input_shape = (None, 3, 256, 192)

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


class TensorRTAttributeRuntime:
    """TensorRT engine 실행 어댑터.

    CUDA Python/pycuda 추가 설치 없이 libcudart를 ctypes로 호출한다.
    호출자에는 완료된 결과를 반환하되 GPU 대기는 이 런타임의 스트림으로 제한한다.
    """

    def __init__(self, engine: object, context: object, input_name: str, output_name: str) -> None:
        self.engine = engine
        self.context = context
        self.input_name = input_name
        self.output_name = output_name
        self.input_shape = tuple(int(dim) for dim in engine.get_tensor_shape(input_name))
        self._cudart = self._load_cudart()
        # 같은 context/버퍼의 동시 사용을 막고, 반복 추론의 할당·전역 동기화를 없앤다.
        self._lock = RLock()
        self._stream = ctypes.c_void_p()
        self._buffers: dict[str, tuple[ctypes.c_void_p, int]] = {}
        self._closed = False
        self._failed = False

    @staticmethod
    def _load_cudart() -> object:
        libname = ctypes.util.find_library("cudart") or "libcudart.so"
        cudart = ctypes.CDLL(libname)
        cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        cudart.cudaMalloc.restype = ctypes.c_int
        cudart.cudaFree.argtypes = [ctypes.c_void_p]
        cudart.cudaFree.restype = ctypes.c_int
        cudart.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        cudart.cudaMemcpyAsync.restype = ctypes.c_int
        cudart.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
        ]
        cudart.cudaStreamCreateWithFlags.restype = ctypes.c_int
        for name in ("cudaStreamSynchronize", "cudaStreamDestroy"):
            function = getattr(cudart, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int
        return cudart

    @staticmethod
    def _check_cuda(code: int, action: str) -> None:
        if code != 0:
            raise RuntimeError(f"{action} failed with cuda error {code}")

    def _cuda_malloc(self, size: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check_cuda(
            self._cudart.cudaMalloc(ctypes.byref(ptr), int(size)),
            "cudaMalloc",
        )
        return ptr

    def run(self, tensor: np.ndarray) -> List[object]:
        return self._run_outputs(tensor, [self.output_name])

    def _run_outputs(
        self, tensor: np.ndarray, output_names: List[str]
    ) -> List[object]:
        with self._lock:
            if self._closed or self._failed:
                raise RuntimeError("TensorRT runtime is closed or failed; recreate it")
            return self._run_outputs_locked(tensor, output_names)

    def _device_buffer(self, name: str, size: int) -> ctypes.c_void_p:
        previous = self._buffers.get(name)
        if previous is not None and previous[1] >= size:
            return previous[0]
        pointer = self._cuda_malloc(size)
        if previous is not None:
            # 이전 호출은 스트림 동기화 후 반환하므로 shape 증가 시에만 교체한다.
            code = self._cudart.cudaFree(previous[0])
            if code != 0:
                self._cudart.cudaFree(pointer)
                self._check_cuda(code, "cudaFree resized buffer")
        self._buffers[name] = (pointer, size)
        return pointer

    def _run_outputs_locked(
        self, tensor: np.ndarray, output_names: List[str]
    ) -> List[object]:
        import tensorrt as trt  # type: ignore

        # 엔진 dtype/확정 shape와 실제 할당 크기를 맞춰 잘못된 버퍼 접근을 예방한다.
        input_tensor = np.ascontiguousarray(
            tensor, dtype=trt.nptype(self.engine.get_tensor_dtype(self.input_name)),
        )
        if not self.context.set_input_shape(self.input_name, tuple(input_tensor.shape)):
            raise ValueError("TensorRT input shape rejected")
        output_tensors = []
        for name in output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"TensorRT unresolved or empty output shape: {name} {shape}")
            output_tensors.append(np.empty(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name))))
        if input_tensor.nbytes == 0:
            raise ValueError("TensorRT empty input")
        try:
            if not self._stream.value:
                self._check_cuda(self._cudart.cudaStreamCreateWithFlags(
                    ctypes.byref(self._stream), 1,  # cudaStreamNonBlocking
                ), "cudaStreamCreateWithFlags")
            input_ptr = self._device_buffer(self.input_name, input_tensor.nbytes)
            output_ptrs = [self._device_buffer(name, output.nbytes)
                           for name, output in zip(output_names, output_tensors)]
            self._check_cuda(
                self._cudart.cudaMemcpyAsync(
                    input_ptr,
                    input_tensor.ctypes.data_as(ctypes.c_void_p),
                    input_tensor.nbytes,
                    1,  # cudaMemcpyHostToDevice
                    self._stream,
                ),
                "cudaMemcpy host to device",
            )
            if not self.context.set_tensor_address(self.input_name, int(input_ptr.value)):
                raise RuntimeError("TensorRT input address rejected")
            for output_name, output_ptr in zip(output_names, output_ptrs):
                if not self.context.set_tensor_address(output_name, int(output_ptr.value)):
                    raise RuntimeError(f"TensorRT output address rejected: {output_name}")
            if not self.context.execute_async_v3(int(self._stream.value)):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            for output_tensor, output_ptr in zip(output_tensors, output_ptrs):
                self._check_cuda(
                    self._cudart.cudaMemcpyAsync(
                        output_tensor.ctypes.data_as(ctypes.c_void_p),
                        output_ptr,
                        output_tensor.nbytes,
                        2,  # cudaMemcpyDeviceToHost
                        self._stream,
                    ),
                    "cudaMemcpy device to host",
                )
            # 복사와 추론을 같은 스트림에 순서대로 넣고 결과 반환 전에만 기다린다.
            self._check_cuda(self._cudart.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")
            return [output.astype(np.float32, copy=False) for output in output_tensors]
        except Exception:
            # 비동기 제출 뒤 실패해도 호스트 배열 수명 안에서 대기하고 재사용을 금지한다.
            self._failed = True
            if self._stream.value:
                self._cudart.cudaStreamSynchronize(self._stream)
            raise

    def close(self) -> None:
        """진행 중인 추론과 직렬화한 뒤 소유한 GPU 자원을 반환한다."""
        with self._lock:
            if self._closed:
                return
            self._failed = True
            if self._stream.value:
                self._check_cuda(self._cudart.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize close")
            for name in list(self._buffers):
                self._check_cuda(self._cudart.cudaFree(self._buffers[name][0]), "cudaFree close")
                del self._buffers[name]
            if self._stream.value:
                self._check_cuda(self._cudart.cudaStreamDestroy(self._stream), "cudaStreamDestroy")
                self._stream = ctypes.c_void_p()
            self._closed = True

    def __del__(self) -> None:
        # 생성 중 실패·인터프리터 종료에서는 정리 예외를 전파하지 않는다.
        try:
            self.close()
        except Exception:
            pass


class TensorRTNamedOutputsRuntime(TensorRTAttributeRuntime):
    """TensorRT adapter that preserves every output tensor name."""

    def __init__(
        self,
        engine: object,
        context: object,
        input_name: str,
        output_names: List[str],
    ) -> None:
        if not output_names:
            raise ValueError("TensorRT named runtime requires at least one output")
        super().__init__(engine, context, input_name, output_names[0])
        self.output_names = output_names

    def run_named(self, tensor: np.ndarray) -> dict[str, object]:
        outputs = self._run_outputs(tensor, self.output_names)
        return dict(zip(self.output_names, outputs))


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


def build_tensorrt_runtime(model_path: Path) -> TensorRTAttributeRuntime:
    """TensorRT engine을 로드한다."""
    import tensorrt as trt  # type: ignore

    validate_tensorrt_cuda_variant_compatibility()
    validate_tensorrt_version(trt.__version__)

    logger = trt.Logger(trt.Logger.ERROR)
    with model_path.open("rb") as handle, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise ValueError(f"TensorRT engine load failed: {model_path}")

    input_name = None
    output_name = None
    for index in range(int(engine.num_io_tensors)):
        name = str(engine.get_tensor_name(index))
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT and input_name is None:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT and output_name is None:
            output_name = name
    if not input_name or not output_name:
        raise ValueError(f"TensorRT engine input/output not found: {model_path}")

    context = engine.create_execution_context()
    if context is None:
        raise ValueError(f"TensorRT execution context create failed: {model_path}")
    return TensorRTAttributeRuntime(engine, context, input_name, output_name)


def build_tensorrt_named_runtime(model_path: Path) -> TensorRTNamedOutputsRuntime:
    """Load a TensorRT engine while retaining all output tensor names."""
    import tensorrt as trt  # type: ignore

    validate_tensorrt_cuda_variant_compatibility()
    validate_tensorrt_version(trt.__version__)

    logger = trt.Logger(trt.Logger.ERROR)
    with model_path.open("rb") as handle, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise ValueError(f"TensorRT engine load failed: {model_path}")

    input_names: List[str] = []
    output_names: List[str] = []
    for index in range(int(engine.num_io_tensors)):
        name = str(engine.get_tensor_name(index))
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    if len(input_names) != 1 or not output_names:
        raise ValueError(
            f"TensorRT named runtime expected one input and at least one output: {model_path}"
        )
    context = engine.create_execution_context()
    if context is None:
        raise ValueError(f"TensorRT execution context create failed: {model_path}")
    return TensorRTNamedOutputsRuntime(engine, context, input_names[0], output_names)
