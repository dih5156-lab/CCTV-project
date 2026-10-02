"""외형 속성 모델 런타임 어댑터 테스트."""

from __future__ import annotations

import ctypes
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.ai._attribute_runtimes import (
    OnnxAttributeRuntime,
    TensorRTAttributeRuntime,
    TensorRTNamedOutputsRuntime,
    resolve_paddle_model_prefix,
    validate_tensorrt_cuda_variant_compatibility,
    validate_tensorrt_version,
)


def test_onnx_attribute_runtime_feeds_named_input():
    class FakeSession:
        def run(self, output_names, feed):
            assert output_names is None
            assert list(feed) == ["input_x"]
            return [feed["input_x"] + 1]

    runtime = OnnxAttributeRuntime(
        FakeSession(),
        input_name="input_x",
        input_shape=[None, 3, 224, 224],
        providers=["CPUExecutionProvider"],
    )
    tensor = np.zeros((1, 3, 2, 2), dtype=np.float32)

    outputs = runtime.run(tensor)

    assert outputs[0].shape == tensor.shape
    assert float(outputs[0].max()) == 1.0


def test_resolve_paddle_model_prefix_for_directory(tmp_path):
    model_dir = tmp_path / "attr"
    model_dir.mkdir()

    assert resolve_paddle_model_prefix(model_dir) == model_dir / "inference"


def test_resolve_paddle_model_prefix_for_json_file():
    assert resolve_paddle_model_prefix(Path("/models/attr/inference.json")) == Path(
        "/models/attr/inference"
    )


def test_validate_tensorrt_version_accepts_same_major_minor():
    validate_tensorrt_version("10.3.0", expected="10.3")


def test_validate_tensorrt_version_rejects_incompatible_binding():
    with pytest.raises(RuntimeError, match="TensorRT version mismatch"):
        validate_tensorrt_version("10.16.1.11", expected="10.3")


def test_validate_tensorrt_cuda_variant_compatibility_passes_when_expected_matches():
    validate_tensorrt_cuda_variant_compatibility(
        expected_cuda_major="12",
        machine="aarch64",
        distribution_names=["tensorrt", "tensorrt_cu12", "numpy"],
    )


def test_validate_tensorrt_cuda_variant_compatibility_rejects_mismatch_on_arm64():
    with pytest.raises(RuntimeError, match="TensorRT CUDA major mismatch"):
        validate_tensorrt_cuda_variant_compatibility(
            expected_cuda_major="12",
            machine="aarch64",
            distribution_names=["tensorrt_cu13"],
        )


def test_validate_tensorrt_cuda_variant_compatibility_ignores_non_arm64():
    validate_tensorrt_cuda_variant_compatibility(
        expected_cuda_major="12",
        machine="x86_64",
        distribution_names=["tensorrt_cu13"],
    )


@pytest.fixture
def cuda_runtime(monkeypatch):
    # 실제 포인터 복사로 버퍼 크기·출력 소유권을 검사하되 GPU 없이 실행한다.
    class Cuda:
        def __init__(self):
            self.buffers = {}
            self.allocations = 0
            self.calls = []
            self.fail_allocation = 0

        def cudaMalloc(self, target, size):
            self.allocations += 1
            if self.allocations == self.fail_allocation:
                return 2
            buffer = ctypes.create_string_buffer(size)
            address = ctypes.addressof(buffer)
            self.buffers[address] = buffer
            target._obj.value = address
            return 0

        def cudaFree(self, pointer):
            self.calls.append('free')
            del self.buffers[pointer.value]
            return 0

        def cudaStreamCreateWithFlags(self, target, flags):
            assert flags == 1
            target._obj.value = 123
            return 0

        def cudaMemcpyAsync(self, target, source, size, kind, stream):
            assert stream.value == 123
            self.calls.append(('copy', kind))
            ctypes.memmove(target, source, size)
            return 0

        def cudaStreamSynchronize(self, stream):
            assert stream.value == 123
            self.calls.append('sync')
            return 0

        def cudaStreamDestroy(self, stream):
            assert stream.value == 123
            self.calls.append('destroy')
            return 0

    class Context:
        def __init__(self):
            self.shape = (1, 3)
            self.addresses = {}
            self.reject_shape = False
            self.reject_address = False
            self.fail_execute = False
            self.unresolved = False

        def set_input_shape(self, name, shape):
            self.shape = shape
            return not self.reject_shape

        def get_tensor_shape(self, name):
            return (-1,) if self.unresolved else self.shape

        def set_tensor_address(self, name, address):
            self.addresses[name] = address
            return not self.reject_address

        def execute_async_v3(self, stream):
            assert stream == 123
            cuda.calls.append('execute')
            if self.fail_execute:
                return False
            for name, address in self.addresses.items():
                if name != 'input':
                    ctypes.memmove(address, self.addresses['input'], int(np.prod(self.shape)) * 2)
            return True

    cuda, context = Cuda(), Context()
    engine = SimpleNamespace(get_tensor_shape=lambda name: (1, 3),
                             get_tensor_dtype=lambda name: np.float16)
    monkeypatch.setitem(sys.modules, 'tensorrt', SimpleNamespace(nptype=lambda dtype: dtype))
    monkeypatch.setattr(TensorRTAttributeRuntime, '_load_cudart', staticmethod(lambda: cuda))
    runtime = TensorRTAttributeRuntime(engine, context, 'input', 'output')
    yield runtime, cuda, context
    runtime.close()


def test_tensorrt_reuses_buffers_without_global_sync_and_owns_results(cuda_runtime):
    runtime, cuda, _ = cuda_runtime
    first = runtime.run(np.array([[1, 2, 3]], dtype=np.float32))[0]
    second = runtime.run(np.array([[4, 5, 6]], dtype=np.float32))[0]
    np.testing.assert_array_equal(first, [[1, 2, 3]])
    np.testing.assert_array_equal(second, [[4, 5, 6]])
    assert first.dtype == np.float32
    assert cuda.allocations == 2
    assert cuda.calls == [('copy', 1), 'execute', ('copy', 2), 'sync'] * 2
    runtime.close()
    assert not cuda.buffers
    assert cuda.calls[-4:] == ['sync', 'free', 'free', 'destroy']
    with pytest.raises(RuntimeError, match='closed or failed'):
        runtime.run(np.ones((1, 3)))


def test_tensorrt_grows_buffers_only_for_larger_shapes(cuda_runtime):
    runtime, cuda, _ = cuda_runtime
    runtime.run(np.ones((1, 3)))
    result = runtime.run(np.full((2, 3), 7))[0]
    np.testing.assert_array_equal(result, np.full((2, 3), 7))
    runtime.run(np.ones((1, 3)))
    assert cuda.allocations == 4
    assert len(cuda.buffers) == 2


@pytest.mark.parametrize('field', ['reject_shape', 'unresolved'])
def test_tensorrt_invalid_shapes_do_not_allocate_or_submit(cuda_runtime, field):
    runtime, cuda, context = cuda_runtime
    setattr(context, field, True)
    with pytest.raises(ValueError):
        runtime.run(np.ones((1, 3)))
    assert cuda.allocations == 0
    assert 'execute' not in cuda.calls


@pytest.mark.parametrize('field', ['reject_address', 'fail_execute'])
def test_tensorrt_failed_submission_waits_and_prevents_reuse(cuda_runtime, field):
    runtime, cuda, context = cuda_runtime
    setattr(context, field, True)
    with pytest.raises(RuntimeError):
        runtime.run(np.ones((1, 3)))
    assert cuda.calls[-1] == 'sync'
    calls = list(cuda.calls)
    with pytest.raises(RuntimeError, match='closed or failed'):
        runtime.run(np.ones((1, 3)))
    assert cuda.calls == calls
    runtime.close()
    assert not cuda.buffers


def test_tensorrt_partial_allocation_failure_is_released_on_close(cuda_runtime):
    runtime, cuda, _ = cuda_runtime
    cuda.fail_allocation = 2
    with pytest.raises(RuntimeError, match='cudaMalloc'):
        runtime.run(np.ones((1, 3)))
    runtime.close()
    runtime.close()
    assert not cuda.buffers


def test_tensorrt_named_outputs_preserve_names_and_buffers(cuda_runtime):
    runtime, cuda, context = cuda_runtime
    named = TensorRTNamedOutputsRuntime(runtime.engine, context, 'input', ['first', 'second'])
    try:
        result = named.run_named(np.array([[1, 2, 3]]))
        assert list(result) == ['first', 'second']
        for output in result.values():
            np.testing.assert_array_equal(output, [[1, 2, 3]])
        assert cuda.allocations == 3
    finally:
        named.close()


def test_tensorrt_concurrent_callers_keep_their_own_results(cuda_runtime):
    runtime, cuda, _ = cuda_runtime
    # 하나의 context를 여러 호출자가 사용해도 입력/출력이 뒤섞이지 않아야 한다.
    def infer(value):
        return runtime.run(np.full((1, 3), value))[0]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(infer, range(24)))
    for value, result in enumerate(results):
        np.testing.assert_array_equal(result, np.full((1, 3), value))
    assert cuda.allocations == 2
