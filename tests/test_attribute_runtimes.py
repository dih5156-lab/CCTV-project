"""외형 속성 모델 런타임 어댑터 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.ai._attribute_runtimes import (
    OnnxAttributeRuntime,
    resolve_paddle_model_prefix,
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
