"""배포 ONNX가 보조 특징 대신 최종 예측만 내보내는지 CPU로 검증한다."""

import importlib.util
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch


class PoseWithAuxiliaryOutputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 56, 1)

    def forward(self, images):
        predictions = self.projection(images).flatten(2)
        return predictions, (images.mean(dim=1), images * 2)


def test_export_rejects_unrecognized_output_structure():
    from scripts.convert.convert_pt_to_onnx import _PredictionOnly

    class InvalidOutput(torch.nn.Module):
        def forward(self, images):
            return {"predictions": images}

    with pytest.raises(TypeError, match="Tensor"):
        _PredictionOnly(InvalidOutput())(torch.ones(1, 3, 8, 8))


@pytest.mark.parametrize("auxiliary", [False, True])
def test_export_preserves_prediction_with_only_one_output(tmp_path, monkeypatch, auxiliary):
    path = Path(__file__).resolve().parents[1] / "scripts/convert/convert_pt_to_onnx.py"
    spec = importlib.util.spec_from_file_location("export_onnx", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    model = PoseWithAuxiliaryOutputs().eval() if auxiliary else torch.nn.Conv2d(3, 56, 1).eval()
    monkeypatch.setattr(exporter.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(exporter.torch, "load", lambda *args, **kwargs: {"model": model})
    monkeypatch.setattr(exporter, "dummy_input", torch.ones(1, 3, 8, 8))
    target = tmp_path / "model.onnx"
    exporter.convert_to_onnx("test-model.pt", target)
    graph = onnx.load(target)
    onnx.checker.check_model(graph)
    assert [output.name for output in graph.graph.output] == ["output"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
    for size in (8, 12):
        images = torch.linspace(0, 1, 3 * size * size).reshape(1, 3, size, size)
        with torch.no_grad():
            expected = model(images)
            if auxiliary:
                expected = expected[0]
        actual = session.run(["output"], {"input": images.numpy()})[0]
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_pose_export_avoids_scatter_and_preserves_dynamic_keypoints(tmp_path, monkeypatch):
    from ultralytics.nn.modules.head import Pose

    from scripts.convert import convert_pt_to_onnx as exporter

    class KeypointModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Pose(nc=1, kpt_shape=(17, 3), ch=(16, 32, 64))

        def forward(self, images):
            batch, _, height, width = images.shape
            raw = images.repeat(1, 17, 1, 1).reshape(batch, 51, height * width)
            grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
            self.head.anchors = torch.stack((grid_x.flatten(), grid_y.flatten())).float()
            self.head.strides = torch.ones(1, height * width) * 8
            return self.head.kpts_decode(raw)

    model = KeypointModel().eval()
    reference = KeypointModel().eval()
    monkeypatch.setattr(exporter.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(exporter.torch, "load", lambda *args, **kwargs: {"model": model})
    monkeypatch.setattr(exporter, "dummy_input", torch.ones(1, 3, 8, 8))
    target = tmp_path / "pose.onnx"
    exporter.convert_to_onnx("pose-test.pt", target)
    graph = onnx.load(target)
    assert not any(node.op_type.startswith("Scatter") for node in graph.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
    for batch, height, width in ((1, 8, 8), (2, 12, 10)):
        images = torch.linspace(-2, 2, batch * 3 * height * width).reshape(batch, 3, height, width)
        expected = reference(images).detach().numpy()
        actual = session.run(["output"], {"input": images.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
