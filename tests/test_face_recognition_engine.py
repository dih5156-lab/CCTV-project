"""얼굴 인식 런타임 선택 테스트."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from src.utils.face_recognition import FaceRecognitionEngine


def _engine_for_device(device: str) -> FaceRecognitionEngine:
    engine = FaceRecognitionEngine.__new__(FaceRecognitionEngine)
    engine._ctx_id = 0 if device.startswith("cuda") else -1
    engine._det_size = (320, 320) if engine._ctx_id >= 0 else (160, 160)
    return engine


def _install_fake_insightface(monkeypatch) -> None:
    insightface_module = ModuleType("insightface")
    app_module = ModuleType("insightface.app")
    app_module.FaceAnalysis = object
    insightface_module.app = app_module
    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)


def test_explicit_insightface_cuda_requires_cuda_provider(monkeypatch):
    _install_fake_insightface(monkeypatch)
    ort_module = ModuleType("onnxruntime")
    ort_module.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    monkeypatch.setenv("FACE_RECOGNITION_BACKEND", "insightface")

    with pytest.raises(RuntimeError, match="InsightFace 실행 환경") as exc_info:
        _engine_for_device("cuda:0")._load_insightface()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "CUDAExecutionProvider" in str(exc_info.value.__cause__)


def test_auto_backend_keeps_opencv_fallback_when_insightface_is_missing(monkeypatch):
    monkeypatch.setenv("FACE_RECOGNITION_BACKEND", "auto")
    monkeypatch.setitem(sys.modules, "insightface", None)
    monkeypatch.setitem(sys.modules, "insightface.app", None)

    assert _engine_for_device("cpu")._load_insightface() is None


def test_disabled_backend_disables_detector_and_gallery(monkeypatch, tmp_path):
    monkeypatch.setenv("FACE_RECOGNITION_BACKEND", "disabled")
    gallery_path = tmp_path / "known_faces.json"
    gallery_path.write_text(
        '[{"name": "tester", "phone": "010", "image": "known_faces/tester.jpg"}]',
        encoding="utf-8",
    )

    engine = FaceRecognitionEngine(gallery_path=str(gallery_path), device="cpu")

    assert engine.insight_app is None
    assert engine.detector is None
    assert engine.gallery == {}
    assert engine.enabled is False
