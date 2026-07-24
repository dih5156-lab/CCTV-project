from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "health" / "check_torch_cuda.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_torch_cuda", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_cuda_status_reports_cpu_only_torch(monkeypatch):
    module = _load_module()

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    class FakeTorch:
        __version__ = "test"
        version = type("Version", (), {"cuda": None})()
        cuda = FakeCuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch())
    ok, status = module.collect_cuda_status()

    assert ok is False
    assert status["cuda_available"] is False
    assert "사용할 수 없음" in status["error"]
