import subprocess
import sys
from pathlib import Path


def test_deepstream_service_import_does_not_require_torch():
    # TensorRT 전용 엣지는 학습/선택 보조 모델의 PyTorch 없이도 서비스가 로드되어야 한다.
    code = """
import importlib.abc
import sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('torch intentionally unavailable')

sys.meta_path.insert(0, NoTorch())
import app.main
from src.core.deepstream_processor import DeepStreamProcessor
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
