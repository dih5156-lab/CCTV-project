import importlib.util
import sys
from pathlib import Path


def _load_script_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_dockerfile_sources = _load_script_module(
    "check_dockerfile_sources",
    "scripts/health/check_dockerfile_sources.py",
)


def test_copy_sources_skips_stage_copy():
    assert check_dockerfile_sources._copy_sources("COPY --from=builder /root/.local /home/cctv/.local") == []


def test_copy_sources_returns_local_sources():
    sources = check_dockerfile_sources._copy_sources(
        "COPY --chown=cctv:cctv models/model_manifest.json /app/models/model_manifest.json"
    )

    assert sources == ["models/model_manifest.json"]


def test_current_dockerfile_copy_sources_exist():
    assert check_dockerfile_sources.find_missing_sources() == []
