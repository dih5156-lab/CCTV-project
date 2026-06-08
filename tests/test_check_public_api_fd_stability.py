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


check_public_api_fd_stability = _load_script_module(
    "check_public_api_fd_stability",
    "scripts/health/check_public_api_fd_stability.py",
)


def test_fd_stability_passes_when_growth_is_bounded(monkeypatch):
    samples = iter([
        {"open": 20, "soft_limit": 100},
        {"open": 22, "soft_limit": 100},
        {"open": 23, "soft_limit": 100},
    ])
    monkeypatch.setattr(check_public_api_fd_stability, "_sample_fd_usage", lambda url, timeout: next(samples))

    result = check_public_api_fd_stability.check_fd_stability(
        "http://localhost/readiness",
        samples=3,
        interval=0,
        timeout=1,
        max_growth=4,
        max_open=None,
    )

    assert result["passed"] is True
    assert result["growth"] == 3
    assert result["max_open"] == 100


def test_fd_stability_fails_when_growth_exceeds_limit(monkeypatch):
    samples = iter([
        {"open": 20, "soft_limit": 100},
        {"open": 24, "soft_limit": 100},
        {"open": 30, "soft_limit": 100},
    ])
    monkeypatch.setattr(check_public_api_fd_stability, "_sample_fd_usage", lambda url, timeout: next(samples))

    result = check_public_api_fd_stability.check_fd_stability(
        "http://localhost/readiness",
        samples=3,
        interval=0,
        timeout=1,
        max_growth=4,
        max_open=100,
    )

    assert result["passed"] is False
    assert result["growth"] == 10
    assert "exceeded limit" in result["detail"]


def test_fd_stability_fails_when_peak_exceeds_limit(monkeypatch):
    samples = iter([
        {"open": 80, "soft_limit": 100},
        {"open": 80, "soft_limit": 100},
        {"open": 80, "soft_limit": 100},
    ])
    monkeypatch.setattr(check_public_api_fd_stability, "_sample_fd_usage", lambda url, timeout: next(samples))

    result = check_public_api_fd_stability.check_fd_stability(
        "http://localhost/readiness",
        samples=3,
        interval=0,
        timeout=1,
        max_growth=4,
        max_open=64,
    )

    assert result["passed"] is False
    assert result["peak"] == 80
    assert "FD count exceeded limit" in result["detail"]


def test_fd_stability_reports_readiness_failure(monkeypatch):
    def _fail(url, timeout):
        raise RuntimeError("readiness is not ready")

    monkeypatch.setattr(check_public_api_fd_stability, "_sample_fd_usage", _fail)

    result = check_public_api_fd_stability.check_fd_stability(
        "http://localhost/readiness",
        samples=3,
        interval=0,
        timeout=1,
        max_growth=4,
        max_open=100,
    )

    assert result["passed"] is False
    assert result["samples_collected"] == 0
    assert result["detail"] == "readiness is not ready"


def test_fd_stability_uses_readiness_soft_limit_by_default(monkeypatch):
    samples = iter([
        {"open": 1200, "soft_limit": 4096},
        {"open": 1201, "soft_limit": 4096},
        {"open": 1202, "soft_limit": 4096},
    ])
    monkeypatch.setattr(check_public_api_fd_stability, "_sample_fd_usage", lambda url, timeout: next(samples))

    result = check_public_api_fd_stability.check_fd_stability(
        "http://localhost/readiness",
        samples=3,
        interval=0,
        timeout=1,
        max_growth=4,
        max_open=None,
    )

    assert result["passed"] is True
    assert result["max_open"] == 4096
