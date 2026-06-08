import importlib.util
import subprocess
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


check_deployment_readiness = _load_script_module(
    "check_deployment_readiness",
    "scripts/health/check_deployment_readiness.py",
)


def test_run_check_returns_passed_for_success(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="ok", stderr="")

    monkeypatch.setattr(check_deployment_readiness.subprocess, "run", fake_run)

    passed, detail = check_deployment_readiness._run_check(
        check_deployment_readiness.Check("ok", ["true"])
    )

    assert passed is True
    assert detail == ""


def test_run_check_returns_output_for_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 2, stdout="stdout detail", stderr="stderr detail")

    monkeypatch.setattr(check_deployment_readiness.subprocess, "run", fake_run)

    passed, detail = check_deployment_readiness._run_check(
        check_deployment_readiness.Check("fail", ["false"])
    )

    assert passed is False
    assert "stdout detail" in detail
    assert "stderr detail" in detail


def test_checks_include_runtime_assumptions():
    names = {check.name for check in check_deployment_readiness.CHECKS}
    assert "compose runtime assumptions" in names
    assert "alarm device config" in names
