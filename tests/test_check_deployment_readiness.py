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


def test_resolve_deployment_target_selects_jetson_for_arm64(monkeypatch):
    monkeypatch.delenv("CCTV_DEPLOYMENT_TARGET", raising=False)
    monkeypatch.setattr(check_deployment_readiness.platform, "machine", lambda: "aarch64")

    assert check_deployment_readiness._resolve_deployment_target("auto") == "jetson"


def test_resolve_deployment_target_preserves_explicit_environment(monkeypatch):
    monkeypatch.setenv("CCTV_DEPLOYMENT_TARGET", "generic")
    monkeypatch.setattr(check_deployment_readiness.platform, "machine", lambda: "aarch64")

    assert check_deployment_readiness._resolve_deployment_target("auto") == "generic"


def test_check_environment_only_overrides_compose_runtime_check(monkeypatch):
    monkeypatch.setenv("EXAMPLE_RUNTIME_VALUE", "keep")
    compose_check = check_deployment_readiness.Check("compose runtime assumptions", ["true"])
    other_check = check_deployment_readiness.Check("other", ["true"])

    compose_environment = check_deployment_readiness._check_environment(compose_check, "jetson")
    assert compose_environment is not None
    assert compose_environment["CCTV_DEPLOYMENT_TARGET"] == "jetson"
    assert compose_environment["EXAMPLE_RUNTIME_VALUE"] == "keep"
    assert check_deployment_readiness._check_environment(other_check, "jetson") is None


def test_check_environment_removes_stale_target_for_generic(monkeypatch):
    monkeypatch.setenv("CCTV_DEPLOYMENT_TARGET", "jetson")
    compose_check = check_deployment_readiness.Check("compose runtime assumptions", ["true"])

    environment = check_deployment_readiness._check_environment(compose_check, "generic")

    assert environment is not None
    assert "CCTV_DEPLOYMENT_TARGET" not in environment


def test_checks_include_runtime_assumptions():
    names = {check.name for check in check_deployment_readiness.CHECKS}
    assert "compose runtime assumptions" in names
    assert "alarm device config" in names
    assert "camera output mapping" in names


def test_python_compile_check_includes_edgex_uat_runner():
    compile_checks = [check for check in check_deployment_readiness.CHECKS if check.name == "python scripts compile"]

    assert len(compile_checks) == 1
    assert "scripts/ops/run_edgex_device_service_uat.py" in compile_checks[0].command


def test_ci_checks_skip_environment_specific_runtime_checks():
    names = {check.name for check in check_deployment_readiness.CI_CHECKS}
    assert "docker compose jetson config" in names
    assert "model manifest artifacts" not in names
    assert "compose runtime assumptions" not in names
    assert "runner action bridge entrypoint" not in names
