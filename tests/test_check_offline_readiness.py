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


check_offline_readiness = _load_script_module(
    "check_offline_readiness",
    "scripts/check_offline_readiness.py",
)


def test_build_checks_quick_uses_focused_pytest():
    checks = check_offline_readiness.build_checks(full=False)
    names = [check.name for check in checks]

    assert "deployment readiness" in names
    assert "alarm config without network" in names
    assert "offline focused pytest" in names
    assert "full pytest" not in names


def test_build_checks_full_uses_full_pytest():
    checks = check_offline_readiness.build_checks(full=True)
    commands = {check.name: check.command for check in checks}

    assert commands["full pytest"] == [sys.executable, "-m", "pytest"]
    assert "offline focused pytest" not in commands


def test_parser_check_sets_pythonpath():
    checks = check_offline_readiness.build_checks()
    parser_check = next(check for check in checks if check.name == "parser pytest")

    assert parser_check.env == {"PYTHONPATH": "parser-python"}
    assert "-c" in parser_check.command
    assert "/dev/null" in parser_check.command


def test_run_check_returns_failure_detail(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="stdout detail", stderr="stderr detail")

    monkeypatch.setattr(check_offline_readiness.subprocess, "run", fake_run)

    passed, detail = check_offline_readiness._run_check(
        check_offline_readiness.Check("fail", ["false"])
    )

    assert passed is False
    assert "stdout detail" in detail
    assert "stderr detail" in detail
