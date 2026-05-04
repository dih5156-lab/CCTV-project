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


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

check_field_network = _load_script_module(
    "check_field_network",
    "scripts/check_field_network.py",
)


def test_parse_route_output_extracts_interface_source_and_gateway():
    parsed = check_field_network._parse_route_output(
        "192.168.88.91 via 192.168.88.1 dev eno1 src 192.168.88.10 uid 1000"
    )

    assert parsed == {
        "interface": "eno1",
        "source": "192.168.88.10",
        "gateway": "192.168.88.1",
    }


def test_run_checks_allows_unconfigured_devices():
    result = check_field_network.run_checks(env={}, allow_unconfigured=True)

    assert result["passed"] is True
    assert all(check["configured"] is False for check in result["checks"])


def test_route_check_passes_when_interface_and_subnet_match(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout="192.168.88.91 dev eno1 src 192.168.88.10 uid 1000\n",
            stderr="",
        )

    monkeypatch.setattr(check_field_network.subprocess, "run", fake_run)

    result = check_field_network.run_route_check(
        "signboard",
        "192.168.88.91",
        expected_interface="eno1",
        expected_subnet="192.168.88.0/24",
    )

    assert result["passed"] is True
    assert result["interface"] == "eno1"
    assert result["source"] == "192.168.88.10"


def test_route_check_fails_when_interface_mismatches(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout="192.168.88.91 dev wlP1p1s0 src 192.168.2.242 uid 1000\n",
            stderr="",
        )

    monkeypatch.setattr(check_field_network.subprocess, "run", fake_run)

    result = check_field_network.run_route_check(
        "signboard",
        "192.168.88.91",
        expected_interface="eno1",
        expected_subnet="192.168.88.0/24",
    )

    assert result["passed"] is False
    assert "expected interface eno1" in result["detail"]


def test_route_check_reports_command_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 2, stdout="", stderr="network is unreachable")

    monkeypatch.setattr(check_field_network.subprocess, "run", fake_run)

    result = check_field_network.run_route_check("signboard", "192.168.88.91")

    assert result["passed"] is False
    assert result["route_ok"] is False
    assert "network is unreachable" in result["detail"]


def test_route_check_can_skip_permission_denied(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            2,
            stdout="",
            stderr="Cannot open netlink socket: Operation not permitted",
        )

    monkeypatch.setattr(check_field_network.subprocess, "run", fake_run)

    result = check_field_network.run_route_check(
        "signboard",
        "192.168.88.91",
        allow_permission_denied=True,
    )

    assert result["passed"] is True
    assert result["route_ok"] is None
    assert result["skipped"] is True
