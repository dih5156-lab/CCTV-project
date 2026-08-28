from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.health import check_jetson_edgex_stack as module

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "health"
    / "check_jetson_edgex_stack.py"
)


def test_tcp_check_falls_back_to_container_health(monkeypatch) -> None:
    item = module.TcpCheck(
        "AIoT Parser DB",
        "127.0.0.1",
        5432,
        fallback_container="aiot-parser-db",
    )

    monkeypatch.setattr(
        module,
        "_check_tcp",
        lambda _item, _timeout: (False, "127.0.0.1:5432 연결 실패"),
    )
    monkeypatch.setattr(
        module,
        "_check_container_health",
        lambda _container_name, _timeout: (
            True,
            "Docker 상태 fallback: status=running, health=healthy",
        ),
    )

    ok, detail = module._check_tcp_with_fallback(item, timeout=1.0)

    assert ok is True
    assert "연결 실패" in detail
    assert "Docker 상태 fallback" in detail


def test_http_check_falls_back_to_container_health(monkeypatch) -> None:
    item = module.HttpCheck(
        "AIoT Parser",
        "http://127.0.0.1:3500/health",
        fallback_container="aiot-parser",
    )

    monkeypatch.setattr(
        module,
        "_check_http",
        lambda _item, _timeout: (False, "Connection refused"),
    )
    monkeypatch.setattr(
        module,
        "_check_container_health",
        lambda _container_name, _timeout: (
            True,
            "Docker 상태 fallback: status=running, health=healthy",
        ),
    )

    ok, detail = module._check_http_with_fallback(item, timeout=1.0)

    assert ok is True
    assert "Connection refused" in detail
    assert "Docker 상태 fallback" in detail


def test_container_health_reports_unhealthy(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="running unhealthy\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    ok, detail = module._check_container_health("aiot-parser", timeout=1.0)

    assert ok is False
    assert "status=running" in detail
    assert "health=unhealthy" in detail


def test_main_includes_requested_device_contract_check(monkeypatch, capsys) -> None:
    monkeypatch.setattr(module, "_build_tcp_checks", lambda **_kwargs: [])
    monkeypatch.setattr(module, "_build_http_checks", lambda **_kwargs: [])
    monkeypatch.setattr(
        module,
        "_check_device_contracts",
        lambda **_kwargs: (False, "3개 계약 문제 발견"),
        raising=False,
    )

    exit_code = module.main(["--json", "--check-device-contracts"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["results"] == [
        {
            "type": "contract",
            "name": "EdgeX Device Contracts",
            "ok": False,
            "detail": "3개 계약 문제 발견",
        }
    ]
    assert output["failures"] == ["EdgeX Device Contracts"]


def test_script_entrypoint_can_load_contract_checker() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--check-device-contracts" in completed.stdout
