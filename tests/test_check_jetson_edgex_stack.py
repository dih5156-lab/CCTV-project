from __future__ import annotations

import subprocess

from scripts.health import check_jetson_edgex_stack as module


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
