"""내부 서비스 헬스체크 응답 테스트."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from runners.run_alert_api import AlertHandler
from src.protocols.rest import _RestHandler


def _make_alert_handler(path: str = "/health") -> AlertHandler:
    """AlertHandler 인스턴스를 테스트용으로 직접 생성한다."""
    handler = AlertHandler.__new__(AlertHandler)
    handler.server = SimpleNamespace(
        log_path=Path("logs/alerts.jsonl"),
        sensor_log_path=Path("logs/sensor.jsonl"),
    )
    handler.path = path
    handler.headers = {}
    handler.wfile = io.BytesIO()
    handler.rfile = io.BytesIO(b"")
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.command = "GET"
    responses: list[tuple[int, dict]] = []

    def _mock_send_json(code: int, body: dict) -> None:
        responses.append((code, body))

    handler._send_json = _mock_send_json  # type: ignore[method-assign]
    handler._responses = responses  # type: ignore[attr-defined]
    return handler


def _make_rest_handler(
    path: str = "/health",
    *,
    running: bool = True,
    mqtt_connected: bool = True,
) -> _RestHandler:
    """_RestHandler 인스턴스를 테스트용으로 직접 생성한다."""
    handler = _RestHandler.__new__(_RestHandler)
    handler.server = SimpleNamespace(
        action_layer=SimpleNamespace(
            _running=running,
            _mqtt_client=SimpleNamespace(
                is_connected=lambda: mqtt_connected,
            ),
            default_mode=SimpleNamespace(value="auto"),
            list_sites=lambda: [{"site_id": "site-1"}],
            get_pending_events=lambda: [{"event_id": "evt-1"}],
        )
    )
    handler.path = path
    handler.headers = {}
    handler.wfile = io.BytesIO()
    handler.rfile = io.BytesIO(b"")
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.command = "GET"
    responses: list[tuple[int, dict]] = []

    def _mock_respond(code: int, body: dict) -> None:
        responses.append((code, body))

    handler._respond = _mock_respond  # type: ignore[method-assign]
    handler._responses = responses  # type: ignore[attr-defined]
    return handler


def test_alert_health_response_contains_service_metadata() -> None:
    handler = _make_alert_handler()
    handler.do_GET()
    code, body = handler._responses[0]  # type: ignore[attr-defined]
    assert code == 200
    assert body["service"] == "cctv-alert-api"
    assert body["status"] == "up"
    assert "checked_at" in body


def test_action_layer_health_response_contains_service_metadata() -> None:
    handler = _make_rest_handler()
    handler.do_GET()
    code, body = handler._responses[0]  # type: ignore[attr-defined]
    assert code == 200
    assert body["service"] == "cctv-action-layer"
    assert body["status"] == "up"
    assert body["mqtt"] == "connected"
    assert "checked_at" in body


def test_action_layer_health_degraded_when_mqtt_is_down() -> None:
    handler = _make_rest_handler(running=True, mqtt_connected=False)
    handler.do_GET()
    code, body = handler._responses[0]  # type: ignore[attr-defined]
    assert code == 503
    assert body["service"] == "cctv-action-layer"
    assert body["status"] == "degraded"
    assert body["mqtt"] == "disconnected"
