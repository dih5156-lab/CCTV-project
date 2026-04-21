"""test_stream_api.py — MJPEG 스트리밍 API 단위 테스트."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from src.services.stream_api import (
    StreamApiHandler,
    _read_jpeg_quality,
    _read_stream_fps,
    start_stream_api_server,
)


# ---------------------------------------------------------------------------
# 도우미 — 최소 HTTPServer + 핸들러 조합
# ---------------------------------------------------------------------------


def _make_handler(path: str, method: str = "GET") -> StreamApiHandler:
    """요청 경로를 가진 StreamApiHandler 를 직접 생성한다."""
    mock_request = MagicMock()
    mock_request.makefile.return_value = io.BytesIO(b"")

    mock_server = MagicMock()
    mock_processor = MagicMock()
    mock_processor.cameras = {"cam-1": MagicMock(), "cam-2": MagicMock()}
    mock_server.processor = mock_processor
    mock_server.server_address = ("localhost", 8769)

    handler = StreamApiHandler.__new__(StreamApiHandler)
    handler.server = mock_server
    handler.path = path
    handler.headers = {}
    handler.wfile = io.BytesIO()
    handler.rfile = io.BytesIO(b"")
    handler.request = mock_request
    handler.client_address = ("127.0.0.1", 12345)
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.command = method

    # _respond 출력을 캡처할 수 있도록 패치
    responses: list[tuple[int, dict]] = []

    def _mock_respond(code: int, body) -> None:
        responses.append((code, body))

    handler._respond = _mock_respond  # type: ignore[method-assign]
    handler._responses = responses  # type: ignore[attr-defined]
    return handler


# ---------------------------------------------------------------------------
# /cameras 엔드포인트
# ---------------------------------------------------------------------------


class TestStreamApiCamerasList:
    def test_list_cameras_returns_all(self) -> None:
        handler = _make_handler("/cameras")
        handler._list_cameras()
        code, body = handler._responses[0]  # type: ignore[attr-defined]
        assert code == 200
        assert set(body["cameras"]) == {"cam-1", "cam-2"}

    def test_list_cameras_root_path(self) -> None:
        handler = _make_handler("/")
        handler._list_cameras()
        code, body = handler._responses[0]  # type: ignore[attr-defined]
        assert code == 200
        assert "cameras" in body

    def test_list_cameras_empty_processor(self) -> None:
        handler = _make_handler("/cameras")
        handler.server.processor.cameras = {}
        handler._list_cameras()
        code, body = handler._responses[0]  # type: ignore[attr-defined]
        assert code == 200
        assert body["cameras"] == []

    def test_health_returns_camera_summary(self) -> None:
        handler = _make_handler("/health")
        handler._health()
        code, body = handler._responses[0]  # type: ignore[attr-defined]
        assert code == 200
        assert body["service"] == "cctv-stream-api"
        assert body["status"] == "ok"
        assert body["camera_count"] == 2
        assert "checked_at" in body
        assert body["stream_fps"] == 15.0
        assert body["jpeg_quality"] == 75


# ---------------------------------------------------------------------------
# /stream/<camera_id> 엔드포인트
# ---------------------------------------------------------------------------


class TestStreamApiStream:
    def test_unknown_camera_returns_404(self) -> None:
        handler = _make_handler("/stream/nonexistent")
        handler._stream("nonexistent")
        code, body = handler._responses[0]  # type: ignore[attr-defined]
        assert code == 404
        assert "not found" in body["error"].lower()

    def test_cv2_import_error_returns_503(self) -> None:
        handler = _make_handler("/stream/cam-1")
        with patch("builtins.__import__", side_effect=ImportError("No module named 'cv2'")):
            # cv2 임포트 실패 시뮬레이션을 위해 _stream 내부의 import를 직접 패치
            with patch.dict("sys.modules", {"cv2": None}):
                handler._stream("cam-1")
        code, body = handler._responses[0]  # type: ignore[attr-defined]
        assert code == 503

    def test_do_get_routes_cameras(self) -> None:
        handler = _make_handler("/cameras")
        handler.do_GET()
        code, body = handler._responses[0]  # type: ignore[attr-defined]
        assert code == 200
        assert "cameras" in body

    def test_do_get_routes_unknown(self) -> None:
        handler = _make_handler("/unknown/path")
        handler.do_GET()
        code, body = handler._responses[0]  # type: ignore[attr-defined]
        assert code == 404

    def test_do_get_routes_health(self) -> None:
        handler = _make_handler("/health")
        handler.do_GET()
        code, body = handler._responses[0]  # type: ignore[attr-defined]
        assert code == 200
        assert body["service"] == "cctv-stream-api"
        assert body["status"] == "ok"

    def test_do_options(self) -> None:
        handler = _make_handler("/cameras", "OPTIONS")
        # send_response / send_header / end_headers 모킹
        handler.send_response = MagicMock()  # type: ignore[method-assign]
        handler.send_header = MagicMock()  # type: ignore[method-assign]
        handler.end_headers = MagicMock()  # type: ignore[method-assign]
        handler.do_OPTIONS()
        handler.send_response.assert_called_once_with(200)


# ---------------------------------------------------------------------------
# start_stream_api_server — 서버 시작 함수
# ---------------------------------------------------------------------------


class TestStartStreamApiServer:
    def test_server_starts_and_thread_daemonized(self) -> None:
        mock_proc = MagicMock()
        mock_proc.cameras = {}

        with patch("src.services.stream_api.ThreadingApiServer") as MockServer:
            mock_instance = MagicMock()
            MockServer.return_value = mock_instance

            with patch("threading.Thread") as MockThread:
                mock_thread = MagicMock()
                MockThread.return_value = mock_thread

                start_stream_api_server(mock_proc, port=19999)

        MockServer.assert_called_once_with(("", 19999), StreamApiHandler)
        assert mock_instance.processor is mock_proc
        mock_thread.start.assert_called_once()

    def test_server_start_failure_logs_error(self) -> None:
        mock_proc = MagicMock()

        with patch("src.services.stream_api.ThreadingApiServer", side_effect=OSError("port in use")):
            # OSError 발생 시 예외가 전파되지 않아야 함
            start_stream_api_server(mock_proc, port=19999)

    def test_stream_port_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STREAM_PORT", "19998")
        mock_proc = MagicMock()

        with patch("src.services.stream_api.ThreadingApiServer") as MockServer:
            mock_instance = MagicMock()
            MockServer.return_value = mock_instance
            with patch("threading.Thread"):
                start_stream_api_server(mock_proc, port=19999)

        # 환경 변수 값(19998)이 사용되어야 함
        MockServer.assert_called_once_with(("", 19998), StreamApiHandler)


class TestStreamApiEnvParsing:
    def test_invalid_stream_fps_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STREAM_FPS", "not-a-number")
        assert _read_stream_fps() == 15.0

    def test_stream_fps_is_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STREAM_FPS", "120")
        assert _read_stream_fps() == 30.0

    def test_invalid_jpeg_quality_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STREAM_JPEG_QUALITY", "bad")
        assert _read_jpeg_quality() == 75

    def test_jpeg_quality_is_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STREAM_JPEG_QUALITY", "10")
        assert _read_jpeg_quality() == 30
