"""
test_face_api.py — FaceApiHandler 단위 / 통합 테스트

테스트 구성:
    TestFaceApiGET    - GET /faces, GET /known_faces/{filename}
    TestFaceApiPOST   - POST /faces
    TestFaceApiDELETE - DELETE /faces/{face_id}
    TestFaceApiRouting - 서버 기동 확인
"""

import json
import threading
import time
import urllib.request
import urllib.error
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.services.face_api import FaceApiHandler, start_face_api_server

# ---------------------------------------------------------------------------
# 공통 픽스처 / 헬퍼
# ---------------------------------------------------------------------------

FACE_RECORD = {
    "id": "face_001",
    "name": "홍길동",
    "phone": "010-1234-5678",
    "department": "보안팀",
}


def _build_processor() -> MagicMock:
    """가짜 VideoProcessor를 생성한다."""
    proc = MagicMock()
    proc.list_registered_faces = MagicMock(return_value=[FACE_RECORD])
    proc.register_face = MagicMock(return_value=FACE_RECORD)
    proc.delete_face = MagicMock(return_value=True)
    return proc


def _live_server(processor, port: int = 0):
    """실제 HTTP 서버를 스레드로 기동하여 반환한다."""
    from http.server import HTTPServer

    try:
        server = HTTPServer(("127.0.0.1", port), FaceApiHandler)
    except PermissionError as exc:
        pytest.skip(f"이 환경에서는 로컬 소켓 바인딩이 허용되지 않음: {exc}")
    server.processor = processor
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    return server


from tests.conftest import http_request as _request


def _make_handler(processor, path: str) -> FaceApiHandler:
    handler = FaceApiHandler.__new__(FaceApiHandler)
    handler.server = SimpleNamespace(processor=processor)
    handler.path = path
    handler.headers = {}
    handler.wfile = BytesIO()
    handler.rfile = BytesIO(b"")
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.command = "GET"
    responses: list[tuple[int, dict]] = []

    def _mock_respond(code: int, body) -> None:
        responses.append((code, body))

    handler._respond = _mock_respond  # type: ignore[method-assign]
    handler._responses = responses  # type: ignore[attr-defined]
    return handler


# ===========================================================================
# GET /faces, GET /known_faces/{filename}
# ===========================================================================


class TestFaceApiGET:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.proc = _build_processor()
        self.server = _live_server(self.proc)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        yield
        self.server.shutdown()

    def test_get_faces_returns_list(self):
        code, body = _request("GET", f"{self.base}/faces")
        assert code == 200
        assert "faces" in body
        assert isinstance(body["faces"], list)
        assert len(body["faces"]) == 1
        assert body["faces"][0]["id"] == "face_001"

    def test_get_faces_calls_processor(self):
        _request("GET", f"{self.base}/faces")
        self.proc.list_registered_faces.assert_called_once()

    def test_get_faces_processor_error_returns_500(self):
        self.proc.list_registered_faces.side_effect = RuntimeError("db error")
        code, body = _request("GET", f"{self.base}/faces")
        assert code == 500
        assert "error" in body

    def test_get_known_face_image_not_found(self):
        code, body = _request("GET", f"{self.base}/known_faces/nonexistent.jpg")
        assert code == 404
        assert "error" in body

    def test_get_known_face_invalid_filename(self):
        code, body = _request("GET", f"{self.base}/known_faces/../secret.txt")
        assert code in (400, 404)

    def test_get_unknown_path_returns_404(self):
        code, _ = _request("GET", f"{self.base}/unknown")
        assert code == 404

    def test_get_health_returns_service_metadata(self):
        code, body = _request("GET", f"{self.base}/health")
        assert code == 200
        assert body["service"] == "cctv-face-api"
        assert body["status"] == "ok"
        assert body["face_count"] == 1
        assert "checked_at" in body


def test_face_health_direct_handler_returns_service_metadata():
    proc = _build_processor()
    handler = _make_handler(proc, "/health")
    handler.do_GET()
    code, body = handler._responses[0]  # type: ignore[attr-defined]
    assert code == 200
    assert body["service"] == "cctv-face-api"
    assert body["status"] == "ok"
    assert body["face_count"] == 1
    assert "checked_at" in body


def test_face_api_requires_internal_token_when_configured():
    proc = _build_processor()
    handler = _make_handler(proc, "/faces")
    with patch.dict("os.environ", {"INTERNAL_SERVICE_TOKEN": "internal-secret"}):
        handler.do_GET()
    code, body = handler._responses[0]  # type: ignore[attr-defined]
    assert code == 401
    assert body["error"] == "Unauthorized"


# ===========================================================================
# POST /faces
# ===========================================================================


class TestFaceApiPOST:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.proc = _build_processor()
        self.server = _live_server(self.proc)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        yield
        self.server.shutdown()

    def test_post_face_success(self):
        code, body = _request(
            "POST",
            f"{self.base}/faces",
            {
                "name": "홍길동",
                "phone": "010-1234-5678",
                "image_base64": "base64encodeddata==",
            },
        )
        assert code == 201
        assert body["status"] == "ok"
        assert "face" in body

    def test_post_face_calls_processor(self):
        _request(
            "POST",
            f"{self.base}/faces",
            {
                "name": "홍길동",
                "phone": "010-1234-5678",
                "image_base64": "base64data==",
            },
        )
        self.proc.register_face.assert_called_once()
        kwargs = self.proc.register_face.call_args[1]
        assert kwargs["name"] == "홍길동"
        assert kwargs["phone"] == "010-1234-5678"

    def test_post_face_missing_name_returns_400(self):
        code, body = _request(
            "POST",
            f"{self.base}/faces",
            {"phone": "010-0000-0000", "image_base64": "data=="},
        )
        assert code == 400
        assert "error" in body

    def test_post_face_missing_phone_returns_400(self):
        code, body = _request(
            "POST",
            f"{self.base}/faces",
            {"name": "홍길동", "image_base64": "data=="},
        )
        assert code == 400
        assert "error" in body

    def test_post_face_missing_image_returns_400(self):
        code, body = _request(
            "POST",
            f"{self.base}/faces",
            {"name": "홍길동", "phone": "010-1234-5678"},
        )
        assert code == 400
        assert "error" in body

    def test_post_face_invalid_json_returns_400(self):
        req = urllib.request.Request(
            f"{self.base}/faces",
            data=b"not-json",
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_post_face_processor_value_error_returns_400(self):
        self.proc.register_face.side_effect = ValueError("얼굴 감지 실패")
        code, body = _request(
            "POST",
            f"{self.base}/faces",
            {"name": "홍길동", "phone": "010-0000-0000", "image_base64": "data=="},
        )
        assert code == 400
        assert "error" in body

    def test_post_face_processor_exception_returns_500(self):
        self.proc.register_face.side_effect = RuntimeError("db error")
        code, body = _request(
            "POST",
            f"{self.base}/faces",
            {"name": "홍길동", "phone": "010-0000-0000", "image_base64": "data=="},
        )
        assert code == 500

    def test_post_unknown_path_returns_404(self):
        code, _ = _request("POST", f"{self.base}/other", {"name": "test"})
        assert code == 404


# ===========================================================================
# DELETE /faces/{face_id}
# ===========================================================================


class TestFaceApiDELETE:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.proc = _build_processor()
        self.server = _live_server(self.proc)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        yield
        self.server.shutdown()

    def test_delete_face_success(self):
        code, body = _request("DELETE", f"{self.base}/faces/face_001")
        assert code == 200
        assert body["status"] == "ok"
        assert body["deleted_face_id"] == "face_001"

    def test_delete_face_calls_processor(self):
        _request("DELETE", f"{self.base}/faces/face_001")
        self.proc.delete_face.assert_called_once_with("face_001")

    def test_delete_nonexistent_face_returns_404(self):
        self.proc.delete_face.return_value = False
        code, body = _request("DELETE", f"{self.base}/faces/nonexistent")
        assert code == 404
        assert "error" in body

    def test_delete_processor_exception_returns_500(self):
        self.proc.delete_face.side_effect = RuntimeError("db error")
        code, body = _request("DELETE", f"{self.base}/faces/face_001")
        assert code == 500
        assert "error" in body

    def test_delete_unknown_path_returns_404(self):
        code, _ = _request("DELETE", f"{self.base}/unknown")
        assert code == 404


# ===========================================================================
# 서버 기동 테스트
# ===========================================================================


class TestFaceApiRouting:

    def test_start_server_starts_thread(self):
        """start_face_api_server 가 데몬 스레드를 생성해야 한다."""
        proc = _build_processor()
        with patch("src.services.face_api.ThreadingApiServer") as mock_srv_cls:
            mock_srv = MagicMock()
            mock_srv_cls.return_value = mock_srv
            start_face_api_server(proc, 19997)
            time.sleep(0.05)
            mock_srv.serve_forever.assert_called_once()
