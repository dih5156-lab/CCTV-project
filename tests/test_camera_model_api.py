"""
test_camera_model_api.py — CameraModelApiHandler 단위 / 통합 테스트

테스트 구성:
    TestCameraModelApiGET  - GET /cameras/{id}/models
    TestCameraModelApiPOST - POST /cameras/{id}/models
    TestCameraModelApiRouting - 미등록 경로 → 404
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

from src.services.camera_model_api import CameraModelApiHandler, start_camera_model_api_server

# ---------------------------------------------------------------------------
# 공통 픽스처 / 헬퍼
# ---------------------------------------------------------------------------

CAMERAS_JSON_DATA = [
    {
        "id": "camera_1",
        "name": "테스트 카메라",
        "enabled": True,
        "detections": ["helmet"],
        "model_settings": {"use_pose": True, "use_helmet": True, "use_person": False},
        "zones": [],
    },
    {
        "id": "camera_2",
        "name": "카메라 2",
        "enabled": False,
        "model_settings": {"use_pose": False, "use_helmet": False, "use_person": False},
        "zones": [],
    },
]


@pytest.fixture
def cameras_json(tmp_path: Path) -> Path:
    p = tmp_path / "cameras.json"
    p.write_text(json.dumps(CAMERAS_JSON_DATA, ensure_ascii=False), encoding="utf-8")
    return p


def _build_processor() -> MagicMock:
    """가짜 VideoProcessor를 생성한다."""
    proc = MagicMock()
    proc.get_camera_status = MagicMock(
        return_value={
            "camera_1": {"status": "online"},
            "camera_2": {"status": "offline"},
        }
    )
    proc.get_camera_model_settings = MagicMock(
        side_effect=lambda camera_id: {
            "camera_1": {"use_pose": True, "use_helmet": True, "use_person": False},
            "camera_2": {"use_pose": False, "use_helmet": False, "use_person": False},
        }.get(camera_id)
    )

    def _update_model_settings(camera_id, settings, *_):
        if camera_id not in ("camera_1", "camera_2"):
            return None
        use_helmet = bool(settings.get("use_helmet", settings.get("helmet", True)))
        use_pose = bool(settings.get("use_pose", settings.get("pose", True)))
        return {
            "use_pose": use_pose,
            "use_helmet": use_helmet,
            "use_person": bool(settings.get("use_person", settings.get("person", False))),
        }

    proc.update_camera_model_settings = MagicMock(side_effect=_update_model_settings)
    return proc


def _live_server(processor, cameras_json_path: str, port: int = 0):
    """실제 HTTP 서버를 스레드로 기동하여 반환한다."""
    from http.server import HTTPServer

    try:
        server = HTTPServer(("127.0.0.1", port), CameraModelApiHandler)
    except PermissionError as exc:
        pytest.skip(f"이 환경에서는 로컬 소켓 바인딩이 허용되지 않음: {exc}")
    server.processor = processor
    server.cameras_json_path = cameras_json_path
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    return server


from tests.conftest import http_request as _request


def _make_handler(processor, cameras_json_path: str, path: str) -> CameraModelApiHandler:
    handler = CameraModelApiHandler.__new__(CameraModelApiHandler)
    handler.server = SimpleNamespace(
        processor=processor,
        cameras_json_path=cameras_json_path,
    )
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
# GET /cameras/{id}/models
# ===========================================================================


class TestCameraModelApiGET:

    @pytest.fixture(autouse=True)
    def setup(self, cameras_json: Path):
        self.proc = _build_processor()
        self.server = _live_server(self.proc, str(cameras_json))
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        yield
        self.server.shutdown()

    def test_get_camera1_model_settings(self):
        code, body = _request("GET", f"{self.base}/cameras/camera_1/models")
        assert code == 200
        assert body["camera_id"] == "camera_1"
        assert body["model_settings"]["use_pose"] is True
        assert body["model_settings"]["use_helmet"] is True

    def test_get_camera2_model_settings(self):
        code, body = _request("GET", f"{self.base}/cameras/camera_2/models")
        assert code == 200
        assert body["camera_id"] == "camera_2"
        assert body["model_settings"]["use_pose"] is False

    def test_get_unknown_camera_returns_404(self):
        code, body = _request("GET", f"{self.base}/cameras/unknown_cam/models")
        assert code == 404
        assert "error" in body

    def test_get_unknown_path_returns_404(self):
        code, _ = _request("GET", f"{self.base}/cameras")
        assert code == 404

    def test_health_returns_service_metadata(self):
        code, body = _request("GET", f"{self.base}/health")
        assert code == 200
        assert body["service"] == "cctv-camera-model-api"
        assert body["status"] == "ok"
        assert body["camera_count"] == 2
        assert "checked_at" in body


def test_health_direct_handler_returns_service_metadata(cameras_json: Path):
    proc = _build_processor()
    handler = _make_handler(proc, str(cameras_json), "/health")
    handler.do_GET()
    code, body = handler._responses[0]  # type: ignore[attr-defined]
    assert code == 200
    assert body["service"] == "cctv-camera-model-api"
    assert body["status"] == "ok"
    assert body["camera_count"] == 2
    assert "checked_at" in body


def test_camera_model_api_requires_internal_token_when_configured(cameras_json: Path):
    proc = _build_processor()
    handler = _make_handler(proc, str(cameras_json), "/cameras/camera_1/models")
    with patch.dict("os.environ", {"INTERNAL_SERVICE_TOKEN": "internal-secret"}):
        handler.do_GET()
    code, body = handler._responses[0]  # type: ignore[attr-defined]
    assert code == 401
    assert body["error"] == "Unauthorized"


# ===========================================================================
# POST /cameras/{id}/models
# ===========================================================================


class TestCameraModelApiPOST:

    @pytest.fixture(autouse=True)
    def setup(self, cameras_json: Path):
        self.proc = _build_processor()
        self.server = _live_server(self.proc, str(cameras_json))
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        yield
        self.server.shutdown()

    def test_post_updates_model_settings(self):
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/models",
            {"use_pose": False, "use_helmet": True},
        )
        assert code == 200
        assert body["camera_id"] == "camera_1"
        assert "model_settings" in body
        self.proc.update_camera_model_settings.assert_called_once()

    def test_post_calls_processor_with_correct_args(self):
        _request(
            "POST",
            f"{self.base}/cameras/camera_1/models",
            {"use_helmet": False},
        )
        call_args = self.proc.update_camera_model_settings.call_args[0]
        assert call_args[0] == "camera_1"
        assert "use_helmet" in call_args[1] or "helmet" in call_args[1]

    def test_post_requires_valid_key(self):
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/models",
            {"foo": "bar"},
        )
        assert code == 400
        assert "error" in body

    def test_post_invalid_json_returns_400(self):
        req = urllib.request.Request(
            f"{self.base}/cameras/camera_1/models",
            data=b"not-json",
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_post_unknown_camera_returns_404(self):
        code, body = _request(
            "POST",
            f"{self.base}/cameras/unknown_cam/models",
            {"use_pose": True},
        )
        assert code == 404
        assert "error" in body

    def test_post_unknown_path_returns_404(self):
        code, _ = _request("POST", f"{self.base}/other", {"use_pose": True})
        assert code == 404


# ===========================================================================
# 서버 기동 테스트
# ===========================================================================


class TestCameraModelApiRouting:

    def test_start_server_starts_thread(self, cameras_json: Path):
        """start_camera_model_api_server 가 데몬 스레드를 생성해야 한다."""
        proc = _build_processor()
        with patch("src.services.camera_model_api.ThreadingApiServer") as mock_srv_cls:
            mock_srv = MagicMock()
            mock_srv_cls.return_value = mock_srv
            start_camera_model_api_server(proc, str(cameras_json), 19998)
            time.sleep(0.05)
            mock_srv.serve_forever.assert_called_once()
