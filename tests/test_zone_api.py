"""
test_zone_api.py — ZoneApiHandler + ZoneManager.save_zones 단위 테스트

테스트 구성:
    TestZoneManager   - save_zones / _save_to_cameras_json / _save_to_zones_config
    TestZoneApiGET    - GET /cameras, GET /cameras/{id}/zones
    TestZoneApiPOST   - POST /cameras/{id}/zones (정상 / 검증 오류 / 500)
    TestZoneApiDELETE - DELETE /cameras/{id}/zones/{zone_id} (정상 / 404 / 503)
    TestZoneApiRouting - 미등록 경로 → 404
"""

import json
import os
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.zone_detection import Zone, ZoneManager
from src.services.zone_api import ZoneApiHandler, start_zone_api_server


# ---------------------------------------------------------------------------
# 공통 픽스처 / 헬퍼
# ---------------------------------------------------------------------------

SQUARE_POLYGON = [[0, 0], [100, 0], [100, 100], [0, 100]]
ZONE_DEF = {"id": "zone_1", "name": "위험구역", "polygon": SQUARE_POLYGON}

CAMERAS_JSON_DATA = [
    {
        "id": "camera_1",
        "name": "테스트 카메라",
        "enabled": True,
        "detections": ["helmet"],
        "model_settings": {"use_pose": True, "use_helmet": True, "use_person": False},
        "model_paths": {},
        "zones": [],
    },
    {
        "id": "camera_2",
        "name": "카메라 2",
        "enabled": False,
        "detections": [],
        "model_settings": {"use_pose": False, "use_helmet": False, "use_person": False},
        "model_paths": {},
        "zones": [ZONE_DEF],
    },
]


@pytest.fixture
def cameras_json(tmp_path: Path) -> Path:
    """임시 cameras.json 파일을 생성하고 경로를 반환한다."""
    p = tmp_path / "cameras.json"
    p.write_text(json.dumps(CAMERAS_JSON_DATA, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def zones_config_json(tmp_path: Path) -> Path:
    """임시 zones_config.json 파일을 생성하고 경로를 반환한다."""
    data = {
        "dwelling_threshold_seconds": 3.0,
        "cameras": {},
    }
    p = tmp_path / "zones_config.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _make_zone_manager(zones_config_path: str) -> ZoneManager:
    """ZoneManager 를 파일 없이 초기화한다."""
    with patch.object(ZoneManager, "_load_config"):
        zm = ZoneManager(zones_config=zones_config_path)
    zm.dwelling_threshold = 3.0
    return zm


def _build_processor(zone_manager=None, update_zones_ok=True) -> MagicMock:
    """가짜 VideoProcessor 를 생성한다."""
    proc = MagicMock()
    proc.zone_manager = zone_manager
    proc.update_zones = MagicMock(return_value=update_zones_ok)
    proc.get_camera_model_settings = MagicMock(
        side_effect=lambda camera_id: {
            "camera_1": {"use_pose": True, "use_helmet": True, "use_person": False},
            "camera_2": {"use_pose": False, "use_helmet": False, "use_person": False},
        }.get(camera_id)
    )
    def _update_model_settings(camera_id, settings, *_):
        if camera_id != "camera_1":
            return None
        use_helmet = bool(settings.get("use_helmet", settings.get("helmet", True)))
        use_pose = bool(settings.get("use_pose", settings.get("pose", True))) or use_helmet
        return {
            "use_pose": use_pose,
            "use_helmet": use_helmet,
            "use_person": bool(settings.get("use_person", settings.get("person", False))),
        }
    proc.update_camera_model_settings = MagicMock(
        side_effect=_update_model_settings
    )
    return proc


def _live_server(processor, cameras_json_path: str, port: int = 0):
    """실제 HTTP 서버를 스레드로 기동하여 반환한다.

    port=0 이면 OS가 사용 가능한 포트를 자동으로 할당한다.
    서버의 실제 포트는 server.server_address[1] 로 얻는다.
    (테스트 후 정리 필요)
    """
    import tempfile
    from http.server import HTTPServer
    from src.utils.zone_presets import ZonePresetStore

    server = HTTPServer(("127.0.0.1", port), ZoneApiHandler)
    server.processor = processor
    server.cameras_json_path = cameras_json_path
    server.preset_store = ZonePresetStore(
        os.path.join(tempfile.mkdtemp(), "presets.json")
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)  # 서버가 준비될 때까지 대기
    return server


def _request(method: str, url: str, body: dict | None = None):
    """urllib 래퍼 — (status_code, dict) 반환."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Length", str(len(data)))
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


# ===========================================================================
# ZoneManager.save_zones 단위 테스트
# ===========================================================================


class TestZoneManager:

    def test_save_to_cameras_json_updates_zones(self, cameras_json: Path, tmp_path: Path):
        """save_zones 호출 후 cameras.json 의 해당 카메라 zones 가 업데이트돼야 한다."""
        zm = _make_zone_manager(str(tmp_path / "zones_config.json"))
        zm.save_zones("camera_1", [ZONE_DEF], cameras_config_path=str(cameras_json))

        saved = json.loads(cameras_json.read_text(encoding="utf-8"))
        cam1 = next(c for c in saved if c["id"] == "camera_1")
        assert len(cam1["zones"]) == 1
        assert cam1["zones"][0]["id"] == "zone_1"

    def test_save_to_cameras_json_preserves_other_cameras(self, cameras_json: Path, tmp_path: Path):
        """camera_1 만 수정해도 camera_2 데이터는 유지돼야 한다."""
        zm = _make_zone_manager(str(tmp_path / "zones_config.json"))
        zm.save_zones("camera_1", [ZONE_DEF], cameras_config_path=str(cameras_json))

        saved = json.loads(cameras_json.read_text(encoding="utf-8"))
        cam2 = next(c for c in saved if c["id"] == "camera_2")
        assert cam2["zones"] == [ZONE_DEF]  # 원래 데이터 그대로

    def test_save_to_cameras_json_reloads_memory(self, cameras_json: Path, tmp_path: Path):
        """save_zones 호출 후 메모리(zm.zones)에도 즉시 반영돼야 한다."""
        zm = _make_zone_manager(str(tmp_path / "zones_config.json"))
        zm.save_zones("camera_1", [ZONE_DEF], cameras_config_path=str(cameras_json))

        assert "camera_1" in zm.zones
        assert "zone_1" in zm.zones["camera_1"]

    def test_save_to_zones_config_creates_entry(self, zones_config_json: Path):
        """zones_config_path 로 저장하면 해당 파일에 카메라 항목이 생겨야 한다."""
        zm = _make_zone_manager(str(zones_config_json))
        zm.save_zones("camera_x", [ZONE_DEF])

        saved = json.loads(zones_config_json.read_text(encoding="utf-8"))
        assert "camera_x" in saved["cameras"]
        assert saved["cameras"]["camera_x"]["zones"][0]["id"] == "zone_1"

    def test_save_to_zones_config_nonexistent_file(self, tmp_path: Path):
        """zones_config.json 이 없어도 새로 생성해 저장해야 한다."""
        path = tmp_path / "new_zones.json"
        zm = _make_zone_manager(str(path))
        zm.save_zones("camera_new", [ZONE_DEF])

        assert path.exists()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert "camera_new" in saved["cameras"]

    def test_save_zones_empty_list_clears_zones(self, cameras_json: Path, tmp_path: Path):
        """빈 리스트로 저장하면 구역이 0개가 돼야 한다."""
        zm = _make_zone_manager(str(tmp_path / "zones_config.json"))
        zm.save_zones("camera_2", [], cameras_config_path=str(cameras_json))

        saved = json.loads(cameras_json.read_text(encoding="utf-8"))
        cam2 = next(c for c in saved if c["id"] == "camera_2")
        assert cam2["zones"] == []

    def test_save_unknown_camera_id_logs_warning(
        self, cameras_json: Path, tmp_path: Path, caplog
    ):
        """cameras.json 에 없는 camera_id 로 저장해도 예외 없이 완료돼야 한다."""
        import logging
        zm = _make_zone_manager(str(tmp_path / "zones_config.json"))
        with caplog.at_level(logging.WARNING):
            zm.save_zones("nonexistent_cam", [ZONE_DEF], cameras_config_path=str(cameras_json))
        assert any("nonexistent_cam" in r.message or "찾을 수 없습니다" in r.message
                   for r in caplog.records)


# ===========================================================================
# HTTP API 통합 테스트 (실제 HTTPServer 기동)
# ===========================================================================

class TestZoneApiGET:

    @pytest.fixture(autouse=True)
    def setup(self, cameras_json: Path):
        zm = _make_zone_manager("nonexistent.json")
        zm.zones["camera_2"] = {"zone_1": Zone("zone_1", SQUARE_POLYGON, "위험구역")}
        self.proc = _build_processor(zone_manager=zm)
        self.server = _live_server(self.proc, str(cameras_json))
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        yield
        self.server.shutdown()

    def test_get_cameras_returns_list(self):
        code, body = _request("GET", f"{self.base}/cameras")
        assert code == 200
        assert isinstance(body, list)
        assert len(body) == 2

    def test_get_cameras_includes_zone_from_memory(self):
        """camera_2 의 zones 는 ZoneManager 메모리에서 읽어야 한다."""
        code, body = _request("GET", f"{self.base}/cameras")
        assert code == 200
        cam2 = next(c for c in body if c["id"] == "camera_2")
        assert len(cam2["zones"]) == 1
        assert cam2["zones"][0]["id"] == "zone_1"

    def test_get_cameras_fallback_to_json_when_no_memory(self):
        """camera_1 은 ZoneManager에 없으므로 cameras.json 의 zones(빈 배열)을 반환해야 한다."""
        code, body = _request("GET", f"{self.base}/cameras")
        cam1 = next(c for c in body if c["id"] == "camera_1")
        assert cam1["zones"] == []

    def test_get_cameras_includes_model_settings(self):
        code, body = _request("GET", f"{self.base}/cameras")
        assert code == 200
        cam1 = next(c for c in body if c["id"] == "camera_1")
        assert cam1["model_settings"]["use_pose"] is True
        assert cam1["model_settings"]["use_helmet"] is True

    def test_get_specific_camera_zones_from_memory(self):
        code, body = _request("GET", f"{self.base}/cameras/camera_2/zones")
        assert code == 200
        assert body["camera_id"] == "camera_2"
        assert len(body["zones"]) == 1

    def test_get_specific_camera_models(self):
        code, body = _request("GET", f"{self.base}/cameras/camera_1/models")
        assert code == 200
        assert body["camera_id"] == "camera_1"
        assert body["model_settings"]["use_pose"] is True

    def test_get_specific_camera_models_404(self):
        code, body = _request("GET", f"{self.base}/cameras/unknown/models")
        assert code == 404
        assert "error" in body

    def test_get_specific_camera_zones_fallback(self):
        """ZoneManager에 없는 camera_1 은 cameras.json 에서 조회해야 한다."""
        code, body = _request("GET", f"{self.base}/cameras/camera_1/zones")
        assert code == 200
        assert body["zones"] == []

    def test_get_unknown_path_returns_404(self):
        code, body = _request("GET", f"{self.base}/unknown")
        assert code == 404

    def test_get_cameras_trailing_slash(self):
        code, _ = _request("GET", f"{self.base}/cameras/")
        assert code == 200


class TestZoneApiPOST:

    @pytest.fixture(autouse=True)
    def setup(self, cameras_json: Path):
        zm = _make_zone_manager("nonexistent.json")
        self.proc = _build_processor(zone_manager=zm, update_zones_ok=True)
        self.server = _live_server(self.proc, str(cameras_json))
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        yield
        self.server.shutdown()

    def test_post_zones_success(self):
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/zones",
            {"zones": [ZONE_DEF]},
        )
        assert code == 200
        assert body["status"] == "ok"
        assert body["zones_count"] == 1

    def test_post_calls_update_zones(self):
        _request("POST", f"{self.base}/cameras/camera_1/zones", {"zones": [ZONE_DEF]})
        self.proc.update_zones.assert_called_once()
        args = self.proc.update_zones.call_args
        assert args[0][0] == "camera_1"
        assert len(args[0][1]) == 1

    def test_post_empty_zones_clears(self):
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/zones",
            {"zones": []},
        )
        assert code == 200
        assert body["zones_count"] == 0

    def test_post_invalid_json_returns_400(self):
        req = urllib.request.Request(
            f"{self.base}/cameras/camera_1/zones",
            data=b"not-json",
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_post_missing_zones_key_returns_400(self):
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/zones",
            {"data": []},
        )
        assert code == 400

    def test_post_zone_missing_id_returns_400(self):
        bad_zone = {"polygon": SQUARE_POLYGON}  # id 없음
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/zones",
            {"zones": [bad_zone]},
        )
        assert code == 400

    def test_post_polygon_too_few_points_returns_400(self):
        bad_zone = {"id": "z1", "polygon": [[0, 0], [1, 1]]}  # 점 2개
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/zones",
            {"zones": [bad_zone]},
        )
        assert code == 400

    def test_post_update_zones_failure_returns_500(self):
        self.proc.update_zones.return_value = False
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/zones",
            {"zones": [ZONE_DEF]},
        )
        assert code == 500

    def test_post_camera_models_updates_settings(self):
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/models",
            {"use_pose": False, "use_helmet": True},
        )
        assert code == 200
        assert body["camera_id"] == "camera_1"
        self.proc.update_camera_model_settings.assert_called()

    def test_post_camera_models_requires_payload(self):
        code, body = _request(
            "POST",
            f"{self.base}/cameras/camera_1/models",
            {"foo": "bar"},
        )
        assert code == 400
        assert "error" in body

    def test_post_camera_models_unknown_camera(self):
        code, body = _request(
            "POST",
            f"{self.base}/cameras/unknown/models",
            {"use_pose": True},
        )
        assert code == 404
        assert "error" in body

    def test_post_unknown_path_returns_404(self):
        code, _ = _request("POST", f"{self.base}/other", {"zones": []})
        assert code == 404


class TestZoneApiDELETE:

    @pytest.fixture(autouse=True)
    def setup(self, cameras_json: Path):
        zm = _make_zone_manager("nonexistent.json")
        zm.zones["camera_1"] = {
            "zone_1": Zone("zone_1", SQUARE_POLYGON, "구역1"),
            "zone_2": Zone("zone_2", SQUARE_POLYGON, "구역2"),
        }
        self.proc = _build_processor(zone_manager=zm, update_zones_ok=True)
        self.server = _live_server(self.proc, str(cameras_json))
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        yield
        self.server.shutdown()

    def test_delete_zone_success(self):
        code, body = _request("DELETE", f"{self.base}/cameras/camera_1/zones/zone_1")
        assert code == 200
        assert body["deleted_zone_id"] == "zone_1"

    def test_delete_calls_update_zones_with_remaining(self):
        """삭제 후 나머지 zone_2 만 update_zones 에 전달돼야 한다."""
        _request("DELETE", f"{self.base}/cameras/camera_1/zones/zone_1")
        call_args = self.proc.update_zones.call_args[0]
        remaining = call_args[1]
        ids = [z["id"] for z in remaining]
        assert "zone_1" not in ids
        assert "zone_2" in ids

    def test_delete_nonexistent_zone_returns_404(self):
        code, body = _request("DELETE", f"{self.base}/cameras/camera_1/zones/zone_999")
        assert code == 404

    def test_delete_nonexistent_camera_returns_404(self):
        code, body = _request("DELETE", f"{self.base}/cameras/ghost_cam/zones/zone_1")
        assert code == 404

    def test_delete_without_zone_manager_returns_503(self):
        self.proc.zone_manager = None
        code, body = _request("DELETE", f"{self.base}/cameras/camera_1/zones/zone_1")
        assert code == 503

    def test_delete_update_failure_returns_500(self):
        self.proc.update_zones.return_value = False
        code, body = _request("DELETE", f"{self.base}/cameras/camera_1/zones/zone_1")
        assert code == 500

    def test_delete_unknown_path_returns_404(self):
        code, _ = _request("DELETE", f"{self.base}/cameras/camera_1")
        assert code == 404


class TestZoneApiRouting:

    @pytest.fixture(autouse=True)
    def setup(self, cameras_json: Path):
        zm = _make_zone_manager("nonexistent.json")
        self.proc = _build_processor(zone_manager=zm)
        self.server = _live_server(self.proc, str(cameras_json))
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        yield
        self.server.shutdown()

    @pytest.mark.parametrize("path", [
        "/",
        "/cameras/camera_1",          # zones 없음
        "/cameras/camera_1/zones/z1/extra",  # 너무 깊은 경로
        "/health",
    ])
    def test_unknown_get_paths_return_404(self, path: str):
        code, _ = _request("GET", f"{self.base}{path}")
        assert code == 404

    def test_start_zone_api_server_starts_thread(self, cameras_json: Path):
        """start_zone_api_server 가 데몬 스레드를 생성해 서버를 기동해야 한다."""
        proc = _build_processor()
        before = threading.active_count()
        server = None
        try:
            from http.server import HTTPServer
            # 별도 포트로 기동 (내부 구현이 스레드를 생성하는지 확인)
            with patch("src.services.zone_api.HTTPServer") as mock_srv_cls:
                mock_srv = MagicMock()
                mock_srv_cls.return_value = mock_srv
                start_zone_api_server(proc, str(cameras_json), 19999)
                time.sleep(0.05)
                mock_srv.serve_forever.assert_called_once()
        finally:
            pass  # 모킹이므로 별도 정리 불필요
