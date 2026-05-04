"""
test_zone_presets.py — ZonePresetStore 단위 테스트 + 프리셋 API 통합 테스트

테스트 구성:
    TestZonePresetStore         - list_all / get / save / delete (파일 I/O)
    TestZoneApiPresetGET        - GET /zone-presets
    TestZoneApiPresetPOST       - POST /zone-presets (정상 / 검증 오류)
    TestZoneApiPresetDELETE     - DELETE /zone-presets/{id} (정상 / 404)
    TestZoneApiApplyPreset      - POST /cameras/{id}/zones/from-preset/{pid}
"""
import json
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.zone_api import ZoneApiHandler, start_zone_api_server
from src.utils.zone_presets import ZonePresetStore

# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

SAMPLE_ZONES = [
    {"id": "zone_1", "name": "입구", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
]


@pytest.fixture()
def preset_file(tmp_path):
    """빈 프리셋 파일 경로를 반환한다."""
    return str(tmp_path / "zone_presets.json")


@pytest.fixture()
def store(preset_file):
    """비어 있는 ZonePresetStore."""
    return ZonePresetStore(preset_file)


@pytest.fixture()
def store_with_one(store):
    """프리셋 1개가 있는 ZonePresetStore."""
    store.save("출입구 구역", SAMPLE_ZONES)
    return store


# ---------------------------------------------------------------------------
# ZonePresetStore 단위 테스트
# ---------------------------------------------------------------------------


class TestZonePresetStore:
    def test_list_all_empty_when_file_missing(self, tmp_path):
        s = ZonePresetStore(str(tmp_path / "nonexistent.json"))
        assert s.list_all() == []

    def test_save_returns_preset_with_fields(self, store):
        p = store.save("전기설비", SAMPLE_ZONES)
        assert p["name"] == "전기설비"
        assert p["zones"] == SAMPLE_ZONES
        assert "id" in p and len(p["id"]) == 8
        assert "created_at" in p

    def test_save_persists_to_file(self, store, preset_file):
        store.save("테스트", SAMPLE_ZONES)
        raw = json.loads(Path(preset_file).read_text(encoding="utf-8"))
        assert len(raw) == 1
        assert raw[0]["name"] == "테스트"

    def test_save_multiple_appends(self, store):
        store.save("A", SAMPLE_ZONES)
        store.save("B", SAMPLE_ZONES)
        assert len(store.list_all()) == 2

    def test_get_returns_correct_preset(self, store_with_one):
        presets = store_with_one.list_all()
        pid = presets[0]["id"]
        p = store_with_one.get(pid)
        assert p is not None
        assert p["name"] == "출입구 구역"

    def test_get_missing_returns_none(self, store):
        assert store.get("nope") is None

    def test_delete_existing_returns_true(self, store_with_one):
        pid = store_with_one.list_all()[0]["id"]
        assert store_with_one.delete(pid) is True
        assert store_with_one.list_all() == []

    def test_delete_missing_returns_false(self, store):
        assert store.delete("ghost") is False

    def test_list_all_graceful_on_bad_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("NOT JSON", encoding="utf-8")
        s = ZonePresetStore(str(bad))
        assert s.list_all() == []


# ---------------------------------------------------------------------------
# 공통 API 서버 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_server(tmp_path):
    """실제 HTTPServer + 목(mock) VideoProcessor로 Zone API를 실행한다."""
    cameras_file = tmp_path / "cameras.json"
    cameras_file.write_text(
        json.dumps([{"id": "cam1", "name": "카메라1", "enabled": True, "zones": [], "detections": []}]),
        encoding="utf-8",
    )
    presets_file = str(tmp_path / "zone_presets.json")

    processor = MagicMock()
    processor.zone_manager = None  # 기본적으로 비활성화

    try:
        server = HTTPServer(("127.0.0.1", 0), ZoneApiHandler)
    except PermissionError as exc:
        pytest.skip(f"이 환경에서는 로컬 소켓 바인딩이 허용되지 않음: {exc}")
    server.processor = processor
    server.cameras_json_path = str(cameras_file)
    server.preset_store = ZonePresetStore(presets_file)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    port = server.server_address[1]

    yield server, port, processor

    server.shutdown()


def _get(port, path) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def _post(port, path, body: dict) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json",
                                          "Connection": "close"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _delete(port, path) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(url, method="DELETE",
                                 headers={"Connection": "close"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ---------------------------------------------------------------------------
# GET /zone-presets
# ---------------------------------------------------------------------------


class TestZoneApiPresetGET:
    def test_empty_list(self, api_server):
        _, port, _ = api_server
        code, body = _get(port, "/zone-presets")
        assert code == 200
        assert body == []

    def test_returns_saved_presets(self, api_server):
        server, port, _ = api_server
        server.preset_store.save("A구역", SAMPLE_ZONES)
        server.preset_store.save("B구역", SAMPLE_ZONES)
        code, body = _get(port, "/zone-presets")
        assert code == 200
        assert len(body) == 2
        names = [p["name"] for p in body]
        assert "A구역" in names and "B구역" in names


# ---------------------------------------------------------------------------
# POST /zone-presets
# ---------------------------------------------------------------------------


class TestZoneApiPresetPOST:
    def test_save_preset_201(self, api_server):
        _, port, _ = api_server
        code, body = _post(port, "/zone-presets",
                           {"name": "전기설비", "zones": SAMPLE_ZONES})
        assert code == 201
        assert body["name"] == "전기설비"
        assert "id" in body and "created_at" in body

    def test_save_preset_missing_name_400(self, api_server):
        _, port, _ = api_server
        code, body = _post(port, "/zone-presets",
                           {"name": "  ", "zones": SAMPLE_ZONES})
        assert code == 400
        assert "name" in body.get("error", "")

    def test_save_preset_missing_zones_400(self, api_server):
        _, port, _ = api_server
        code, body = _post(port, "/zone-presets", {"name": "테스트"})
        assert code == 400
        assert "zones" in body.get("error", "")

    def test_save_preset_bad_json_400(self, api_server):
        _, port, _ = api_server
        url = f"http://127.0.0.1:{port}/zone-presets"
        req = urllib.request.Request(url, data=b"NOT JSON",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                code = resp.status
        except urllib.error.HTTPError as e:
            code = e.code
        assert code == 400


# ---------------------------------------------------------------------------
# DELETE /zone-presets/{preset_id}
# ---------------------------------------------------------------------------


class TestZoneApiPresetDELETE:
    def test_delete_existing_200(self, api_server):
        server, port, _ = api_server
        p = server.preset_store.save("삭제대상", SAMPLE_ZONES)
        code, body = _delete(port, f"/zone-presets/{p['id']}")
        assert code == 200
        assert body["deleted_preset_id"] == p["id"]
        assert server.preset_store.get(p["id"]) is None

    def test_delete_missing_404(self, api_server):
        _, port, _ = api_server
        code, _ = _delete(port, "/zone-presets/ghost")
        assert code == 404


# ---------------------------------------------------------------------------
# POST /cameras/{id}/zones/from-preset/{preset_id}
# ---------------------------------------------------------------------------


class TestZoneApiApplyPreset:
    def test_apply_existing_preset_200(self, api_server):
        server, port, processor = api_server
        processor.update_zones.return_value = True
        p = server.preset_store.save("적용테스트", SAMPLE_ZONES)
        code, body = _post(port, f"/cameras/cam1/zones/from-preset/{p['id']}", {})
        assert code == 200
        assert body["preset_id"] == p["id"]
        assert body["zones_count"] == len(SAMPLE_ZONES)
        processor.update_zones.assert_called_once()

    def test_apply_missing_preset_404(self, api_server):
        _, port, _ = api_server
        code, _ = _post(port, "/cameras/cam1/zones/from-preset/ghost", {})
        assert code == 404

    def test_apply_processor_failure_500(self, api_server):
        server, port, processor = api_server
        processor.update_zones.return_value = False
        p = server.preset_store.save("실패테스트", SAMPLE_ZONES)
        code, body = _post(port, f"/cameras/cam1/zones/from-preset/{p['id']}", {})
        assert code == 500
        assert "error" in body
