"""외형 로그 DB + 검색 API 테스트."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import httpx
import pytest

# ── AppearanceLog 단위 테스트 ────────────────────────────────────────


class TestAppearanceLog:
    """SQLite 기반 외형 기록 저장소 테스트."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from src.services.appearance_log import AppearanceLog

        self.db_path = str(tmp_path / "test.db")
        self.log = AppearanceLog(self.db_path)
        yield
        self.log.close()

    def test_insert_and_search(self):
        ok = self.log.insert(
            camera_id="cam01",
            track_id=1,
            upper_color="black",
            lower_color="blue",
            has_helmet=True,
            helmet_color="yellow",
            gender="male",
            timestamp=1000.0,
        )
        assert ok is True
        rows = self.log.search(upper_color="black")
        assert len(rows) == 1
        assert rows[0]["camera_id"] == "cam01"
        assert rows[0]["event_id"] is not None
        assert rows[0]["upper_color"] == "black"
        assert rows[0]["has_helmet"] is True
        assert rows[0]["helmet_color"] == "yellow"
        assert rows[0]["gender"] == "male"

    def test_insert_and_search_attribute_metadata(self):
        metadata = {
            "color_sources": {"upper_color": "pa100k_sgie"},
            "color_candidates": {
                "upper_color": {
                    "selected": "blue",
                    "hsv_color": "black",
                    "hsv_ratio": 0.18,
                    "lab_color": "blue",
                },
            },
        }

        ok = self.log.insert(
            camera_id="cam01",
            track_id=1,
            upper_color="blue",
            attribute_backend="pa100k_sgie",
            attribute_metadata=metadata,
            timestamp=1000.0,
        )

        assert ok is True
        rows = self.log.search(upper_color="blue")
        assert rows[0]["attribute_metadata"] == metadata

    def test_insert_cooldown(self):
        """같은 track_id로 3초 이내 재삽입하면 무시."""
        now = time.time()
        assert self.log.insert(camera_id="cam01", track_id=1, timestamp=now) is True
        assert self.log.insert(camera_id="cam01", track_id=1, timestamp=now + 1) is False
        assert self.log.insert(camera_id="cam01", track_id=1, timestamp=now + 4) is True
        assert self.log.count() == 2

    def test_search_filters(self):
        self.log.insert(camera_id="cam01", track_id=1, upper_color="black", gender="male", timestamp=1000.0)
        self.log.insert(camera_id="cam01", track_id=2, upper_color="white", gender="female", timestamp=1010.0)
        self.log.insert(camera_id="cam02", track_id=3, upper_color="black", gender="female", timestamp=1020.0)

        assert len(self.log.search(upper_color="black")) == 2
        assert len(self.log.search(gender="male")) == 1
        assert len(self.log.search(camera_id="cam02")) == 1
        assert len(self.log.search(upper_color="black", gender="female")) == 1

    def test_search_time_range(self):
        self.log.insert(camera_id="c", track_id=1, timestamp=100.0)
        self.log.insert(camera_id="c", track_id=2, timestamp=200.0)
        self.log.insert(camera_id="c", track_id=3, timestamp=300.0)

        rows = self.log.search(time_from=150.0, time_to=250.0)
        assert len(rows) == 1
        assert rows[0]["track_id"] == 2

    def test_search_pagination(self):
        for i in range(10):
            self.log.insert(camera_id="c", track_id=i, timestamp=float(i * 10))

        rows = self.log.search(limit=3, offset=0)
        assert len(rows) == 3
        rows2 = self.log.search(limit=3, offset=3)
        assert len(rows2) == 3
        # 서로 다른 결과
        ids1 = {r["id"] for r in rows}
        ids2 = {r["id"] for r in rows2}
        assert ids1.isdisjoint(ids2)

    def test_count(self):
        self.log.insert(camera_id="c", track_id=1, upper_color="red", timestamp=100.0)
        self.log.insert(camera_id="c", track_id=2, upper_color="blue", timestamp=200.0)
        assert self.log.count() == 2
        assert self.log.count(upper_color="red") == 1

    def test_face_name_search(self):
        self.log.insert(camera_id="c", track_id=1, face_name="홍길동", timestamp=100.0)
        self.log.insert(camera_id="c", track_id=2, face_name="김철수", timestamp=200.0)
        rows = self.log.search(face_name="길동")
        assert len(rows) == 1
        assert rows[0]["face_name"] == "홍길동"

    def test_bag_filters(self):
        self.log.insert(camera_id="c", track_id=1, has_backpack=True, timestamp=100.0)
        self.log.insert(camera_id="c", track_id=2, has_handbag=True, timestamp=200.0)
        assert len(self.log.search(has_backpack=True)) == 1
        assert len(self.log.search(has_handbag=True)) == 1
        assert len(self.log.search(has_backpack=False)) == 1

    def test_helmet_filters(self):
        self.log.insert(
            camera_id="c",
            track_id=1,
            has_helmet=True,
            helmet_color="yellow",
            timestamp=100.0,
        )
        self.log.insert(camera_id="c", track_id=2, has_helmet=False, timestamp=200.0)
        assert len(self.log.search(has_helmet=True)) == 1
        assert len(self.log.search(helmet_color="yellow")) == 1
        assert len(self.log.search(has_helmet=False)) == 1


# ── Search API 엔드포인트 테스트 ─────────────────────────────────────


class TestSearchAPI:
    """FastAPI search 엔드포인트 테스트."""

    class _SyncASGIClient:
        """httpx.AsyncClient를 동기 테스트에서 간단히 감싸는 래퍼."""

        def __init__(self, app):
            self._transport = httpx.ASGITransport(app=app)
            self._base_url = "http://testserver"

        def get(self, path: str):
            async def _request():
                async with httpx.AsyncClient(
                    transport=self._transport,
                    base_url=self._base_url,
                ) as client:
                    return await client.get(path)

            return asyncio.run(_request())

        def close(self) -> None:
            asyncio.run(self._transport.aclose())

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, monkeypatch):
        from src.api.v1 import search as search_mod
        from src.services.appearance_log import AppearanceLog

        self.db_path = str(tmp_path / "api_test.db")
        self.log = AppearanceLog(self.db_path)
        monkeypatch.setattr(search_mod, "_log_instance", self.log)

        # crop 디렉터리 설정
        self.crop_dir = tmp_path / "crops"
        self.crop_dir.mkdir()
        monkeypatch.setattr(search_mod, "_CROP_DIR", self.crop_dir)

        from src.api.app import app

        self.client = self._SyncASGIClient(app)
        yield
        self.client.close()
        self.log.close()

    def test_search_empty(self):
        resp = self.client.get("/api/v1/search")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["items"] == []
        assert body["total"] == 0

    def test_search_natural_language_color_query(self):
        self.log.insert(
            camera_id="cam01",
            track_id=1,
            upper_color="black",
            lower_color="red",
            timestamp=1000.0,
        )
        self.log.insert(
            camera_id="cam01",
            track_id=2,
            upper_color="black",
            lower_color="blue",
            timestamp=1010.0,
        )

        resp = self.client.get(
            "/api/v1/search?q=%EA%B2%80%EC%A0%95%EC%83%89%20%EC%83%81%EC%9D%98%20%EB%B9%A8%EA%B0%84%EC%83%89%20%ED%95%98%EC%9D%98%20%EC%82%AC%EB%9E%8C"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["upper_color"] == "black"
        assert body["items"][0]["lower_color"] == "red"

    def test_search_naive_time_range_is_interpreted_as_kst(self):
        self.log.insert(camera_id="c", track_id=1, timestamp=10.0)

        resp = self.client.get(
            "/api/v1/search?time_from=1970-01-01T09:00:10&time_to=1970-01-01T09:00:11"
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_search_with_results(self):
        self.log.insert(
            camera_id="cam01",
            track_id=1,
            upper_color="black",
            has_helmet=True,
            helmet_color="yellow",
            gender="male",
            timestamp=1000.0,
        )
        self.log.insert(camera_id="cam01", track_id=2, upper_color="white", gender="female", timestamp=1010.0)

        resp = self.client.get("/api/v1/search?upper_color=black")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        item = body["items"][0]
        assert item["upper_color"] == "black"
        assert item["has_helmet"] is True
        assert item["helmet_color"] == "yellow"
        assert item["gender"] == "male"

    def test_search_time_range(self):
        self.log.insert(camera_id="c", track_id=1, timestamp=1000.0)
        self.log.insert(camera_id="c", track_id=2, timestamp=2000.0)

        resp = self.client.get("/api/v1/search?time_from=1970-01-01T09:16:00&time_to=1970-01-01T09:17:00")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1

    def test_search_pagination(self):
        for i in range(10):
            self.log.insert(camera_id="c", track_id=i, timestamp=float(i * 10))

        resp = self.client.get("/api/v1/search?limit=3&offset=0")
        body = resp.json()
        assert len(body["items"]) == 3
        assert body["total"] == 10

    def test_crop_image_serving(self):
        # 가짜 crop 파일 생성
        (self.crop_dir / "test_1_100.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")

        resp = self.client.get("/api/v1/search/crops/test_1_100.jpg")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_crop_image_not_found(self):
        resp = self.client.get("/api/v1/search/crops/nonexist.jpg")
        assert resp.status_code == 404

    def test_crop_path_traversal_blocked(self):
        resp = self.client.get("/api/v1/search/crops/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (400, 404, 422)

    def test_search_result_has_crop_url(self):
        (self.crop_dir / "cam01_1_1000.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
        self.log.insert(
            camera_id="cam01",
            track_id=1,
            crop_path="data/crops/cam01_1_1000.jpg",
            timestamp=1000.0,
        )
        resp = self.client.get("/api/v1/search")
        body = resp.json()
        assert body["items"][0]["crop_url"] == "/api/v1/search/crops/cam01_1_1000.jpg"

    def test_search_result_hides_deleted_crop_url(self):
        self.log.insert(
            camera_id="cam01",
            track_id=1,
            crop_path="data/crops/deleted.jpg",
            timestamp=1000.0,
        )
        resp = self.client.get("/api/v1/search")
        body = resp.json()
        assert body["items"][0]["crop_url"] is None

    def test_duplicate_event_id_is_ignored(self):
        event_id = "evt_same"
        assert self.log.insert(camera_id="cam01", track_id=1, event_id=event_id, timestamp=1000.0) is True
        assert self.log.insert(camera_id="cam01", track_id=2, event_id=event_id, timestamp=1005.0) is True
        assert self.log.count() == 1


class TestAppearanceStatusAPI:
    """외형 상태 계산/엔드포인트 테스트."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, monkeypatch):
        from src.api.v1 import appearances as appearances_mod

        self.db_path = tmp_path / "appearance_status.db"
        self.appearances_mod = appearances_mod
        monkeypatch.setattr(appearances_mod, "_DB_PATH", self.db_path)
        monkeypatch.setenv("APPEARANCE_BACKEND", "hsv")
        monkeypatch.setenv("DS_APPEARANCE_ENABLED", "true")
        monkeypatch.setenv("DS_HELMET_ENABLED", "true")
        monkeypatch.setenv("DS_FACE_ENABLED", "true")
        monkeypatch.setenv("CAMERAS_JSON", str(tmp_path / "missing_cameras.json"))
        monkeypatch.delenv("PUBLIC_API_KEY", raising=False)
        yield

    def _insert_appearance_row(
        self,
        *,
        timestamp: float,
        camera_id: str = "cam01",
        track_id: int = 1,
        gender: str | None = None,
        has_helmet: bool = False,
        has_backpack: bool = False,
        has_handbag: bool = False,
        has_suitcase: bool = False,
        attribute_backend: str | None = None,
    ) -> None:
        from src.services.appearance_log import AppearanceLog

        log = AppearanceLog(str(self.db_path))
        try:
            inserted = log.insert(
                camera_id=camera_id,
                track_id=track_id,
                timestamp=timestamp,
                gender=gender,
                has_helmet=has_helmet,
                has_backpack=has_backpack,
                has_handbag=has_handbag,
                has_suitcase=has_suitcase,
                attribute_backend=attribute_backend,
            )
            assert inserted is True
        finally:
            log.close()

    def test_status_returns_runtime_stats_and_warnings(self):
        self._insert_appearance_row(timestamp=1000.0, track_id=1, gender="male", attribute_backend=None)
        self._insert_appearance_row(timestamp=1005.0, track_id=2, attribute_backend=None)

        body = asyncio.run(self.appearances_mod.get_appearance_status(None))
        body = body.model_dump(mode="json")
        assert body["success"] is True

        data = body["data"]
        assert data["db_path"].endswith("appearance_status.db")
        assert data["backend"] == "hsv"
        assert data["data_stats"]["total_records"] == 2
        assert data["data_stats"]["gender_filled"] == 1
        assert data["data_stats"]["helmet_true"] == 0
        assert data["backend_counts"] == {"unknown": 2}

        field_map = {field["field"]: field for field in data["fields"]}
        assert field_map["gender"]["ready"] is True
        assert field_map["gender"]["observed_count"] == 1
        assert field_map["has_helmet"]["ready"] is True
        assert field_map["has_helmet"]["observed_count"] == 0
        assert field_map["has_backpack"]["ready"] is False
        assert field_map["has_backpack"]["reason"] == "backend=hsv, bag_labels=none"

        warning_text = "\n".join(data["warnings"])
        assert "attribute_backend가 모두 unknown" in warning_text
        assert "has_helmet는 설정상 활성화되어 있지만 실제 적재 건수가 0" in warning_text
        assert "backend=hsv 환경에서는 bag 값이 detector nearby_objects에 의존" in warning_text
        assert any("/api/v1/appearances/status" in step for step in data["next_steps"])

    def test_status_recognizes_bag_label_alias_as_ready(self, monkeypatch):
        monkeypatch.setenv("DS_YOLO_LABELS", "person,back_pack")
        self._insert_appearance_row(
            timestamp=2000.0,
            track_id=10,
            has_backpack=True,
            attribute_backend="hsv",
        )

        data = self.appearances_mod._build_runtime_status().model_dump()
        field_map = {field["field"]: field for field in data["fields"]}

        assert data["backend_counts"] == {"hsv": 1}
        assert field_map["has_backpack"]["ready"] is True
        assert field_map["has_backpack"]["observed_count"] == 1
        assert field_map["has_backpack"]["observed_ratio"] == 1.0
        assert field_map["has_handbag"]["ready"] is True
        assert field_map["has_suitcase"]["ready"] is True

    def test_status_respects_camera_level_helmet_off(self, tmp_path: Path, monkeypatch):
        cameras_path = tmp_path / "cameras.json"
        cameras_path.write_text(
            """
            [
              {
                "id": "cam01",
                "enabled": true,
                "detections": ["person", "appearance"],
                "model_settings": {
                  "use_helmet": false,
                  "use_face": true,
                  "use_appearance": true
                }
              }
            ]
            """,
            encoding="utf-8",
        )
        monkeypatch.setenv("CAMERAS_JSON", str(cameras_path))
        self._insert_appearance_row(timestamp=2100.0, track_id=11, attribute_backend="hsv")

        data = self.appearances_mod._build_runtime_status().model_dump()
        field_map = {field["field"]: field for field in data["fields"]}
        warning_text = "\n".join(data["warnings"])

        assert field_map["has_helmet"]["enabled"] is False
        assert field_map["has_helmet"]["ready"] is False
        assert "has_helmet는 설정상 활성화되어 있지만 실제 적재 건수가 0" not in warning_text

    def test_status_uses_pphuman_label_map_for_bag_fields(self, tmp_path: Path, monkeypatch):
        label_map_path = tmp_path / "labels.json"
        label_map_path.write_text(
            """
            {
              "labels": [
                { "index": 15, "field": "has_handbag", "value": true },
                { "index": 17, "field": "has_backpack", "value": true }
              ]
            }
            """,
            encoding="utf-8",
        )
        monkeypatch.setenv("APPEARANCE_BACKEND", "pphuman")
        monkeypatch.setenv("APPEARANCE_LABEL_MAP_PATH", str(label_map_path))
        self._insert_appearance_row(timestamp=2200.0, track_id=12, attribute_backend="pphuman")

        data = self.appearances_mod._build_runtime_status().model_dump()
        field_map = {field["field"]: field for field in data["fields"]}
        warning_text = "\n".join(data["warnings"])

        assert field_map["has_backpack"]["ready"] is True
        assert field_map["has_handbag"]["ready"] is True
        assert field_map["has_suitcase"]["ready"] is False
        assert "backend=hsv 환경에서는 bag 값이 detector nearby_objects에 의존" not in warning_text

    def test_status_recognizes_pa100k_sgie_as_attribute_backend(self, tmp_path: Path, monkeypatch):
        label_map_path = tmp_path / "appearance_pa100k_labels.json"
        label_map_path.write_text(
            """
            {
              "model": "Rethinking_of_PAR PA100K resnet50",
              "labels": [
                { "index": 9, "field": "has_handbag", "value": true },
                { "index": 11, "field": "has_backpack", "value": true }
              ]
            }
            """,
            encoding="utf-8",
        )
        monkeypatch.setenv("APPEARANCE_BACKEND", "hsv")
        monkeypatch.setenv("DS_PPHUMAN_SGIE_ENABLED", "1")
        monkeypatch.setenv("APPEARANCE_LABEL_MAP_PATH", str(label_map_path))
        self._insert_appearance_row(
            timestamp=2300.0,
            track_id=13,
            has_backpack=True,
            attribute_backend="pa100k_sgie",
        )

        data = self.appearances_mod._build_runtime_status().model_dump()
        field_map = {field["field"]: field for field in data["fields"]}
        warning_text = "\n".join(data["warnings"])

        assert data["backend"] == "pa100k_sgie"
        assert data["backend_counts"] == {"pa100k_sgie": 1}
        assert field_map["has_backpack"]["source"] == "attribute_backend"
        assert field_map["has_backpack"]["ready"] is True
        assert field_map["has_handbag"]["ready"] is True
        assert field_map["has_suitcase"]["ready"] is False
        assert "backend=hsv 환경에서는 bag 값이 detector nearby_objects에 의존" not in warning_text

    def test_status_handles_missing_appearance_log_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS search_conditions (id TEXT PRIMARY KEY)")
            conn.commit()

        data = self.appearances_mod._build_runtime_status().model_dump()
        assert data["data_stats"]["total_records"] == 0
        assert data["backend_counts"] == {}
        assert any("appearance_log 데이터가 아직 없습니다" in warning for warning in data["warnings"])
