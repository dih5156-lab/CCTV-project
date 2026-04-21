"""외형 로그 DB + 검색 API 테스트."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

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

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, monkeypatch):
        from src.services.appearance_log import AppearanceLog
        from src.api.v1 import search as search_mod

        self.db_path = str(tmp_path / "api_test.db")
        self.log = AppearanceLog(self.db_path)
        monkeypatch.setattr(search_mod, "_log_instance", self.log)

        # crop 디렉터리 설정
        self.crop_dir = tmp_path / "crops"
        self.crop_dir.mkdir()
        monkeypatch.setattr(search_mod, "_CROP_DIR", self.crop_dir)

        from fastapi.testclient import TestClient
        from src.api.app import app

        self.client = TestClient(app)
        yield
        self.log.close()

    def test_search_empty(self):
        resp = self.client.get("/api/v1/search")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["items"] == []
        assert body["total"] == 0

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

        resp = self.client.get("/api/v1/search?time_from=1970-01-01T00:16:00&time_to=1970-01-01T00:17:00")
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
        self.log.insert(
            camera_id="cam01",
            track_id=1,
            crop_path="data/crops/cam01_1_1000.jpg",
            timestamp=1000.0,
        )
        resp = self.client.get("/api/v1/search")
        body = resp.json()
        assert body["items"][0]["crop_url"] == "/api/v1/search/crops/cam01_1_1000.jpg"

    def test_duplicate_event_id_is_ignored(self):
        event_id = "evt_same"
        assert self.log.insert(camera_id="cam01", track_id=1, event_id=event_id, timestamp=1000.0) is True
        assert self.log.insert(camera_id="cam01", track_id=2, event_id=event_id, timestamp=1005.0) is True
        assert self.log.count() == 1
