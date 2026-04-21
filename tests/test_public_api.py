"""CCTV Public API 테스트.

FastAPI TestClient를 사용해 각 엔드포인트의 기본 동작을 검증한다.
내부 서비스(action-layer, alert-api)는 httpx mock으로 격리한다.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/v1/health
# ---------------------------------------------------------------------------


def test_health_up(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["status"] == "up"
    assert data["data"]["service"] == "cctv-public-api"


# ---------------------------------------------------------------------------
# /api/v1/alerts
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_post_alert_success(self, client: TestClient) -> None:
        """유효한 payload로 POST → 202 Accepted."""
        payload = {
            "camera_id": "cam-01",
            "event_type": "helmet",
            "severity": "normal",
            "confidence": 0.95,
            "timestamp": 1700000000.0,
            "bbox": {"x": 10, "y": 20, "width": 100, "height": 80},
        }

        async def _mock_post(*args, **kwargs):
            mock = MagicMock()
            mock.status_code = 202
            mock.raise_for_status = MagicMock()
            return mock

        with patch("httpx.AsyncClient.post", new=_mock_post):
            resp = client.post("/api/v1/alerts", json=payload)

        assert resp.status_code == 202
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["accepted"] is True
        assert body["data"]["camera_id"] == "cam-01"
        assert body["data"]["event_type"] == "helmet"

    def test_post_alert_invalid_event_type(self, client: TestClient) -> None:
        """유효하지 않은 event_type → 422"""
        payload = {
            "camera_id": "cam-01",
            "event_type": "invalid_type",
            "confidence": 0.9,
            "timestamp": 1700000000.0,
        }
        resp = client.post("/api/v1/alerts", json=payload)
        assert resp.status_code == 422
        body = resp.json()
        assert body["success"] is False
        assert body["error"]

    def test_post_alert_missing_required_field(self, client: TestClient) -> None:
        """필수 필드 누락 → 422"""
        payload = {"event_type": "helmet", "confidence": 0.9}
        resp = client.post("/api/v1/alerts", json=payload)
        assert resp.status_code == 422
        assert resp.json()["success"] is False

    def test_post_alert_confidence_out_of_range(self, client: TestClient) -> None:
        """confidence 범위 초과 → 422"""
        payload = {
            "camera_id": "cam-01",
            "event_type": "helmet",
            "confidence": 1.5,
            "timestamp": 1700000000.0,
        }
        resp = client.post("/api/v1/alerts", json=payload)
        assert resp.status_code == 422
        assert resp.json()["success"] is False

    def test_post_alert_fallback_on_internal_error(self, client: TestClient, tmp_path: Path) -> None:
        """내부 alert-api 실패 시 fallback 파일에 저장되고 202 반환."""
        import src.api.v1.alerts as alerts_module

        original_log = alerts_module._FALLBACK_LOG
        alerts_module._FALLBACK_LOG = tmp_path / "fallback.jsonl"

        import httpx

        async def _fail(*args, **kwargs):
            raise httpx.ConnectError("연결 실패")

        try:
            with patch("httpx.AsyncClient.post", new=_fail):
                resp = client.post(
                    "/api/v1/alerts",
                    json={
                        "camera_id": "cam-02",
                        "event_type": "fall_detected",
                        "severity": "critical",
                        "confidence": 0.99,
                        "timestamp": 1700000001.0,
                    },
                )
            assert resp.status_code == 202
            assert alerts_module._FALLBACK_LOG.exists()
        finally:
            alerts_module._FALLBACK_LOG = original_log


# ---------------------------------------------------------------------------
# /api/v1/events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_list_events_empty_log(self, client: TestClient) -> None:
        """로그 파일 없을 때 → 빈 목록 반환."""
        import src.api.v1.events as events_module

        original = events_module._ALERT_LOG
        events_module._ALERT_LOG = Path("/nonexistent/path/events.jsonl")
        try:
            resp = client.get("/api/v1/events")
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["items"] == []
            assert body["total"] == 0
        finally:
            events_module._ALERT_LOG = original

    def test_list_events_with_data(self, client: TestClient, tmp_path: Path) -> None:
        """JSONL 파일이 있을 때 파싱 및 반환 확인."""
        import src.api.v1.events as events_module

        log_file = tmp_path / "test_events.jsonl"
        entries = [
            {
                "receivedAt": "2024-01-01T00:00:00",
                "payload": {
                    "camera_id": "cam-01",
                    "type": "helmet",
                    "severity": "normal",
                    "confidence": 0.9,
                    "timestamp": 1700000000.0,
                },
            },
            {
                "receivedAt": "2024-01-01T00:01:00",
                "payload": {
                    "camera_id": "cam-02",
                    "type": "fall_detected",
                    "severity": "critical",
                    "confidence": 0.99,
                    "timestamp": 1700000060.0,
                },
            },
        ]
        with log_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        original = events_module._ALERT_LOG
        events_module._ALERT_LOG = log_file
        try:
            resp = client.get("/api/v1/events")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 2
            assert len(body["items"]) == 2
        finally:
            events_module._ALERT_LOG = original

    def test_list_events_camera_filter(self, client: TestClient, tmp_path: Path) -> None:
        """camera_id 필터링 동작 확인."""
        import src.api.v1.events as events_module

        log_file = tmp_path / "test_events_filter.jsonl"
        entries = [
            {
                "receivedAt": "2024-01-01T00:00:00",
                "payload": {"camera_id": "cam-A", "type": "helmet", "severity": "normal", "confidence": 0.9, "timestamp": 1700000000.0},
            },
            {
                "receivedAt": "2024-01-01T00:01:00",
                "payload": {"camera_id": "cam-B", "type": "helmet", "severity": "normal", "confidence": 0.8, "timestamp": 1700000060.0},
            },
        ]
        with log_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        original = events_module._ALERT_LOG
        events_module._ALERT_LOG = log_file
        try:
            resp = client.get("/api/v1/events?camera_id=cam-A")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 1
            assert body["items"][0]["camera_id"] == "cam-A"
        finally:
            events_module._ALERT_LOG = original

    def test_list_events_pagination(self, client: TestClient, tmp_path: Path) -> None:
        """페이지네이션 파라미터 동작 확인."""
        import src.api.v1.events as events_module

        log_file = tmp_path / "test_events_paged.jsonl"
        with log_file.open("w", encoding="utf-8") as f:
            for i in range(10):
                entry = {
                    "receivedAt": "2024-01-01T00:00:00",
                    "payload": {"camera_id": f"cam-{i:02d}", "type": "person", "severity": "normal", "confidence": 0.8, "timestamp": float(1700000000 + i)},
                }
                f.write(json.dumps(entry) + "\n")

        original = events_module._ALERT_LOG
        events_module._ALERT_LOG = log_file
        try:
            resp = client.get("/api/v1/events?limit=3&offset=0")
            body = resp.json()
            assert body["total"] == 10
            assert len(body["items"]) == 3
            assert body["limit"] == 3
        finally:
            events_module._ALERT_LOG = original

    def test_list_events_time_filter(self, client: TestClient, tmp_path: Path) -> None:
        """time_from / time_to 필터 동작 확인."""
        import src.api.v1.events as events_module

        log_file = tmp_path / "test_events_time.jsonl"
        entries = [
            {"receivedAt": "2024-01-01T00:00:00", "payload": {"camera_id": "cam-01", "type": "helmet", "severity": "normal", "confidence": 0.9, "timestamp": 1700000000.0}},
            {"receivedAt": "2024-01-01T00:01:00", "payload": {"camera_id": "cam-01", "type": "helmet", "severity": "normal", "confidence": 0.9, "timestamp": 1700000060.0}},
            {"receivedAt": "2024-01-01T00:02:00", "payload": {"camera_id": "cam-01", "type": "helmet", "severity": "normal", "confidence": 0.9, "timestamp": 1700000120.0}},
        ]
        with log_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        original = events_module._ALERT_LOG
        events_module._ALERT_LOG = log_file
        try:
            # 중간 구간만 선택 (time_from=1700000030, time_to=1700000090)
            resp = client.get("/api/v1/events?time_from=1700000030&time_to=1700000090")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 1
            assert body["items"][0]["timestamp"] == pytest.approx(1700000060.0)
        finally:
            events_module._ALERT_LOG = original

    def test_list_events_event_type_filter(self, client: TestClient, tmp_path: Path) -> None:
        """event_type 필터 동작 확인."""
        import src.api.v1.events as events_module

        log_file = tmp_path / "test_events_etype.jsonl"
        entries = [
            {"receivedAt": "2024-01-01T00:00:00", "payload": {"camera_id": "cam-01", "type": "helmet", "severity": "normal", "confidence": 0.9, "timestamp": 1700000000.0}},
            {"receivedAt": "2024-01-01T00:01:00", "payload": {"camera_id": "cam-01", "type": "fall_detected", "severity": "critical", "confidence": 0.99, "timestamp": 1700000060.0}},
            {"receivedAt": "2024-01-01T00:02:00", "payload": {"camera_id": "cam-01", "type": "helmet", "severity": "normal", "confidence": 0.85, "timestamp": 1700000120.0}},
        ]
        with log_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        original = events_module._ALERT_LOG
        events_module._ALERT_LOG = log_file
        try:
            resp = client.get("/api/v1/events?event_type=fall_detected")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 1
            assert body["items"][0]["event_type"] == "fall_detected"
        finally:
            events_module._ALERT_LOG = original

    def test_list_events_combined_filters(self, client: TestClient, tmp_path: Path) -> None:
        """camera_id + event_type + time_from 복합 필터 확인."""
        import src.api.v1.events as events_module

        log_file = tmp_path / "test_events_combined.jsonl"
        entries = [
            {"receivedAt": "2024-01-01T00:00:00", "payload": {"camera_id": "cam-A", "type": "helmet", "severity": "normal", "confidence": 0.9, "timestamp": 1700000000.0}},
            {"receivedAt": "2024-01-01T00:01:00", "payload": {"camera_id": "cam-A", "type": "helmet", "severity": "normal", "confidence": 0.9, "timestamp": 1700000060.0}},
            {"receivedAt": "2024-01-01T00:02:00", "payload": {"camera_id": "cam-B", "type": "helmet", "severity": "normal", "confidence": 0.9, "timestamp": 1700000120.0}},
        ]
        with log_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        original = events_module._ALERT_LOG
        events_module._ALERT_LOG = log_file
        try:
            resp = client.get("/api/v1/events?camera_id=cam-A&event_type=helmet&time_from=1700000050")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 1
            assert body["items"][0]["camera_id"] == "cam-A"
            assert body["items"][0]["timestamp"] == pytest.approx(1700000060.0)
        finally:
            events_module._ALERT_LOG = original


# ---------------------------------------------------------------------------
# /api/v1/cameras
# ---------------------------------------------------------------------------


class TestCameras:
    def test_list_cameras_no_file(self, client: TestClient) -> None:
        """cameras.json 없을 때 빈 목록 반환."""
        import src.api.v1.cameras as cam_module

        original = cam_module._CAMERAS_JSON
        cam_module._CAMERAS_JSON = Path("/nonexistent/cameras.json")
        try:
            resp = client.get("/api/v1/cameras")
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["data"] == []
        finally:
            cam_module._CAMERAS_JSON = original

    def test_list_cameras_strips_credentials(self, client: TestClient, tmp_path: Path) -> None:
        """RTSP URL에서 자격증명이 제거되는지 확인."""
        import src.api.v1.cameras as cam_module

        cameras_file = tmp_path / "cameras.json"
        cameras_file.write_text(
            json.dumps(
                [{"id": "cam-01", "name": "정문", "url": "rtsp://admin:secret@192.168.1.10:554/stream"}]
            ),
            encoding="utf-8",
        )
        original = cam_module._CAMERAS_JSON
        cam_module._CAMERAS_JSON = cameras_file
        try:
            resp = client.get("/api/v1/cameras")
            assert resp.status_code == 200
            url = resp.json()["data"][0]["url"]
            assert "secret" not in url
            assert "admin" not in url
        finally:
            cam_module._CAMERAS_JSON = original

    def test_get_camera_not_found_uses_wrapped_error(self, client: TestClient) -> None:
        import src.api.v1.cameras as cam_module

        original = cam_module._CAMERAS_JSON
        cam_module._CAMERAS_JSON = Path("/nonexistent/cameras.json")
        try:
            resp = client.get("/api/v1/cameras/missing-camera")
            assert resp.status_code == 404
            body = resp.json()
            assert body["success"] is False
            assert body["error"] == "카메라를 찾을 수 없습니다."
            assert "timestamp" in body
        finally:
            cam_module._CAMERAS_JSON = original

    def test_get_camera_not_found(self, client: TestClient, tmp_path: Path) -> None:
        """존재하지 않는 camera_id → 404."""
        import src.api.v1.cameras as cam_module

        cameras_file = tmp_path / "cameras.json"
        cameras_file.write_text(json.dumps([{"id": "cam-01", "name": "정문"}]), encoding="utf-8")
        original = cam_module._CAMERAS_JSON
        cam_module._CAMERAS_JSON = cameras_file
        try:
            resp = client.get("/api/v1/cameras/nonexistent")
            assert resp.status_code == 404
        finally:
            cam_module._CAMERAS_JSON = original


# ---------------------------------------------------------------------------
# API Key 인증
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_key_when_not_configured(self, client: TestClient) -> None:
        """PUBLIC_API_KEY 미설정 시 인증 없이 통과."""
        env = os.environ.copy()
        env.pop("PUBLIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_valid_key_accepted(self, client: TestClient) -> None:
        """올바른 API Key → 통과."""
        with patch.dict(os.environ, {"PUBLIC_API_KEY": "test-secret-key"}):
            resp = client.get("/api/v1/health", headers={"X-API-Key": "test-secret-key"})
        assert resp.status_code == 200

    def test_invalid_key_rejected(self, client: TestClient, tmp_path: Path) -> None:
        """잘못된 API Key → 403 (카메라 목록 조회 사용)."""
        import src.api.v1.cameras as cam_module

        cameras_file = tmp_path / "cameras.json"
        cameras_file.write_text(json.dumps([]), encoding="utf-8")
        original = cam_module._CAMERAS_JSON
        cam_module._CAMERAS_JSON = cameras_file
        try:
            with patch.dict(os.environ, {"PUBLIC_API_KEY": "test-secret-key"}):
                resp = client.get("/api/v1/cameras", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 403
        finally:
            cam_module._CAMERAS_JSON = original

    def test_missing_key_when_configured(self, client: TestClient, tmp_path: Path) -> None:
        """API Key 설정됐는데 헤더 누락 → 401."""
        import src.api.v1.cameras as cam_module

        cameras_file = tmp_path / "cameras.json"
        cameras_file.write_text(json.dumps([]), encoding="utf-8")
        original = cam_module._CAMERAS_JSON
        cam_module._CAMERAS_JSON = cameras_file
        try:
            with patch.dict(os.environ, {"PUBLIC_API_KEY": "test-secret-key"}):
                resp = client.get("/api/v1/cameras")
            assert resp.status_code == 401
        finally:
            cam_module._CAMERAS_JSON = original

    def test_query_param_key_accepted(self, client: TestClient, tmp_path: Path) -> None:
        """?api_key= 쿼리 파라미터로도 인증 가능."""
        import src.api.v1.cameras as cam_module

        cameras_file = tmp_path / "cameras.json"
        cameras_file.write_text(json.dumps([]), encoding="utf-8")
        original = cam_module._CAMERAS_JSON
        cam_module._CAMERAS_JSON = cameras_file
        try:
            with patch.dict(os.environ, {"PUBLIC_API_KEY": "test-secret-key"}):
                resp = client.get("/api/v1/cameras?api_key=test-secret-key")
            assert resp.status_code == 200
        finally:
            cam_module._CAMERAS_JSON = original


# ---------------------------------------------------------------------------
# 응답 공통 형식 검증
# ---------------------------------------------------------------------------


class TestResponseFormat:
    def test_success_response_has_required_fields(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "data" in body
        assert "timestamp" in body

    def test_cameras_response_format(self, client: TestClient, tmp_path: Path) -> None:
        """BaseResponse 래퍼 형식 (success, data, error, timestamp) 검증."""
        import src.api.v1.cameras as cam_module

        cameras_file = tmp_path / "cameras.json"
        cameras_file.write_text(json.dumps([]), encoding="utf-8")
        original = cam_module._CAMERAS_JSON
        cam_module._CAMERAS_JSON = cameras_file
        try:
            resp = client.get("/api/v1/cameras")
            body = resp.json()
            assert "success" in body
            assert "data" in body
            assert "timestamp" in body
            assert body["error"] is None
        finally:
            cam_module._CAMERAS_JSON = original
