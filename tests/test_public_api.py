"""CCTV Public API 테스트.

httpx ASGI transport로 각 엔드포인트의 기본 동작을 검증한다.
내부 서비스(action-layer, alert-api)는 httpx mock으로 격리한다.
"""

from __future__ import annotations

import json
import os
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.api.app import app


class SyncASGIClient:
    """동기 테스트에서 ASGI 앱을 직접 호출하는 작은 래퍼."""

    def __init__(self, asgi_app):
        self._transport = httpx.ASGITransport(app=asgi_app)
        self._base_url = "http://testserver"

    def request(self, method: str, path: str, **kwargs):
        async def _request():
            async with httpx.AsyncClient(
                transport=self._transport,
                base_url=self._base_url,
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(_request())

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)

    def close(self) -> None:
        asyncio.run(self._transport.aclose())


@pytest.fixture
def client():
    client = SyncASGIClient(app)
    yield client
    client.close()


# ---------------------------------------------------------------------------
# /api/v1/health
# ---------------------------------------------------------------------------


def test_root_guides_browser_users(client: SyncASGIClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "cctv-public-api"
    assert data["docs"] == "/docs"
    assert data["health"] == "/api/v1/health"
    assert data["events"] == "/api/v1/events"


def test_health_up(client: SyncASGIClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["status"] == "up"
    assert data["data"]["service"] == "cctv-public-api"


def test_readiness_up_when_dependencies_are_healthy(client: SyncASGIClient) -> None:
    async def _mock_get(self, url, *args, **kwargs):
        mock = MagicMock()
        mock.status_code = 200
        return mock

    with patch("httpx.AsyncClient.get", new=_mock_get):
        resp = client.get("/api/v1/readiness")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ready"
    assert {dep["name"] for dep in body["data"]["dependencies"]} == {
        "action-layer",
        "alert-api",
    }


def test_readiness_degraded_when_dependency_is_down(client: SyncASGIClient) -> None:
    async def _mock_get(self, url, *args, **kwargs):
        mock = MagicMock()
        mock.status_code = 503 if "cctv-action-layer" in url else 200
        return mock

    with patch("httpx.AsyncClient.get", new=_mock_get):
        resp = client.get("/api/v1/readiness")

    assert resp.status_code == 503
    body = resp.json()
    assert body["success"] is False
    assert body["data"]["status"] == "degraded"
    statuses = {dep["name"]: dep["status"] for dep in body["data"]["dependencies"]}
    assert statuses["action-layer"] == "down"
    assert statuses["alert-api"] == "up"


def test_metrics_counter_records_http_requests(client: SyncASGIClient) -> None:
    client.get("/api/v1/health")
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "cctv_public_api_http_requests_total" in body
    assert 'path_prefix="/api/v1/health"' in body


# ---------------------------------------------------------------------------
# /api/v1/alerts
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_post_alert_success(self, client: SyncASGIClient) -> None:
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

    def test_post_alert_invalid_event_type(self, client: SyncASGIClient) -> None:
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


    def test_post_alert_missing_required_field(self, client: SyncASGIClient) -> None:
        """필수 필드 누락 → 422"""
        payload = {"event_type": "helmet", "confidence": 0.9}
        resp = client.post("/api/v1/alerts", json=payload)
        assert resp.status_code == 422
        assert resp.json()["success"] is False

    def test_post_alert_confidence_out_of_range(self, client: SyncASGIClient) -> None:
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

    def test_post_alert_fallback_on_internal_error(self, client: SyncASGIClient, tmp_path: Path) -> None:
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
# /api/v1/sensor-readings
# ---------------------------------------------------------------------------


class TestSensorReadings:
    def test_get_sensor_readings_returns_latest_tlv_logs(
        self,
        client: SyncASGIClient,
        tmp_path: Path,
    ) -> None:
        import src.api.v1.sensor_readings as sensor_module

        log_file = tmp_path / "sensor_readings.jsonl"
        map_file = tmp_path / "sensor_devices.json"
        map_file.write_text(
            json.dumps({"devices": [{"device_id": "sensor-02", "name": "설비실 온도 센서"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        log_file.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "receivedAt": "2026-05-14T01:00:00+00:00",
                            "payload": {
                                "device_id": "sensor-01",
                                "table": "t34957",
                                "data": {"temperature": 24.5, "angle_x": 1.2},
                                "received_at": 1778720400000,
                            },
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "receivedAt": "2026-05-14T01:01:00+00:00",
                            "payload": {
                                "device_id": "sensor-02",
                                "table": "t34958",
                                "data": {"temperature": 31.5},
                            },
                        },
                        ensure_ascii=False,
                    ),
                ]
            ),
            encoding="utf-8",
        )
        original = sensor_module._SENSOR_LOG
        original_map = sensor_module._SENSOR_DEVICE_MAP
        sensor_module._SENSOR_LOG = log_file
        sensor_module._SENSOR_DEVICE_MAP = map_file
        try:
            resp = client.get("/api/v1/sensor-readings?limit=10")
        finally:
            sensor_module._SENSOR_LOG = original
            sensor_module._SENSOR_DEVICE_MAP = original_map

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["items"][0]["device_id"] == "sensor-02"
        assert body["items"][0]["device_name"] == "설비실 온도 센서"
        assert body["items"][1]["table"] == "t34957"
        assert body["items"][1]["data"]["temperature"] == 24.5

    def test_post_sensor_reading_forwards_to_alert_api(self, client: SyncASGIClient) -> None:
        payload = {
            "device_id": "demo-tlv-01",
            "table": "t34957",
            "data": {"temperature": 26.2, "angle_x": 3.1},
            "received_at": 1778720400000,
        }

        async def _mock_post(*args, **kwargs):
            mock = MagicMock()
            mock.status_code = 202
            mock.raise_for_status = MagicMock()
            return mock

        with patch("httpx.AsyncClient.post", new=_mock_post):
            resp = client.post("/api/v1/sensor-readings", json=payload)

        assert resp.status_code == 202
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["accepted"] is True
        assert body["data"]["device_id"] == "demo-tlv-01"
        assert body["data"]["table"] == "t34957"


# ---------------------------------------------------------------------------
# /api/v1/events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_list_events_empty_log(self, client: SyncASGIClient) -> None:
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

    def test_list_events_with_data(self, client: SyncASGIClient, tmp_path: Path) -> None:
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

    def test_list_events_camera_filter(self, client: SyncASGIClient, tmp_path: Path) -> None:
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

    def test_list_events_pagination(self, client: SyncASGIClient, tmp_path: Path) -> None:
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

    def test_list_events_time_filter(self, client: SyncASGIClient, tmp_path: Path) -> None:
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

    def test_list_events_accepts_iso_timestamp(self, client: SyncASGIClient, tmp_path: Path) -> None:
        """실사용 로그의 ISO timestamp도 Unix seconds로 변환해 반환한다."""
        import src.api.v1.events as events_module

        log_file = tmp_path / "test_events_iso_timestamp.jsonl"
        entry = {
            "receivedAt": "2026-05-06T01:55:27.452483+00:00",
            "payload": {
                "camera_id": "camera_1",
                "type": "fall_detected",
                "severity": "critical",
                "confidence": 0.92,
                "timestamp": "2026-05-06T01:55:27.452483+00:00",
            },
        }
        log_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        original = events_module._ALERT_LOG
        events_module._ALERT_LOG = log_file
        try:
            resp = client.get("/api/v1/events?limit=1")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 1
            assert body["items"][0]["timestamp"] == pytest.approx(1778032527.452483)
            assert body["items"][0]["received_at"] == "2026-05-06T01:55:27.452483Z"
        finally:
            events_module._ALERT_LOG = original

    def test_list_events_reads_nested_alert_event(self, client: SyncASGIClient, tmp_path: Path) -> None:
        """Alert API가 저장하는 payload.event 포맷을 파싱한다."""
        import src.api.v1.events as events_module

        log_file = tmp_path / "test_nested_alert_event.jsonl"
        entry = {
            "receivedAt": "2026-05-06T01:57:55.574782+00:00",
            "payload": {
                "topic": "cctv/ai/events/camera_1/person",
                "event": {
                    "camera_id": "camera_1",
                    "type": "person",
                    "confidence": 0.858,
                    "timestamp": 1778032673.848273,
                    "object_id": 28,
                    "event": {
                        "event_type": "person",
                        "severity": "normal",
                    },
                    "raw": {
                        "bbox": {"x": 346, "y": 108, "width": 1570, "height": 971},
                        "metadata": {"direction": "left"},
                    },
                },
            },
        }
        log_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        original = events_module._ALERT_LOG
        events_module._ALERT_LOG = log_file
        try:
            resp = client.get("/api/v1/events?limit=1")
            assert resp.status_code == 200
            item = resp.json()["items"][0]
            assert item["camera_id"] == "camera_1"
            assert item["event_type"] == "person"
            assert item["confidence"] == pytest.approx(0.858)
            assert item["bbox"]["x"] == 346
            assert item["metadata"]["direction"] == "left"
        finally:
            events_module._ALERT_LOG = original

    def test_list_events_event_type_filter(self, client: SyncASGIClient, tmp_path: Path) -> None:
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

    def test_list_events_combined_filters(self, client: SyncASGIClient, tmp_path: Path) -> None:
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
    def test_list_cameras_no_file(self, client: SyncASGIClient) -> None:
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

    def test_list_cameras_strips_credentials(self, client: SyncASGIClient, tmp_path: Path) -> None:
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

    def test_get_camera_not_found_uses_wrapped_error(self, client: SyncASGIClient) -> None:
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

    def test_get_camera_not_found(self, client: SyncASGIClient, tmp_path: Path) -> None:
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
    def test_no_key_when_not_configured(self, client: SyncASGIClient) -> None:
        """PUBLIC_API_KEY 미설정 시 인증 없이 통과."""
        env = os.environ.copy()
        env.pop("PUBLIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_valid_key_accepted(self, client: SyncASGIClient) -> None:
        """올바른 API Key → 통과."""
        with patch.dict(os.environ, {"PUBLIC_API_KEY": "test-secret-key"}):
            resp = client.get("/api/v1/health", headers={"X-API-Key": "test-secret-key"})
        assert resp.status_code == 200

    def test_invalid_key_rejected(self, client: SyncASGIClient, tmp_path: Path) -> None:
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

    def test_missing_key_when_configured(self, client: SyncASGIClient, tmp_path: Path) -> None:
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

    def test_query_param_key_rejected_by_default(self, client: SyncASGIClient, tmp_path: Path) -> None:
        """기본 설정에서는 ?api_key= 쿼리 파라미터 인증을 허용하지 않는다."""
        import src.api.v1.cameras as cam_module

        cameras_file = tmp_path / "cameras.json"
        cameras_file.write_text(json.dumps([]), encoding="utf-8")
        original = cam_module._CAMERAS_JSON
        cam_module._CAMERAS_JSON = cameras_file
        try:
            with patch.dict(os.environ, {"PUBLIC_API_KEY": "test-secret-key"}):
                resp = client.get("/api/v1/cameras?api_key=test-secret-key")
            assert resp.status_code == 401
        finally:
            cam_module._CAMERAS_JSON = original


# ---------------------------------------------------------------------------
# 응답 공통 형식 검증
# ---------------------------------------------------------------------------


class TestResponseFormat:
    def test_success_response_has_required_fields(self, client: SyncASGIClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "data" in body
        assert "timestamp" in body

    def test_cameras_response_format(self, client: SyncASGIClient, tmp_path: Path) -> None:
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
