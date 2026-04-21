"""
test_sensor.py — SirenDevice / SensorConfig 단위 테스트

전략: requests.post를 mock 처리하여 실제 경광등 없이 HTTP 제어 흐름을 검증한다.
"""
import threading
from unittest.mock import MagicMock, patch, call

import pytest

from src.devices.siren import SensorConfig, SirenDevice


# ---------------------------------------------------------------------------
# SensorConfig 테스트
# ---------------------------------------------------------------------------

class TestSensorConfig:
    def test_is_configured_all_fields(self):
        cfg = SensorConfig(host="192.168.0.1", username="admin", password="pw")
        assert cfg.is_configured is True

    def test_is_configured_missing_host(self):
        cfg = SensorConfig(username="admin", password="pw")
        assert cfg.is_configured is False

    def test_is_configured_missing_username(self):
        cfg = SensorConfig(host="192.168.0.1", password="pw")
        assert cfg.is_configured is False

    def test_is_configured_missing_password(self):
        cfg = SensorConfig(host="192.168.0.1", username="admin")
        assert cfg.is_configured is False

    def test_defaults(self):
        cfg = SensorConfig()
        assert cfg.port == 80
        assert cfg.auto_stop_seconds == 10.0
        assert cfg.connect_timeout == 3
        assert cfg.read_timeout == 7


# ---------------------------------------------------------------------------
# SirenDevice 테스트
# ---------------------------------------------------------------------------

def _mock_resp(json_data=None, status=200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {"Execute": "OK"}
    resp.raise_for_status = MagicMock()
    return resp


class TestSirenDevice:
    def _make_device(self, auto_stop=0.0, **kwargs) -> SirenDevice:
        cfg = SensorConfig(
            host="192.168.0.1",
            username="admin",
            password="pw",
            auto_stop_seconds=auto_stop,
            **kwargs,
        )
        return SirenDevice(cfg)

    def test_trigger_returns_false_when_unconfigured(self):
        device = SirenDevice(SensorConfig())
        assert device.trigger() is False

    def test_stop_returns_false_when_unconfigured(self):
        device = SirenDevice(SensorConfig())
        assert device.stop() is False

    @patch("src.devices.siren.requests.post")
    def test_trigger_calls_warnbell_control(self, mock_post):
        mock_post.return_value = _mock_resp()
        device = self._make_device(auto_stop=0.0)

        result = device.trigger(event_type="head", camera_id="cam1")
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "/Warnbell/Control" in call_kwargs[0][0]
        body = call_kwargs[1]["json"]
        assert body["Run"] is True

    @patch("src.devices.siren.requests.post")
    def test_stop_calls_warnbell_control_run_false(self, mock_post):
        mock_post.return_value = _mock_resp()
        device = self._make_device()
        device._client = device._get_client()  # 클라이언트 초기화

        result = device.stop()
        assert result is True
        call_kwargs = mock_post.call_args
        body = call_kwargs[1]["json"]
        assert body["Run"] is False

    @patch("src.devices.siren.requests.post")
    def test_trigger_returns_false_on_exception(self, mock_post):
        mock_post.side_effect = OSError("연결 실패")
        device = self._make_device()
        result = device.trigger()
        assert result is False

    @patch("src.devices.siren.requests.post")
    def test_auto_stop_timer_scheduled(self, mock_post):
        mock_post.return_value = _mock_resp()
        device = self._make_device(auto_stop=60.0)

        device.trigger()
        assert device._stop_timer is not None
        assert device._stop_timer.is_alive()
        device._stop_timer.cancel()  # 타이머 정리

    @patch("src.devices.siren.requests.post")
    def test_no_auto_stop_timer_when_disabled(self, mock_post):
        mock_post.return_value = _mock_resp()
        device = self._make_device(auto_stop=0.0)

        device.trigger()
        assert device._stop_timer is None

    @patch("src.devices.siren.requests.post")
    def test_trigger_cancels_previous_timer(self, mock_post):
        mock_post.return_value = _mock_resp()
        device = self._make_device(auto_stop=60.0)

        device.trigger()
        first_timer = device._stop_timer

        device.trigger()
        # 두 번째 trigger → 새 타이머로 교체
        assert device._stop_timer is not first_timer
        device._stop_timer.cancel()
