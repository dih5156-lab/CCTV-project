"""
test_mqtt.py — MqttEventPublisher 단위 테스트

커버리지 대상:
  - 초기화 속성
  - publish_event: 미연결 / 연결됨 / 발행 실패 / 예외 처리
  - get_stats 구조
  - 재연결 백오프 (_ensure_connected, _reconnect_delay)
"""

import json

import pytest
from unittest.mock import MagicMock, patch

from src.protocols.mqtt import (
    MqttEventPublisher,
    _RECONNECT_MAX_DELAY,
    _RECONNECT_MIN_DELAY,
)


# ---------------------------------------------------------------------------
# 후처리 헬퍼: connected publisher fixture
# ---------------------------------------------------------------------------


def _connected_publisher(**kwargs) -> MqttEventPublisher:
    """실제 브로커 없이 '이미 연결된' 상태의 publisher를 반환한다."""
    pub = MqttEventPublisher(**kwargs)
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.rc = 0
    mock_client.publish.return_value = mock_result
    pub._client = mock_client
    pub._connected = True
    return pub


# ---------------------------------------------------------------------------
# 초기화 / 속성
# ---------------------------------------------------------------------------


class TestMqttEventPublisherInit:
    def test_default_broker(self):
        pub = MqttEventPublisher()
        assert pub.broker == "localhost"

    def test_default_port(self):
        pub = MqttEventPublisher()
        assert pub.port == 1883

    def test_default_topic_prefix(self):
        pub = MqttEventPublisher()
        assert pub.topic_prefix == "cctv/ai/events"

    def test_is_connected_initially_false(self):
        pub = MqttEventPublisher()
        assert pub.is_connected is False

    def test_custom_broker_and_port(self):
        pub = MqttEventPublisher(broker="192.168.1.1", port=8883)
        assert pub.broker == "192.168.1.1"
        assert pub.port == 8883

    def test_connect_timeout_minimum_enforced(self):
        """0이나 음수는 최소값으로 클램핑된다."""
        pub = MqttEventPublisher(connect_timeout=0.0)
        assert pub.connect_timeout >= 0.1

    def test_topic_prefix_trailing_slash_stripped(self):
        pub = MqttEventPublisher(topic_prefix="cctv/ai/events/")
        assert not pub.topic_prefix.endswith("/")

    def test_initial_stats_all_zero(self):
        pub = MqttEventPublisher()
        stats = pub.get_stats()
        assert stats["publish_count"] == 0
        assert stats["publish_fail_count"] == 0
        assert stats["is_connected"] is False


# ---------------------------------------------------------------------------
# publish_event — 미연결 상태
# ---------------------------------------------------------------------------


class TestPublishWhenDisconnected:
    @pytest.fixture
    def pub(self) -> MqttEventPublisher:
        """_ensure_connected 를 False 반환으로 고정한 publisher."""
        p = MqttEventPublisher()
        p._ensure_connected = MagicMock(return_value=False)
        return p

    def test_returns_false(self, pub):
        result = pub.publish_event({"camera_id": "cam1", "type": "helmet"})
        assert result is False

    def test_fail_count_increments(self, pub):
        pub.publish_event({"camera_id": "cam1", "type": "helmet"})
        assert pub.get_stats()["publish_fail_count"] == 1

    def test_multiple_failures_counted(self, pub):
        pub.publish_event({"camera_id": "cam1", "type": "helmet"})
        pub.publish_event({"camera_id": "cam1", "type": "head"})
        assert pub.get_stats()["publish_fail_count"] == 2

    def test_publish_count_stays_zero(self, pub):
        pub.publish_event({"camera_id": "cam1", "type": "helmet"})
        assert pub.get_stats()["publish_count"] == 0


# ---------------------------------------------------------------------------
# publish_event — 연결된 상태 (mock 클라이언트)
# ---------------------------------------------------------------------------


class TestPublishWhenConnected:
    @pytest.fixture
    def pub(self) -> MqttEventPublisher:
        return _connected_publisher(topic_prefix="cctv/ai/events", qos=0)

    def test_returns_true(self, pub):
        result = pub.publish_event({"camera_id": "cam1", "type": "helmet"})
        assert result is True

    def test_publish_count_increments(self, pub):
        pub.publish_event({"camera_id": "cam1", "type": "helmet"})
        assert pub.get_stats()["publish_count"] == 1

    def test_topic_format(self, pub):
        pub.publish_event({"camera_id": "cam2", "type": "fall_detected"})
        topic = pub._client.publish.call_args[0][0]
        assert topic == "cctv/ai/events/cam2/fall_detected"

    def test_payload_is_valid_json(self, pub):
        pub.publish_event({"camera_id": "cam1", "type": "helmet", "confidence": 0.9})
        payload = pub._client.publish.call_args[0][1]
        parsed = json.loads(payload)
        assert parsed["camera_id"] == "cam1"
        assert parsed["confidence"] == pytest.approx(0.9)

    def test_failure_rc_returns_false(self, pub):
        bad_result = MagicMock()
        bad_result.rc = 1
        pub._client.publish.return_value = bad_result
        assert pub.publish_event({"camera_id": "cam1", "type": "helmet"}) is False
        assert pub.get_stats()["publish_fail_count"] == 1

    def test_exception_during_publish_returns_false(self, pub):
        pub._client.publish.side_effect = RuntimeError("connection lost")
        assert pub.publish_event({"camera_id": "cam1", "type": "helmet"}) is False
        assert pub.get_stats()["publish_fail_count"] == 1

    def test_missing_camera_id_uses_unknown_in_topic(self, pub):
        pub.publish_event({"type": "helmet"})
        topic = pub._client.publish.call_args[0][0]
        assert "unknown" in topic

    def test_missing_type_uses_unknown_in_topic(self, pub):
        pub.publish_event({"camera_id": "cam1"})
        topic = pub._client.publish.call_args[0][0]
        assert "unknown" in topic

    def test_qos_and_retain_forwarded(self):
        pub = _connected_publisher(qos=1, retain=True)
        pub.publish_event({"camera_id": "c", "type": "t"})
        _, kwargs = pub._client.publish.call_args
        assert kwargs.get("qos") == 1
        assert kwargs.get("retain") is True

    def test_multiple_publishes_counted(self, pub):
        for _ in range(5):
            pub.publish_event({"camera_id": "c", "type": "t"})
        assert pub.get_stats()["publish_count"] == 5


# ---------------------------------------------------------------------------
# 재연결 백오프
# ---------------------------------------------------------------------------


class TestReconnectBackoff:
    """paho.mqtt.client.Client.connect 을 mock 처리하여 DNS 조회 없이 테스트."""

    @staticmethod
    def _patched_pub(**kwargs) -> MqttEventPublisher:
        """connect() 가 즉시 OSError 를 발생시키는 publisher 반환."""
        with patch("src.protocols.mqtt.mqtt") as mock_mqtt_mod:
            mock_cli = MagicMock()
            mock_cli.connect.side_effect = OSError("Connection refused")
            mock_mqtt_mod.Client.return_value = mock_cli
            pub = MqttEventPublisher(broker="localhost", connect_timeout=0.001, **kwargs)
            pub._last_attempt_time = 0  # 즉시 재시도 허용
            # _ensure_connected 에서 mqtt.Client 를 생성할 때도 같은 mock 이 필요
            # → _client 를 미리 주입하여 재생성을 막는다
            pub._client = mock_cli
        return pub

    def test_initial_delay_is_min(self):
        pub = MqttEventPublisher()
        assert pub._reconnect_delay == _RECONNECT_MIN_DELAY

    def test_backoff_increases_on_failed_connect(self):
        pub = self._patched_pub()
        before = pub._reconnect_delay
        pub._ensure_connected()
        assert pub._reconnect_delay > before

    def test_backoff_does_not_exceed_max(self):
        pub = self._patched_pub()
        pub._reconnect_delay = _RECONNECT_MAX_DELAY - 0.01
        pub._ensure_connected()
        assert pub._reconnect_delay <= _RECONNECT_MAX_DELAY

    def test_ensure_connected_returns_false_on_error(self):
        pub = self._patched_pub()
        assert pub._ensure_connected() is False

    def test_ensure_connected_returns_true_when_already_connected(self):
        pub = MqttEventPublisher()
        pub._connected = True
        assert pub._ensure_connected() is True

    def test_cooldown_skips_reconnect(self):
        """백오프 대기 중에는 재시도하지 않고 False 를 즉시 반환."""
        import time
        pub = MqttEventPublisher()
        pub._last_attempt_time = time.monotonic()
        pub._reconnect_delay = 9999.0  # 매우 긴 대기
        result = pub._ensure_connected()
        assert result is False


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_required_keys_present(self):
        pub = MqttEventPublisher()
        stats = pub.get_stats()
        assert {"is_connected", "publish_count", "publish_fail_count", "broker"} <= set(stats.keys())

    def test_broker_format(self):
        pub = MqttEventPublisher(broker="10.0.0.1", port=9999)
        assert pub.get_stats()["broker"] == "10.0.0.1:9999"

    def test_stats_reflect_publishes_and_failures(self):
        pub = _connected_publisher()
        pub.publish_event({"camera_id": "c", "type": "t"})  # 성공

        bad_result = MagicMock()
        bad_result.rc = 1
        pub._client.publish.return_value = bad_result
        pub.publish_event({"camera_id": "c", "type": "t"})  # 실패

        stats = pub.get_stats()
        assert stats["publish_count"] == 1
        assert stats["publish_fail_count"] == 1
