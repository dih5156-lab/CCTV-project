"""MQTT 토픽 구독 클라이언트."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Optional, Sequence

import paho.mqtt.client as mqtt

from ._mqtt_factory import RECONNECT_MIN_DELAY, RECONNECT_MULTIPLIER, create_mqtt_client

logger = logging.getLogger(__name__)

_RECONNECT_MIN_DELAY = RECONNECT_MIN_DELAY
_RECONNECT_MAX_DELAY = 30.0           # subscriber: 최대 30초
_RECONNECT_MULTIPLIER = RECONNECT_MULTIPLIER


class MqttTopicSubscriber:
    """지정한 MQTT 토픽 집합을 구독하는 범용 클라이언트."""

    def __init__(
        self,
        *,
        broker: str = "localhost",
        port: int = 1883,
        topics: Sequence[str] = ("#",),
        message_handler: Callable[[str, bytes], None],
        client_id_prefix: str = "cctv-subscriber",
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        connect_timeout: float = 2.0,
    ) -> None:
        if not topics:
            raise ValueError("구독 토픽은 최소 1개 이상이어야 합니다")

        self.broker = broker
        self.port = int(port)
        self.topics = tuple(topic.strip() for topic in topics if topic.strip())
        if not self.topics:
            raise ValueError("유효한 구독 토픽이 없습니다")

        self.message_handler = message_handler
        self.client_id_prefix = client_id_prefix
        self.client_id = client_id
        self.username = username
        self.password = password
        self.connect_timeout = max(0.1, float(connect_timeout))

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._loop_running = False
        self._last_attempt_time = 0.0
        self._reconnect_delay = _RECONNECT_MIN_DELAY

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _build_client(self) -> mqtt.Client:
        client = create_mqtt_client(
            client_id_prefix=self.client_id_prefix,
            client_id=self.client_id,
            username=self.username,
            password=self.password,
        )
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        return client

    def _on_connect(self, client, userdata, flags, rc, *args) -> None:
        self._connected = rc == 0
        if not self._connected:
            logger.error(
                "External MQTT 연결 실패 (rc=%s): %s:%s client_id=%s",
                rc,
                self.broker,
                self.port,
                self.client_id or "<auto>",
            )
            if str(rc).lower() == "not authorized":
                logger.error(
                    "인증 또는 ACL 문제 가능성이 큽니다. username/password/client_id/topic 권한을 확인하세요. "
                    "현재 username=%s topic_count=%d",
                    self.username or "<none>",
                    len(self.topics),
                )
            return

        self._reconnect_delay = _RECONNECT_MIN_DELAY
        logger.info("External MQTT 연결 성공: %s:%s", self.broker, self.port)
        for topic in self.topics:
            client.subscribe(topic, qos=0)
            logger.info("External MQTT 구독: %s", topic)

    def _on_disconnect(self, client, userdata, rc, *args) -> None:
        self._connected = False
        if rc != 0:
            logger.warning("External MQTT 예상치 못한 연결 해제 (rc=%s)", rc)

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        try:
            preview = bytes(msg.payload[:120]).decode("utf-8", errors="replace")
            logger.debug(
                "External MQTT raw message 도착: topic=%s payload_len=%d preview=%r",
                msg.topic,
                len(msg.payload),
                preview,
            )
            self.message_handler(msg.topic, bytes(msg.payload))
        except Exception as exc:
            logger.error("External MQTT 메시지 처리 실패: %s", exc, exc_info=True)

    def connect(self) -> bool:
        """브로커 연결을 시도하고 loop를 시작한다."""
        if self._connected:
            return True

        now = time.monotonic()
        wait_remaining = self._reconnect_delay - (now - self._last_attempt_time)
        if wait_remaining > 0:
            logger.debug("External MQTT 재연결 대기 중 (%.1f초)", wait_remaining)
            return False

        self._last_attempt_time = now

        if self._client is None:
            self._client = self._build_client()

        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            if not self._loop_running:
                self._client.loop_start()
                self._loop_running = True

            deadline = time.monotonic() + self.connect_timeout
            while time.monotonic() < deadline:
                if self._connected:
                    return True
                time.sleep(0.05)

            self._reconnect_delay = min(
                self._reconnect_delay * _RECONNECT_MULTIPLIER,
                _RECONNECT_MAX_DELAY,
            )
            return False
        except Exception as exc:
            logger.error("External MQTT 연결 오류: %s", exc)
            self._reconnect_delay = min(
                self._reconnect_delay * _RECONNECT_MULTIPLIER,
                _RECONNECT_MAX_DELAY,
            )
            return False

    def disconnect(self) -> None:
        if not self._client:
            return
        try:
            if self._loop_running:
                self._client.loop_stop()
                self._loop_running = False
            self._client.disconnect()
        finally:
            self._connected = False
