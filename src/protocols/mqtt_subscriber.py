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
_RECONNECT_MAX_DELAY = 30.0  # subscriber: 최대 30초
_RECONNECT_MULTIPLIER = RECONNECT_MULTIPLIER


class MqttTopicSubscriber:
    """지정한 MQTT 토픽 집합을 구독하는 범용 클라이언트.

    durable 모드의 handler는 원본 commit 후 반환하고 실패 시 예외를 전달해야 한다.
    소유 루프가 reconnect_if_needed를 호출해 저장 실패로 중단한 구독을 재개한다.
    """

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
        durable: bool = False,
        transport: str = "tcp",
    ) -> None:
        if not topics:
            raise ValueError("구독 토픽은 최소 1개 이상이어야 합니다")
        if durable and (not client_id or not client_id.strip()):
            raise ValueError("Durable MQTT requires an explicit stable client_id")

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
        self.durable = durable
        self.transport = transport
        self.session_present = False
        self._pending_subscriptions: set[int] = set()
        self._restart_requested = False
        self._qos0_warned = False

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
            durable=self.durable,
            transport=self.transport,
        )
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        client.on_subscribe = self._on_subscribe
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
        self.session_present = bool(
            flags.get("session present", False)
            if isinstance(flags, dict)
            else getattr(flags, "session_present", False)
        )
        self._pending_subscriptions.clear()
        if self.durable:
            self._connected = False
        logger.info("External MQTT 연결 성공: %s:%s", self.broker, self.port)
        for topic in self.topics:
            subscription = client.subscribe(topic, qos=1 if self.durable else 0)
            if self.durable:
                result, mid = subscription
                if result != mqtt.MQTT_ERR_SUCCESS:
                    self._request_reconnect(client)
                    return
                self._pending_subscriptions.add(mid)
            logger.info("External MQTT 구독: %s", topic)

    def _on_subscribe(
        self, client, userdata, mid, granted_qos, properties=None
    ) -> None:
        if not self.durable or self._restart_requested:
            return
        if mid not in self._pending_subscriptions:
            return
        if not granted_qos or any(
            getattr(qos, "value", qos) != 1 for qos in granted_qos
        ):
            logger.error("Durable MQTT subscription did not grant QoS 1")
            self._request_reconnect(client)
            return
        self._pending_subscriptions.remove(mid)
        self._connected = not self._pending_subscriptions

    def _on_disconnect(self, client, userdata, rc, *args) -> None:
        self._connected = False
        if len(args) == 2:  # Paho callback API v2: flags, reason_code, properties
            rc = args[0]
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
            if self.durable:
                if msg.qos == 1:
                    if client.ack(msg.mid, msg.qos) != mqtt.MQTT_ERR_SUCCESS:
                        self._request_reconnect(client)
                elif msg.qos == 0 and not self._qos0_warned:
                    logger.warning(
                        "External MQTT QoS 0 input: broker redelivery is not guaranteed"
                    )
                    self._qos0_warned = True
        except Exception as exc:
            logger.error("External MQTT 메시지 처리 실패: %s", exc, exc_info=True)
            if self.durable:
                self._request_reconnect(client)

    def _request_reconnect(self, client) -> None:
        self._connected = False
        self._restart_requested = True
        self._last_attempt_time = time.monotonic()
        # Do not join the network thread from its own callback.
        client.disconnect()

    def reconnect_if_needed(self) -> bool:
        """Resume a rejected receipt using the same broker session, from the owner loop."""
        if not self._restart_requested:
            return False
        if time.monotonic() - self._last_attempt_time < self._reconnect_delay:
            return False
        if self._client is not None and self._loop_running:
            self._client.loop_stop()
        self._client = None
        self._loop_running = False
        self._restart_requested = False
        connected = self.connect()
        if not connected:
            self._restart_requested = True
        return connected

    def connect(self) -> bool:
        """브로커 연결을 시도하고 loop를 시작한다."""
        if self._restart_requested:
            return self.reconnect_if_needed()
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
        self._restart_requested = False
        if not self._client:
            return
        try:
            if self._loop_running:
                self._client.loop_stop()
                self._loop_running = False
            self._client.disconnect()
        finally:
            self._connected = False
