"""
mqtt/base_publisher.py
======================
재연결 백오프를 내장한 MQTT 발행 기반 클래스입니다.

src/protocols/mqtt.py 의 MqttEventPublisher 와 동일한 연결 관리 패턴을
parser-python 패키지 전용으로 독립적으로 구현합니다.

서브클래스는 publish() 메서드만 오버라이드하면 됩니다.
"""

import logging
import time
import uuid
from threading import Event, Lock
from typing import Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

_PAHO_V2 = hasattr(mqtt, "CallbackAPIVersion")

_RECONNECT_MIN_DELAY = 1.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_MULTIPLIER = 2.0


class BaseMqttPublisher:
    """
    재연결 백오프를 내장한 MQTT 발행 기반 클래스.
    서브클래스는 publish() 를 구현하고, _ensure_connected() 를 호출합니다.
    """

    def __init__(
        self,
        broker: str,
        port: int = 1883,
        client_id_prefix: str = "mqtt-publisher",
        qos: int = 0,
        connect_timeout: float = 3.0,
    ):
        self.broker = broker
        self.port = int(port)
        self.client_id_prefix = client_id_prefix
        self.qos = qos
        self.connect_timeout = max(0.1, float(connect_timeout))

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._loop_running = False
        self._connect_waiter = Event()
        self._lock = Lock()

        self._reconnect_delay = _RECONNECT_MIN_DELAY
        self._last_attempt_time: float = 0.0

    # ------------------------------------------------------------------ #
    # 콜백                                                                 #
    # ------------------------------------------------------------------ #

    def _on_connect(self, client, userdata, flags, rc, *args):
        self._connected = (rc == 0)
        self._connect_waiter.set()
        if self._connected:
            self._reconnect_delay = _RECONNECT_MIN_DELAY
            logger.info("%s 연결 성공: %s:%s", self.__class__.__name__, self.broker, self.port)
        else:
            logger.error("%s 연결 실패 rc=%s: %s:%s", self.__class__.__name__, rc, self.broker, self.port)

    def _on_disconnect(self, client, userdata, rc, *args):
        self._connected = False
        if rc != 0:
            logger.warning("%s 연결 끊김 rc=%s — 재연결 대기 중", self.__class__.__name__, rc)

    # ------------------------------------------------------------------ #
    # 내부 연결 관리                                                         #
    # ------------------------------------------------------------------ #

    def _ensure_connected(self) -> bool:
        """연결 확인 후 필요 시 재연결. 백오프 적용."""
        if self._connected:
            return True

        now = time.monotonic()
        wait = self._reconnect_delay - (now - self._last_attempt_time)
        if wait > 0:
            return False

        self._last_attempt_time = now

        if self._client is None:
            client_id = f"{self.client_id_prefix}-{uuid.uuid4().hex[:8]}"
            if _PAHO_V2:
                self._client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2,
                    client_id=client_id,
                    clean_session=True,
                )
            else:
                self._client = mqtt.Client(client_id=client_id, clean_session=True)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

        try:
            self._connect_waiter.clear()
            self._client.connect(self.broker, self.port, keepalive=60)
            if not self._loop_running:
                self._client.loop_start()
                self._loop_running = True

            if self._connect_waiter.wait(timeout=self.connect_timeout) and self._connected:
                return True
        except Exception as e:
            logger.error("%s 연결 오류: %s", self.__class__.__name__, e)

        self._reconnect_delay = min(
            self._reconnect_delay * _RECONNECT_MULTIPLIER,
            _RECONNECT_MAX_DELAY,
        )
        return False

    # ------------------------------------------------------------------ #
    # 공개 API                                                              #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self):
        if not self._client:
            return
        try:
            if self._loop_running:
                self._client.loop_stop()
                self._loop_running = False
            self._client.disconnect()
            self._connected = False
        except Exception as e:
            logger.error("%s 종료 오류: %s", self.__class__.__name__, e)
