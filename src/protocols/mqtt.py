"""MQTT 이벤트 발행 클라이언트

AI 추론 결과 이벤트를 MQTT 브로커로 발행하는 클라이언트.
재연결 백오프, 연결 상태 추적, 발행 통계를 제공한다.
"""

import json
import logging
import time
import uuid
from threading import Event, Lock
from typing import Dict, Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# 재연결 백오프 설정
_RECONNECT_MIN_DELAY = 1.0    # 최초 재시도 대기 시간 (초)
_RECONNECT_MAX_DELAY = 60.0   # 최대 재시도 대기 시간 (초)
_RECONNECT_MULTIPLIER = 2.0   # 대기 시간 배율


class MqttEventPublisher:
    """MQTT 브로커로 AI 이벤트를 발행하는 클라이언트.

    재연결 백오프와 연결 상태 추적을 내장하여
    브로커 장애 시에도 안정적으로 동작한다.
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic_prefix: str = "cctv/ai/events",
        client_id_prefix: str = "cctv-ai-engine",
        qos: int = 0,
        retain: bool = False,
        connect_timeout: float = 2.0,
    ):
        self.broker = broker
        self.port = int(port)
        self.topic_prefix = topic_prefix.rstrip("/")
        self.client_id_prefix = client_id_prefix
        self.qos = qos
        self.retain = retain
        self.connect_timeout = max(0.1, float(connect_timeout))

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._loop_running = False
        self._connect_waiter = Event()
        self._stats_lock = Lock()

        # 재연결 백오프 상태
        self._reconnect_delay = _RECONNECT_MIN_DELAY
        self._last_attempt_time: float = 0.0

        # 발행 통계
        self._publish_count = 0
        self._publish_fail_count = 0

    # ------------------------------------------------------------------
    # 공개 속성
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """현재 MQTT 브로커에 연결되어 있으면 True."""
        return self._connected

    # ------------------------------------------------------------------
    # 콜백
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT 연결 성공/실패 콜백."""
        self._connected = rc == 0
        self._connect_waiter.set()
        if self._connected:
            self._reconnect_delay = _RECONNECT_MIN_DELAY  # 성공 시 백오프 초기화
            logger.info("MQTT 연결 성공: %s:%s", self.broker, self.port)
        else:
            logger.error("MQTT 연결 실패 (rc=%s): %s:%s", rc, self.broker, self.port)

    def _on_disconnect(self, client, userdata, rc):
        """MQTT 연결 해제 콜백."""
        self._connected = False
        if rc != 0:
            logger.warning("MQTT 예상치 못한 연결 해제 (rc=%s) - 재연결 시도 예정", rc)

    # ------------------------------------------------------------------
    # 내부 연결 관리
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        """연결 확인 후 필요시 재연결을 시도한다.

        백오프를 적용해 재시도 주기를 조절하며,
        연결 성공 시 True, 실패 시 False를 반환한다.
        """
        if self._connected:
            return True

        # 백오프 대기 — 너무 자주 재시도하는 것을 방지
        now = time.monotonic()
        wait_remaining = self._reconnect_delay - (now - self._last_attempt_time)
        if wait_remaining > 0:
            logger.debug("MQTT 재연결 대기 중 (%.1f초)", wait_remaining)
            return False

        self._last_attempt_time = now

        if self._client is None:
            try:
                import paho.mqtt.client as _mqtt_mod
                # paho-mqtt 2.x: CallbackAPIVersion 필수, 1.x: 없어도 됨
                if hasattr(_mqtt_mod, "CallbackAPIVersion"):
                    self._client = mqtt.Client(
                        mqtt.CallbackAPIVersion.VERSION1,
                        client_id=f"{self.client_id_prefix}-{uuid.uuid4().hex[:8]}",
                        clean_session=True,
                    )
                else:
                    self._client = mqtt.Client(
                        client_id=f"{self.client_id_prefix}-{uuid.uuid4().hex[:8]}",
                        clean_session=True,
                    )
            except Exception:
                # 최후 폴백: 키워드 없이 시도
                self._client = mqtt.Client(
                    client_id=f"{self.client_id_prefix}-{uuid.uuid4().hex[:8]}",
                )
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

        try:
            self._connect_waiter.clear()
            self._client.connect(self.broker, self.port, keepalive=60)
            if not self._loop_running:
                self._client.loop_start()
                self._loop_running = True

            connected = self._connect_waiter.wait(timeout=self.connect_timeout)
            if connected and self._connected:
                return True

            # 실패 → 백오프 증가
            self._reconnect_delay = min(
                self._reconnect_delay * _RECONNECT_MULTIPLIER,
                _RECONNECT_MAX_DELAY,
            )
            return False

        except Exception as error:
            logger.error("MQTT 연결 오류: %s", error)
            self._reconnect_delay = min(
                self._reconnect_delay * _RECONNECT_MULTIPLIER,
                _RECONNECT_MAX_DELAY,
            )
            return False

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def publish_event(self, event_data: Dict) -> bool:
        """AI 이벤트 데이터를 MQTT 토픽으로 발행한다.

        토픽 형식: ``{topic_prefix}/{camera_id}/{event_type}``

        매개변수:
            event_data: 발행할 이벤트 딕셔너리.

        반환값:
            발행 성공 시 True, 실패 시 False.
        """
        if not self._ensure_connected():
            with self._stats_lock:
                self._publish_fail_count += 1
            return False

        camera_id = event_data.get("camera_id", "unknown")
        event_type = event_data.get("type", "unknown")
        topic = f"{self.topic_prefix}/{camera_id}/{event_type}"

        try:
            payload = json.dumps(event_data, ensure_ascii=False)
            result = self._client.publish(topic, payload, qos=self.qos, retain=self.retain)
            if result.rc == 0:
                with self._stats_lock:
                    self._publish_count += 1
                return True
            logger.error("MQTT 발행 실패 (rc=%s): %s", result.rc, topic)
        except Exception as error:
            logger.error("MQTT 발행 오류: %s", error, exc_info=True)

        with self._stats_lock:
            self._publish_fail_count += 1
        return False

    def get_stats(self) -> Dict:
        """발행 통계를 반환한다.

        반환값:
            publish_count, publish_fail_count, is_connected 를 담은 딕셔너리.
        """
        with self._stats_lock:
            return {
                "is_connected": self._connected,
                "publish_count": self._publish_count,
                "publish_fail_count": self._publish_fail_count,
                "broker": f"{self.broker}:{self.port}",
            }

    def disconnect(self):
        """MQTT 연결을 종료하고 리소스를 정리한다."""
        if not self._client:
            return
        try:
            if self._loop_running:
                self._client.loop_stop()
                self._loop_running = False
            self._client.disconnect()
            self._connected = False
        except Exception as error:
            logger.error("MQTT 연결 종료 오류: %s", error)
