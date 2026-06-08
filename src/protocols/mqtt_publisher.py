"""MQTT 이벤트 발행 클라이언트

AI 추론 결과 이벤트를 MQTT 브로커로 발행하는 클라이언트.
재연결 백오프, 연결 상태 추적, 발행 통계를 제공한다.
"""

import json
import logging
import os
import time
from threading import Event, Lock, Thread
from typing import Dict, Optional

import paho.mqtt.client as mqtt

from ..canonical_event import (
    canonicalize_event_payload,
    get_payload_camera_id,
    get_payload_event_type,
)
from ._mqtt_factory import RECONNECT_MIN_DELAY, RECONNECT_MULTIPLIER, create_mqtt_client
from .mqtt_outbox import MqttEventOutbox

logger = logging.getLogger(__name__)

# 재연결 백오프 설정
_RECONNECT_MIN_DELAY = RECONNECT_MIN_DELAY
_RECONNECT_MAX_DELAY = 60.0           # publisher: 최대 60초
_RECONNECT_MULTIPLIER = RECONNECT_MULTIPLIER


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
        outbox_db_path: Optional[str] = None,
        outbox_retry_interval: Optional[float] = None,
        outbox_replay_batch_size: int = 100,
        outbox_max_retry: int = 1000,
    ):
        self.broker = broker
        self.port = int(port)
        self.topic_prefix = topic_prefix.rstrip("/")
        self.client_id_prefix = client_id_prefix
        self.qos = qos
        self.retain = retain
        self.connect_timeout = max(0.1, float(connect_timeout))
        self.outbox_db_path = outbox_db_path or os.environ.get("MQTT_EVENT_OUTBOX_DB")
        self.outbox_retry_interval = max(
            1.0,
            float(
                outbox_retry_interval
                if outbox_retry_interval is not None
                else os.environ.get("MQTT_EVENT_OUTBOX_RETRY_INTERVAL", "5")
            ),
        )
        self.outbox_replay_batch_size = max(1, int(outbox_replay_batch_size))
        self.outbox_max_retry = max(1, int(outbox_max_retry))

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
        self._outbox_replay_count = 0

        self._outbox: Optional[MqttEventOutbox] = None
        self._outbox_stop = Event()
        self._outbox_thread: Optional[Thread] = None
        if self.outbox_db_path:
            self._outbox = MqttEventOutbox(
                self.outbox_db_path,
                destination_name=self.topic_prefix,
            )
            self._start_outbox_replay_worker()

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

    def _on_connect(self, client, userdata, flags, rc, *args):
        """MQTT 연결 성공/실패 콜백."""
        self._connected = rc == 0
        self._connect_waiter.set()
        if self._connected:
            self._reconnect_delay = _RECONNECT_MIN_DELAY  # 성공 시 백오프 초기화
            logger.info("MQTT 연결 성공: %s:%s", self.broker, self.port)
        else:
            logger.error("MQTT 연결 실패 (rc=%s): %s:%s", rc, self.broker, self.port)

    def _on_disconnect(self, client, userdata, rc, *args):
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

        loop_start() 이후에는 paho가 자동 재연결을 관리하므로
        수동 connect()를 다시 호출하지 않는다 (중복 CONNECT 방지).
        """
        if self._connected:
            return True

        # loop_start()가 이미 동작 중이면 paho 자동 재연결에 위임
        if self._loop_running:
            return False

        # 백오프 대기 — 너무 자주 재시도하는 것을 방지
        now = time.monotonic()
        wait_remaining = self._reconnect_delay - (now - self._last_attempt_time)
        if wait_remaining > 0:
            logger.debug("MQTT 재연결 대기 중 (%.1f초)", wait_remaining)
            return False

        self._last_attempt_time = now

        if self._client is None:
            self._client = create_mqtt_client(self.client_id_prefix)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

        try:
            self._connect_waiter.clear()
            self._client.connect(self.broker, self.port, keepalive=60)
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
        event_data = canonicalize_event_payload(event_data)
        topic = self._event_topic(event_data)
        outbox_row_id = self._store_outbox_pending(topic, event_data)

        if not self._ensure_connected():
            self._increment_publish_fail_count()
            return False

        ok, error_message = self._publish_serialized(topic, event_data)
        if ok:
            if outbox_row_id:
                self._outbox.mark_sent(outbox_row_id)  # type: ignore[union-attr]
            self._increment_publish_count()
            return True

        if outbox_row_id:
            self._outbox.mark_retry_failed(outbox_row_id, error_message)  # type: ignore[union-attr]
        self._increment_publish_fail_count()
        return False

    def _publish_serialized(self, topic: str, event_data: Dict) -> tuple[bool, str]:
        try:
            payload = json.dumps(event_data, ensure_ascii=False)
            result = self._client.publish(topic, payload, qos=self.qos, retain=self.retain)
            if result.rc == 0:
                return True, ""
            logger.error("MQTT 발행 실패 (rc=%s): %s", result.rc, topic)
            return False, f"mqtt publish rc={result.rc}"
        except Exception as error:
            logger.error("MQTT 발행 오류: %s", error, exc_info=True)
            return False, str(error)

    def _store_outbox_pending(self, topic: str, event_data: Dict) -> int:
        if self._outbox is None:
            return 0
        try:
            return self._outbox.save_pending(topic, event_data)
        except Exception as error:
            logger.error("MQTT outbox 저장 실패: %s", error, exc_info=True)
            return 0

    def _start_outbox_replay_worker(self) -> None:
        if self._outbox_thread and self._outbox_thread.is_alive():
            return
        self._outbox_stop.clear()
        self._outbox_thread = Thread(
            target=self._outbox_replay_loop,
            daemon=True,
            name="MqttEventOutboxReplay",
        )
        self._outbox_thread.start()

    def _outbox_replay_loop(self) -> None:
        logger.info("MQTT outbox 재전송 워커 시작: %s", self.outbox_db_path)
        while not self._outbox_stop.wait(self.outbox_retry_interval):
            self.replay_pending_once()

    def replay_pending_once(self) -> int:
        """Replay one batch of pending outbox rows and return sent count."""
        if self._outbox is None:
            return 0
        if not self._ensure_connected():
            return 0

        sent_count = 0
        rows = self._outbox.get_pending(
            limit=self.outbox_replay_batch_size,
            max_retry=self.outbox_max_retry,
        )
        for row in rows:
            ok, error_message = self._publish_serialized(row["topic"], row["payload"])
            if ok:
                self._outbox.mark_sent(row["id"])
                sent_count += 1
                continue
            self._outbox.mark_retry_failed(row["id"], error_message)
        if sent_count:
            with self._stats_lock:
                self._outbox_replay_count += sent_count
            logger.info("MQTT outbox 재전송 성공: %d건", sent_count)
        return sent_count

    def _event_topic(self, event_data: Dict) -> str:
        camera_id = get_payload_camera_id(event_data)
        event_type = get_payload_event_type(event_data)
        return f"{self.topic_prefix}/{camera_id}/{event_type}"

    def _increment_publish_count(self) -> None:
        with self._stats_lock:
            self._publish_count += 1

    def _increment_publish_fail_count(self) -> None:
        with self._stats_lock:
            self._publish_fail_count += 1

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
                "outbox_enabled": self._outbox is not None,
                "outbox_pending_count": self._outbox.pending_count() if self._outbox else 0,
                "outbox_replay_count": self._outbox_replay_count,
                "broker": f"{self.broker}:{self.port}",
            }

    def disconnect(self):
        """MQTT 연결을 종료하고 리소스를 정리한다."""
        self._outbox_stop.set()
        if self._outbox_thread and self._outbox_thread.is_alive():
            self._outbox_thread.join(timeout=3)
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
