"""status_reporter.py - 카메라 상태·탐지 스냅샷 MQTT 발행 서비스.

VideoProcessor 에서 카메라 연결 상태와 최신 탐지 결과를 수집해
MQTT 브로커로 주기적으로 발행한다. 대시보드 등 실시간 관제 화면에서
구독할 수 있도록 토픽을 분리한다.

MQTT 토픽 구조::

    {topic_prefix}/cameras/{camera_id}   카메라별 연결 상태 + 현재 탐지 객체
    {topic_prefix}/stats                 시스템 전체 통계 (FPS·오류 수·가동 시간 등)

기본 topic_prefix: ``cctv/status``

카메라 토픽 페이로드 예시::

    {
        "camera_id": "camera-1",
        "timestamp": 1741600001.234,
        "status": "online",           // online | reconnecting | offline
        "connected": true,
        "reconnect_attempts": 0,
        "last_frame_age_sec": 0.3,
        "detections": [
            {
                "type": "person",
                "bbox": {"x": 120, "y": 80, "width": 60, "height": 140},
                "confidence": 0.87,
                "object_id": 1,
                "timestamp": 1741600000.8
            }
        ]
    }

통계 토픽 페이로드 예시::

    {
        "timestamp": 1741600001.234,
        "camera_count": 3,
        "fps": 12.4,
        "frames_processed": 1200,
        "frames_dropped": 5,
        "events_detected": 47,
        "events_sent": 45,
        "inference_errors": 0,
        "camera_errors": 2,
        "avg_inference_ms": 38.2,
        "uptime_seconds": 96.5
    }

사용 예::

    from src.services.status_reporter import StatusReporter

    reporter = StatusReporter(
        processor=processor,
        broker="localhost",
        port=1883,
        topic_prefix="cctv/status",
        interval=3.0,
    )
    reporter.start()
    ...
    reporter.stop()
"""

import json
import logging
import time
import uuid
from threading import Event, Thread
from typing import TYPE_CHECKING, Dict, Optional

import paho.mqtt.client as mqtt

if TYPE_CHECKING:
    from ..core import VideoProcessor

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL     = 3.0   # 발행 주기 (초)
_DEFAULT_TOPIC_PREFIX = "cctv/status"
_CONNECT_TIMEOUT      = 5.0   # 초기 연결 대기 (초)
_RECONNECT_MIN_DELAY  = 2.0
_RECONNECT_MAX_DELAY  = 30.0
_RECONNECT_MULTIPLIER = 2.0


class StatusReporter:
    """카메라 상태와 탐지 스냅샷을 MQTT 브로커로 주기적으로 발행한다.

    기존 탐지 알림(``cctv/ai/events/…``)과 토픽이 분리되어 있어
    대시보드에서 독립적으로 구독할 수 있다.

    Args:
        processor:    VideoProcessor 인스턴스.
        broker:       MQTT 브로커 호스트. 기본값 ``"localhost"``.
        port:         MQTT 브로커 포트. 기본값 ``1883``.
        topic_prefix: 토픽 접두사. 기본값 ``"cctv/status"``.
        interval:     발행 주기(초). 기본값 ``3.0``.
        qos:          MQTT QoS (0·1·2). 기본값 ``0``.
    """

    def __init__(
        self,
        processor: "VideoProcessor",
        broker: str = "localhost",
        port: int = 1883,
        topic_prefix: str = _DEFAULT_TOPIC_PREFIX,
        interval: float = _DEFAULT_INTERVAL,
        qos: int = 0,
    ) -> None:
        self._processor = processor
        self._broker = broker
        self._port = int(port)
        self._topic_prefix = topic_prefix.rstrip("/")
        self._interval = interval
        self._qos = qos

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._connect_waiter = Event()
        self._reconnect_delay = _RECONNECT_MIN_DELAY
        self._last_attempt_time: float = 0.0

        self._running = False
        self._thread: Optional[Thread] = None

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """백그라운드 발행 스레드를 시작한다."""
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._loop, daemon=True, name="StatusReporter")
        self._thread.start()
        logger.info(
            "[StatusReporter] 시작됨 — broker=%s:%d, prefix=%s, interval=%.1fs",
            self._broker, self._port, self._topic_prefix, self._interval,
        )

    def stop(self) -> None:
        """발행 스레드를 중지하고 MQTT 연결을 해제한다."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval + 2)
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
        logger.info("[StatusReporter] 중지됨")

    def build_camera_payload(self, camera_id: str) -> dict:
        """카메라 한 대의 상태 페이로드를 반환한다 (테스트·디버그 용도)."""
        return self._build_camera_payload(
            camera_id,
            self._processor.get_camera_status(),
            self._processor.get_detection_snapshot(),
        )

    def build_stats_payload(self) -> dict:
        """시스템 통계 페이로드를 반환한다 (테스트·디버그 용도)."""
        return self._build_stats_payload()

    # ------------------------------------------------------------------
    # 내부 — 페이로드 구성
    # ------------------------------------------------------------------

    def _build_camera_payload(
        self,
        camera_id: str,
        camera_status: Dict[str, dict],
        snapshots: Dict[str, dict],
    ) -> dict:
        now = time.time()
        cam_info = camera_status.get(camera_id, {
            "status": "unknown",
            "connected": False,
            "reconnect_attempts": 0,
            "last_frame_time": None,
            "last_frame_age_sec": None,
        })
        snap = snapshots.get(camera_id)
        return {
            "camera_id": camera_id,
            "timestamp": now,
            "status": cam_info.get("status", "unknown"),
            "connected": cam_info.get("connected", False),
            "reconnect_attempts": cam_info.get("reconnect_attempts", 0),
            "last_frame_age_sec": cam_info.get("last_frame_age_sec"),
            "detections": snap["detections"] if snap else [],
        }

    def _build_stats_payload(self) -> dict:
        stats = self._processor.get_stats()
        stats["timestamp"] = time.time()
        return stats

    # ------------------------------------------------------------------
    # 내부 — MQTT 연결
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = (rc == 0)
        self._connect_waiter.set()
        if self._connected:
            self._reconnect_delay = _RECONNECT_MIN_DELAY
            logger.info("[StatusReporter] MQTT 연결 성공: %s:%d", self._broker, self._port)
        else:
            logger.warning("[StatusReporter] MQTT 연결 실패 (rc=%d)", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("[StatusReporter] MQTT 연결 끊김 (rc=%d), 재연결 대기", rc)

    def _ensure_connected(self) -> bool:
        if self._connected:
            return True

        now = time.monotonic()
        wait = self._reconnect_delay - (now - self._last_attempt_time)
        if wait > 0:
            return False
        self._last_attempt_time = now

        if self._client is None:
            self._client = mqtt.Client(
                client_id=f"cctv-status-reporter-{uuid.uuid4().hex[:8]}",
                clean_session=True,
            )
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

        try:
            self._connect_waiter.clear()
            self._client.connect(self._broker, self._port, keepalive=60)
            self._client.loop_start()
            connected = self._connect_waiter.wait(timeout=_CONNECT_TIMEOUT)
            if connected and self._connected:
                return True
        except Exception as exc:
            logger.warning("[StatusReporter] MQTT 연결 오류: %s", exc)

        self._reconnect_delay = min(
            self._reconnect_delay * _RECONNECT_MULTIPLIER, _RECONNECT_MAX_DELAY
        )
        return False

    def _publish(self, topic: str, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        result = self._client.publish(topic, data, qos=self._qos)
        if result.rc != 0:
            logger.warning("[StatusReporter] 발행 실패 (rc=%d): %s", result.rc, topic)

    # ------------------------------------------------------------------
    # 내부 — 루프
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                if self._ensure_connected():
                    camera_status = self._processor.get_camera_status()
                    snapshots = self._processor.get_detection_snapshot()

                    # 카메라별 토픽 발행
                    all_ids = set(camera_status) | set(snapshots)
                    for cam_id in all_ids:
                        payload = self._build_camera_payload(cam_id, camera_status, snapshots)
                        topic = f"{self._topic_prefix}/cameras/{cam_id}"
                        self._publish(topic, payload)

                    # 전체 통계 토픽 발행
                    self._publish(
                        f"{self._topic_prefix}/stats",
                        self._build_stats_payload(),
                    )

                    logger.debug(
                        "[StatusReporter] 발행 완료 (%d cameras)", len(all_ids)
                    )
            except Exception as exc:
                logger.error("[StatusReporter] 예외: %s", exc, exc_info=True)

            time.sleep(self._interval)
