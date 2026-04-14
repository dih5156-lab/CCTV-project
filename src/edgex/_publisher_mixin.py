"""
EdgeX 메시지 버스 퍼블리셔 믹스인

Redis Pub/Sub 및 MQTT 를 통해 EdgeX 이벤트를 발행하고,
지수 백오프 재연결 로직을 관리한다.
각 메서드는 self.redis_host, self.redis_port, self.mqtt_broker,
self.mqtt_port, self.mqtt_topic_prefix, self.service_name,
self.PROFILE_NAME 에 의존한다.
"""

import base64
import json
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    import redis as redis_module

import paho.mqtt.client as mqtt

try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# 연결 시도 간 최소 간격을 보장하기 위한 모듈 수준 락
_redis_lock = threading.Lock()
_mqtt_lock = threading.Lock()


class _PublisherMixin:
    """Redis/MQTT 메시지 버스 발행 및 연결 관리 믹스인."""

    _redis_lock = _redis_lock
    _mqtt_lock = _mqtt_lock
    _redis_base_cooldown_sec: float = 5
    _mqtt_base_cooldown_sec: float = 5
    _max_cooldown_sec: float = 60

    # ── 연결 보장 (지수 백오프) ──────────────────────────────────────────────

    def _ensure_connection(
        self,
        bus_type: str,
        lock: threading.Lock,
        connect_fn,
    ) -> bool:
        """Redis/MQTT 공통 연결 보장 헬퍼.

        Args:
            bus_type: "redis" 또는 "mqtt"
            lock: 해당 버스의 Lock
            connect_fn: 실제 연결 콜백. 성공 시 클라이언트 객체 반환.
        """
        client_attr = f"_{bus_type}_client"
        fail_count_attr = f"_{bus_type}_fail_count"
        last_fail_attr = f"_{bus_type}_last_fail_time"
        base_cooldown = getattr(type(self), f"_{bus_type}_base_cooldown_sec")

        with lock:
            if getattr(self, client_attr):
                return True
            now = time.time()
            fail_count = getattr(self, fail_count_attr)
            cooldown = min(
                base_cooldown * (2 ** fail_count),
                type(self)._max_cooldown_sec,
            )
            if now - getattr(self, last_fail_attr) < cooldown:
                logger.debug(
                    "%s 재연결 쿨다운 중 (%.1f초 대기, 실패 횟수=%d)",
                    bus_type.upper(),
                    cooldown - (now - getattr(self, last_fail_attr)),
                    fail_count,
                )
                return False
            try:
                client = connect_fn()
                setattr(self, client_attr, client)
                setattr(self, fail_count_attr, 0)
                logger.info("✓ %s 연결됨", bus_type.upper())
                return True
            except Exception as exc:
                new_count = fail_count + 1
                setattr(self, fail_count_attr, new_count)
                setattr(self, last_fail_attr, now)
                next_cd = min(
                    base_cooldown * (2 ** new_count),
                    type(self)._max_cooldown_sec,
                )
                logger.warning(
                    "%s 연결 실패 (횟수=%d, 다음 재시도 %.0f초 후): %s",
                    bus_type.upper(), new_count, next_cd, exc,
                )
                setattr(self, client_attr, None)
                return False

    def _ensure_redis_client(self) -> bool:
        def _connect():
            client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=0,
                socket_connect_timeout=3,
                socket_timeout=3,
                decode_responses=True,
            )
            client.ping()
            return client

        return self._ensure_connection("redis", type(self)._redis_lock, _connect)

    def _ensure_mqtt_client(self) -> bool:
        def _connect():
            client = mqtt.Client()
            client.connect(self.mqtt_broker, self.mqtt_port, 60)
            client.loop_start()
            return client

        return self._ensure_connection("mqtt", type(self)._mqtt_lock, _connect)

    # ── Redis 발행 ───────────────────────────────────────────────────────────

    def _publish_event_redis(
        self,
        device_name: str,
        resource_name: str,
        event_type: str,
        confidence: float,
        x: int,
        y: int,
        width: int,
        height: int,
        object_id: Optional[int],
        timestamp: str,
    ) -> bool:
        """Redis Pub/Sub 를 통해 EdgeX v3 envelope 형식으로 이벤트 발행."""
        if not self._ensure_redis_client():
            return False

        try:
            bundle = self._build_detection_payload_bundle(
                device_name, resource_name, event_type,
                confidence, x, y, width, height, object_id, timestamp,
            )
            event_payload = bundle["event_payload"]

            payload_raw = json.dumps(event_payload, separators=(",", ":"), ensure_ascii=False)
            payload_b64 = base64.b64encode(payload_raw.encode("utf-8")).decode("utf-8")

            envelope = {
                "apiVersion": "",
                "receivedTopic": "",
                "correlationID": str(uuid.uuid4()),
                "requestID": "",
                "errorCode": 0,
                "payload": payload_b64,
                "contentType": "application/json",
            }

            topic_prefix = self.mqtt_topic_prefix.replace("/", ".")
            channel = (
                f"{topic_prefix}.{self.service_name}."
                f"{self.PROFILE_NAME}.{device_name}.{resource_name}"
            )
            publish_count = self._redis_client.publish(
                channel, json.dumps(envelope, ensure_ascii=False)
            )

            if publish_count >= 0:
                logger.info("✓ Redis 발행 성공: %s (subscribers=%s)", channel, publish_count)
                return True

            logger.error("Redis 발행 실패: %s", channel)
            return False
        except Exception as exc:
            logger.error("Redis 전송 오류: %s", exc, exc_info=True)
            return False

    # ── MQTT 발행 ────────────────────────────────────────────────────────────

    def _publish_event_mqtt(
        self,
        device_name: str,
        resource_name: str,
        event_type: str,
        confidence: float,
        x: int,
        y: int,
        width: int,
        height: int,
        object_id: Optional[int],
        timestamp: str,
    ) -> bool:
        """MQTT 를 통해 EdgeX v3 envelope 형식으로 이벤트 발행."""
        if not self._ensure_mqtt_client():
            return False

        try:
            logger.info(
                "MQTT 이벤트 발행 시작: device=%s, resource=%s, type=%s",
                device_name, resource_name, event_type,
            )
            bundle = self._build_detection_payload_bundle(
                device_name, resource_name, event_type,
                confidence, x, y, width, height, object_id, timestamp,
            )
            envelope = bundle["envelope"]

            topic = (
                f"{self.mqtt_topic_prefix}/{self.service_name}"
                f"/{device_name}/{resource_name}"
            )
            logger.info("MQTT 토픽: %s", topic)

            result = self._mqtt_client.publish(topic, json.dumps(envelope), qos=0)
            if result.rc == 0:
                logger.info("✓ MQTT 발행 성공: %s (mid=%s)", topic, result.mid)
                return True

            logger.error("MQTT 발행 실패: %s (rc=%s)", topic, result.rc)
            return False
        except Exception as exc:
            logger.error("MQTT 전송 오류: %s", exc, exc_info=True)
            return False

    # ── 범용 디바이스 이벤트 발행 ────────────────────────────────────────────

    def publish_device_event(
        self,
        device_id: str,
        device_type: str,
        resource_name: str,
        event_data: Dict,
    ) -> bool:
        """다양한 디바이스 타입을 지원하는 통합 MQTT 이벤트 발행 인터페이스.

        Args:
            device_id:    디바이스 ID (예: camera-1, thermal-1)
            device_type:  디바이스 타입 (예: cctv, thermal, sensor)
            resource_name: 리소스명 (예: helmet_detection, temperature)
            event_data:   이벤트 데이터 딕셔너리
        """
        if not self._ensure_mqtt_client():
            return False

        try:
            logger.info("범용 디바이스 이벤트 발행: %s/%s", device_id, resource_name)

            timestamp_raw = event_data.get("timestamp")
            timestamp = self._normalize_timestamp(
                timestamp_raw if timestamp_raw is not None else None
            )
            origin = self._to_origin_nanos(
                timestamp_raw if timestamp_raw is not None else timestamp
            )

            payload_value = {
                "type": event_data.get("type", "unknown"),
                "device": device_id,
                "device_type": device_type,
                "resource": resource_name,
                "confidence": event_data.get("confidence", 0.0),
                "value": event_data.get("value"),
                "bbox": event_data.get("bbox"),
                "object_id": event_data.get("object_id"),
                "timestamp": timestamp,
                "metadata": {
                    "service": self.service_name,
                    "version": "v1",
                    "device_type": device_type,
                },
            }
            event_payload = self._build_event_payload(
                device_id, resource_name, origin, payload_value
            )
            envelope = self._build_envelope(event_payload)

            topic = (
                f"{self.mqtt_topic_prefix}/{self.service_name}"
                f"/{device_type}/{device_id}/{resource_name}"
            )
            logger.info("MQTT 토픽: %s", topic)

            result = self._mqtt_client.publish(
                topic, json.dumps(envelope, ensure_ascii=False), qos=0
            )
            if result.rc == 0:
                logger.info("✓ 범용 디바이스 이벤트 발행 성공: %s (mid=%s)", topic, result.mid)
                return True

            logger.error("범용 디바이스 이벤트 발행 실패: %s (rc=%s)", topic, result.rc)
            return False
        except Exception as exc:
            logger.error("범용 디바이스 이벤트 발행 오류: %s", exc, exc_info=True)
            return False

    # ── 연결 정리 ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """열려 있는 메시지 버스 연결 정리."""
        try:
            if self._mqtt_client:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
        except Exception as exc:
            logger.debug("MQTT 연결 정리 중 오류 (무시됨): %s", exc)
        finally:
            self._mqtt_client = None

        try:
            if self._redis_client:
                self._redis_client.close()
        except Exception as exc:
            logger.debug("Redis 연결 정리 중 오류 (무시됨): %s", exc)
        finally:
            self._redis_client = None
