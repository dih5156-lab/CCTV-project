"""
경량 EdgeX 디바이스 서비스 어댑터

역할:
- CCTV AI 엔진 MQTT 이벤트 구독
- EdgeX 메타데이터(DeviceService/Profile/Device) 관리
- EdgeX 메시지버스 토픽으로 이벤트 재발행
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Event, Thread
from typing import Dict, Optional, Set

import paho.mqtt.client as mqtt
import redis

from .device_service import CCTVDeviceService
from ..config import EdgeXConfig

logger = logging.getLogger(__name__)

_EDGEX_DEFAULTS = EdgeXConfig()


class EdgeXDeviceAdapterService:
    """AI 엔진 -> EdgeX 브리지 어댑터 (경량 구독형 서비스)"""

    SENSOR_TOPIC = "aiot/rules/sensor/#"

    def __init__(
        self,
        ai_mqtt_broker: str = "localhost",
        ai_mqtt_port: int = 1883,
        ai_topic_prefix: str = "cctv/ai/events",
        metadata_url: str = _EDGEX_DEFAULTS.metadata_url,
        data_url: str = _EDGEX_DEFAULTS.data_url,
        edgex_mqtt_broker: str = _EDGEX_DEFAULTS.mqtt_broker,
        edgex_mqtt_port: int = _EDGEX_DEFAULTS.mqtt_port,
        edgex_topic_prefix: str = "edgex/events/device",
        service_name: str = "cctv-device-service",
        outbox_db_path: Optional[str] = None,
    ):
        self.ai_mqtt_broker = ai_mqtt_broker
        self.ai_mqtt_port = int(ai_mqtt_port)
        self.ai_topic_prefix = ai_topic_prefix.rstrip("/")
        self.subscribe_topic = f"{self.ai_topic_prefix}/#"
        self.service_name = service_name

        self._subscriber: Optional[mqtt.Client] = None
        self._registered_cameras: Set[str] = set()
        self._validation_stop = Event()
        self._validation_thread: Optional[Thread] = None
        self._validation_redis: Optional[redis.Redis] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_loop_thread: Optional[Thread] = None
        self._async_loop_ready = Event()
        self._outbox_stop = Event()
        self._outbox_thread: Optional[Thread] = None
        self._outbox_poll_interval_seconds = 5.0
        self._coro_timeout_seconds = 10.0

        # 환경변수 우선, 인자값, 기본값 순으로 outbox 경로 결정
        resolved_outbox = (
            outbox_db_path
            or os.environ.get("EDGEX_OUTBOX_DB")
            or "data/detection_outbox.db"
        )

        self.edgex_service = CCTVDeviceService(
            {
                "coreMetadataUrl": metadata_url,
                "coreDataUrl": data_url,
                "deviceServiceName": service_name,
                "baseUrl": "http://cctv-device-service:59986",
                "mqttBroker": edgex_mqtt_broker,
                "mqttPort": edgex_mqtt_port,
                "mqttTopicPrefix": edgex_topic_prefix,
                "messageBusType": "redis",
                "redisHost": "edgex-redis",
                "redisPort": 6379,
                "outboxDbPath": resolved_outbox,
            }
        )

    def _start_async_loop(self) -> None:
        if self._async_loop and self._async_loop.is_running():
            return

        self._async_loop_ready.clear()

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._async_loop = loop
            self._async_loop_ready.set()

            try:
                loop.run_forever()
            finally:
                pending_tasks = asyncio.all_tasks(loop)
                for task in pending_tasks:
                    task.cancel()
                if pending_tasks:
                    try:
                        loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
                    except Exception:
                        pass
                loop.close()

        self._async_loop_thread = Thread(target=_run_loop, daemon=True, name="adapter-async-loop")
        self._async_loop_thread.start()

        if not self._async_loop_ready.wait(timeout=2):
            raise RuntimeError("비동기 이벤트 루프 초기화 실패")

    def _run_coro(self, coroutine):
        if not self._async_loop or not self._async_loop.is_running():
            raise RuntimeError("비동기 이벤트 루프가 실행 중이 아닙니다")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._async_loop)
        try:
            return future.result(timeout=self._coro_timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError(
                f"코루틴 실행 시간 초과 ({self._coro_timeout_seconds}초)"
            ) from error

    def _stop_async_loop(self) -> None:
        loop = self._async_loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

        if self._async_loop_thread and self._async_loop_thread.is_alive():
            self._async_loop_thread.join(timeout=3)

        self._async_loop = None
        self._async_loop_thread = None
        self._async_loop_ready.clear()

    def _start_validation_responder(self) -> None:
        if self._validation_thread and self._validation_thread.is_alive():
            return

        if getattr(self.edgex_service, "message_bus_type", "redis") != "redis":
            logger.info("검증 응답기 비활성화 (메시지버스가 redis가 아님)")
            return

        request_channels = [
            f"edgex.{self.service_name}.validate.device",
            f"edgex/{self.service_name}/validate/device",
        ]

        def _run() -> None:
            pubsub = None
            try:
                client = redis.Redis(
                    host=self.edgex_service.redis_host,
                    port=self.edgex_service.redis_port,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                )
                client.ping()
                self._validation_redis = client
                pubsub = client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(*request_channels)
                logger.info("검증 응답기 시작: %s", ', '.join(request_channels))

                while not self._validation_stop.is_set():
                    message = pubsub.get_message(timeout=1.0)
                    if not message or message.get("type") != "message":
                        continue

                    raw_data = message.get("data")
                    if not raw_data:
                        continue

                    try:
                        envelope = json.loads(raw_data)
                    except Exception:
                        logger.warning("검증 요청 JSON 파싱 실패")
                        continue

                    request_id = envelope.get("requestID") or envelope.get("requestId") or ""
                    if not request_id:
                        logger.warning("검증 요청 requestID 누락")
                        continue

                    request_topic = message.get("channel", "")
                    response_channel_dot = f"edgex.response.{self.service_name}.{request_id}"
                    response_channel_slash = f"edgex/response/{self.service_name}/{request_id}"
                    response = {
                        "apiVersion": "",
                        "receivedTopic": request_topic,
                        "correlationID": envelope.get("correlationID", ""),
                        "requestID": request_id,
                        "errorCode": 0,
                        "payload": "",
                        "contentType": "application/json",
                    }
                    payload = json.dumps(response, ensure_ascii=False)
                    client.publish(response_channel_dot, payload)
                    client.publish(response_channel_slash, payload)
            except Exception as error:
                logger.error("검증 응답기 오류: %s", error, exc_info=True)
            finally:
                try:
                    if pubsub:
                        pubsub.close()
                except Exception:
                    pass

        self._validation_stop.clear()
        self._validation_thread = Thread(target=_run, daemon=True, name="validation-responder")
        self._validation_thread.start()

    async def initialize(self) -> None:
        """EdgeX 메타데이터 준비 (service/profile 등록)"""
        logger.info("EdgeX 어댑터 초기화 시작")
        await self.edgex_service.initialize()

        service_ok = await self.edgex_service.register_device_service()
        if not service_ok:
            logger.warning("디바이스 서비스 등록에 실패했지만 실행은 계속합니다")

        profile_ok = await self.edgex_service.create_device_profile()
        if not profile_ok:
            logger.warning("디바이스 프로파일 생성에 실패했지만 실행은 계속합니다")

        logger.info("EdgeX 어댑터 초기화 완료")

    def _extract_from_topic(self, topic: str) -> Dict[str, str]:
        """토픽에서 camera_id 와 event_type 추출.

        지원 형식:
        - cctv/ai/events/{camera_id}/{event_type}
        - aiot/rules/sensor/{event_type}   (sensor 이벤트)
        """
        parts = topic.split("/")

        # 센서 토픽: aiot/rules/sensor/{type}
        if topic.startswith("aiot/rules/sensor/"):
            event_type = parts[3] if len(parts) > 3 else "sensor_data"
            return {"camera_id": "sensor", "event_type": f"{event_type}_alert" if not event_type.endswith("_alert") else event_type}

        # AI 이벤트 토픽: cctv/ai/events/{camera_id}/{event_type}
        prefix_parts = self.ai_topic_prefix.split("/")

        if len(parts) <= len(prefix_parts):
            return {"camera_id": "unknown", "event_type": "unknown"}

        camera_id = parts[len(prefix_parts)] if len(parts) > len(prefix_parts) else "unknown"
        event_type = parts[len(prefix_parts) + 1] if len(parts) > len(prefix_parts) + 1 else "unknown"

        return {"camera_id": camera_id, "event_type": event_type}

    def _ensure_camera_registered(self, camera_id: str, rtsp_source: Optional[str] = None) -> bool:
        if not camera_id or camera_id == "unknown":
            return False

        if camera_id in self._registered_cameras:
            return True

        source = rtsp_source or f"adapter://{camera_id}"

        try:
            device_id = self._run_coro(self.edgex_service.add_camera(camera_id, source))
            if not device_id:
                return False
            self._registered_cameras.add(camera_id)
            return True
        except Exception as error:
            logger.error("카메라 등록 실패 (%s): %s", camera_id, error)
            return False

    def _replay_outbox_once(self) -> None:
        expired = self.edgex_service.expire_pending_detection_events()
        if expired:
            logger.info("EdgeX outbox 만료 정리 완료: %d건", expired)

        pending = self.edgex_service.get_pending_detection_events()
        if not pending:
            return

        replayed = 0
        for row in pending:
            camera_id = str(row.get("camera_id") or "unknown")
            event_data = row.get("event_data") or {}
            outbox_ref = (row["_table"], int(row["id"])) if "_table" in row else int(row["id"])
            rtsp_source = event_data.get("source") if isinstance(event_data, dict) else None

            if not self._ensure_camera_registered(camera_id, rtsp_source=rtsp_source):
                logger.debug("Outbox 재전송 보류: 카메라 등록 실패 (%s)", camera_id)
                continue

            sent = self._run_coro(
                self.edgex_service.replay_detection_event(outbox_ref, camera_id, event_data)
            )
            if sent:
                replayed += 1

        if replayed:
            logger.info("EdgeX outbox 재전송 완료: %d건", replayed)

    def _start_outbox_replay_worker(self) -> None:
        if self._outbox_thread and self._outbox_thread.is_alive():
            return

        def _run() -> None:
            while not self._outbox_stop.is_set():
                try:
                    self._replay_outbox_once()
                except Exception as error:
                    logger.error("EdgeX outbox 재전송 워커 오류: %s", error, exc_info=True)
                self._outbox_stop.wait(self._outbox_poll_interval_seconds)

        self._outbox_stop.clear()
        self._outbox_thread = Thread(target=_run, daemon=True, name="edgex-outbox-replay")
        self._outbox_thread.start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("AI MQTT 구독 연결 성공: %s:%s", self.ai_mqtt_broker, self.ai_mqtt_port)
            client.subscribe(self.subscribe_topic, qos=0)
            client.subscribe(self.SENSOR_TOPIC, qos=0)
            logger.info("구독 시작: %s, %s", self.subscribe_topic, self.SENSOR_TOPIC)
        else:
            logger.error("AI MQTT 구독 연결 실패 (rc=%s)", rc)

    def _on_message(self, client, userdata, msg):
        try:
            topic_info = self._extract_from_topic(msg.topic)
            payload = json.loads(msg.payload.decode("utf-8"))

            # Kuiper는 결과를 JSON 배열로 발행 → 개별 dict 로 순회
            items = payload if isinstance(payload, list) else [payload]

            for event_data in items:
                if not isinstance(event_data, dict):
                    logger.warning("비정상 페이로드 항목 무시: %s", type(event_data).__name__)
                    continue

                # 센서 이벤트는 device_id 를 camera_id 로 사용
                camera_id = (
                    event_data.get("camera_id")
                    or event_data.get("device_id")
                    or topic_info["camera_id"]
                )
                event_type = event_data.get("type") or topic_info["event_type"]
                if "type" not in event_data:
                    event_data["type"] = event_type
                if "camera_id" not in event_data:
                    event_data["camera_id"] = camera_id

                rtsp_source = event_data.get("source")
                if not self._ensure_camera_registered(camera_id, rtsp_source=rtsp_source):
                    logger.warning("메타데이터 등록 실패로 이벤트 스킵: camera_id=%s", camera_id)
                    continue

                published = self._run_coro(self.edgex_service.send_detection_event(camera_id, [event_data]))
                if not published:
                    logger.warning("EdgeX 이벤트 발행 실패: camera_id=%s, type=%s", camera_id, event_type)

        except json.JSONDecodeError as error:
            logger.error("JSON 파싱 실패: %s", error)
        except Exception as error:
            logger.error("메시지 처리 오류: %s", error, exc_info=True)

    def start(self) -> None:
        """서비스 시작 (블로킹)"""
        self._start_async_loop()
        self._run_coro(self.initialize())
        self._start_validation_responder()
        self._start_outbox_replay_worker()

        self._subscriber = mqtt.Client()
        self._subscriber.on_connect = self._on_connect
        self._subscriber.on_message = self._on_message

        logger.info("AI MQTT 브로커 연결 중...")
        self._subscriber.connect(self.ai_mqtt_broker, self.ai_mqtt_port, keepalive=60)
        self._subscriber.loop_start()

        logger.info("EdgeX 디바이스 어댑터 실행 중 (Ctrl+C 종료)")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("종료 신호 감지")
        finally:
            self.stop()

    def stop(self) -> None:
        self._outbox_stop.set()
        if self._outbox_thread and self._outbox_thread.is_alive():
            self._outbox_thread.join(timeout=2)
        self._outbox_thread = None
        self._validation_stop.set()
        if self._validation_thread and self._validation_thread.is_alive():
            self._validation_thread.join(timeout=2)
        if self._validation_redis:
            try:
                self._validation_redis.close()
            except Exception:
                pass

        if self._subscriber:
            try:
                self._subscriber.loop_stop()
                self._subscriber.disconnect()
            except Exception as error:
                logger.error("구독 클라이언트 종료 오류: %s", error)

        if self.edgex_service._mqtt_client:
            try:
                self.edgex_service._mqtt_client.loop_stop()
                self.edgex_service._mqtt_client.disconnect()
            except Exception as error:
                logger.error("EdgeX MQTT 클라이언트 종료 오류: %s", error)

        self._stop_async_loop()
