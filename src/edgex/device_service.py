"""
CCTV용 EdgeX 디바이스 서비스
CCTV 카메라를 EdgeX Foundry 장치로 관리

책임 분리:
  _OutboxMixin    — SQLite store-and-forward 저장/재전송
  _HttpMixin      — EdgeX REST HTTP 통신 (버전 폴백, 헬스 프로브 등)
  _PayloadMixin   — EdgeX 페이로드 구성 및 데이터 변환
  _PublisherMixin — Redis/MQTT 발행 및 지수 백오프 연결 관리
  CCTVDeviceService — 위 믹스인 조합 + 카메라 등록/이벤트 전송 조율
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# 테스트 패치 호환을 위해 redis/mqtt 를 이 모듈 네임스페이스에도 노출한다.
import paho.mqtt.client as mqtt  # noqa: F401

try:
    import redis  # noqa: F401
except ImportError:
    redis = None  # type: ignore[assignment]

from ._http_mixin import _HttpMixin
from ._outbox_mixin import _OutboxMixin
from ._payload_mixin import _PayloadMixin
from ._publisher_mixin import _PublisherMixin

logger = logging.getLogger(__name__)


class CCTVDeviceService(_OutboxMixin, _HttpMixin, _PayloadMixin, _PublisherMixin):
    """EdgeX CCTV 장치 서비스.

    CCTV 카메라를 EdgeX Foundry 장치로 관리하며, 탐지 이벤트를
    Redis Message Bus 또는 MQTT 채널로 발행한다.
    연결 실패 시 SQLite Outbox 에 이벤트를 보관하고 복구 후 재전송한다.
    """

    PROFILE_NAME = "CCTV-Camera-Profile"

    def __init__(self, config: Dict):
        self.metadata_url = config.get("coreMetadataUrl", "http://localhost:59881")
        self.data_url = config.get("coreDataUrl", "http://localhost:59880")
        self.service_name = config.get("deviceServiceName", "cctv-device-service")
        self.mqtt_broker = config.get("mqttBroker", "localhost")
        self.mqtt_port = int(config.get("mqttPort", 1883))
        self.mqtt_topic_prefix = config.get("mqttTopicPrefix", "edgex/events/device")
        self.redis_host = config.get("redisHost", "edgex-redis")
        self.redis_port = int(config.get("redisPort", 6379))
        self.message_bus_type = str(config.get("messageBusType", "redis")).lower()
        self.enable_rest_event_post = self._to_bool(config.get("enableRestEventPost", False))
        self.enable_store_and_forward = self._to_bool(config.get("enableStoreAndForward", True))
        self.outbox_db_path = Path(config.get("outboxDbPath", "data/detection_outbox.db"))
        self.outbox_flush_batch_size = int(config.get("outboxFlushBatchSize", 100))
        self._mqtt_client: Optional[object] = None
        self._redis_client: Optional[object] = None
        self._redis_last_fail_time = 0.0
        self._mqtt_last_fail_time = 0.0
        self._redis_fail_count = 0
        self._mqtt_fail_count = 0
        self._outbox_lock = threading.Lock()
        self.base_url = config.get("baseUrl", "http://cctv-device-service:59986")
        self.devices: Dict[str, str] = {}  # camera_id -> device_id 매핑

        logger.info("EdgeX Device Service 초기화: %s", self.service_name)
        logger.info("  - Metadata URL: %s", self.metadata_url)
        logger.info("  - Data URL: %s", self.data_url)
        if self.enable_store_and_forward:
            logger.info("  - Store-and-forward DB: %s", self.outbox_db_path)
        self._init_outbox()

    # ── 초기화 ───────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """EdgeX 서비스 연결 확인."""
        try:
            self._init_outbox()
            await self._probe_service_health(self.metadata_url, "Core Metadata")
            await self._probe_service_health(self.data_url, "Core Data")
        except Exception as exc:
            logger.error("EdgeX 연결 오류: %s", exc)

    # ── 카메라 등록 ──────────────────────────────────────────────────────────

    async def add_camera(self, camera_id: str, rtsp_source: str) -> Optional[str]:
        """카메라를 EdgeX 장치로 등록하고 device_id 를 반환한다."""
        try:
            device_name = f"camera-{camera_id}"

            existing_device = await self._get_device_by_name(device_name)
            if existing_device:
                existing_service = existing_device.get("serviceName", "")
                if existing_service == self.service_name:
                    self.devices[camera_id] = existing_device.get("id") or device_name
                    logger.info("✓ 카메라 이미 존재(서비스 일치): %s -> %s", camera_id, device_name)
                    return self.devices[camera_id]

                logger.warning(
                    "기존 디바이스 서비스 불일치 감지: %s (%s -> %s)",
                    device_name, existing_service, self.service_name,
                )
                if not await self._delete_device_by_name(device_name):
                    logger.error("기존 디바이스 삭제 실패로 재등록 중단: %s", device_name)
                    return None
                await asyncio.sleep(0.3)

            rtsp_conn = self._parse_rtsp_address_port(rtsp_source)
            device_payload = {
                "apiVersion": "v2",
                "device": {
                    "name": device_name,
                    "description": f"CCTV Camera {camera_id}",
                    "adminState": "UNLOCKED",
                    "operatingState": "UP",
                    "profileName": self.PROFILE_NAME,
                    "serviceName": self.service_name,
                    "protocols": {
                        "rtsp": {
                            "Address": rtsp_conn["Address"],
                            "Port": rtsp_conn["Port"],
                            "URL": rtsp_source,
                        }
                    },
                    "labels": ["cctv", f"camera_{camera_id}"],
                },
            }

            for endpoint in self._versioned_endpoints(self.metadata_url, "device"):
                try:
                    payload = self._payload_for_endpoint(endpoint, device_payload)
                    response = await self._request_post(endpoint, payload, timeout=10)
                    if response is None:
                        continue
                    status_code = self._response_status_code(response)

                    if status_code in [200, 201]:
                        device_id = self._response_id(response) or device_name
                        self.devices[camera_id] = device_id
                        logger.info(
                            "✓ 카메라 등록 성공: %s -> %s (ID: %s)", camera_id, device_name, device_id
                        )
                        return device_id

                    if status_code == 404:
                        logger.debug("엔드포인트 없음: %s", endpoint)
                        continue

                    if status_code == 409:
                        existing = await self._get_device_by_name(device_name)
                        existing_service = (existing or {}).get("serviceName", "")
                        if existing_service == self.service_name:
                            self.devices[camera_id] = (
                                existing.get("id") if existing else device_name
                            )
                            logger.info("✓ 카메라 이미 존재: %s -> %s", camera_id, device_name)
                            return self.devices[camera_id]

                        logger.warning(
                            "기존 디바이스 서비스 불일치(충돌 응답): %s (%s -> %s)",
                            device_name, existing_service, self.service_name,
                        )
                        if await self._delete_device_by_name(device_name):
                            await asyncio.sleep(0.3)
                            continue
                        logger.error("기존 디바이스 삭제 실패로 재등록 중단: %s", device_name)
                        return None

                    if response.status_code == 207:
                        logger.warning(
                            "Device 등록 실패 (%s): 207 응답 - %s", camera_id, response.text
                        )
                        continue

                    logger.warning(
                        "Device 등록 실패 (%s): %s - %s",
                        camera_id, status_code, self._describe_http_status(status_code),
                    )
                    continue
                except Exception as exc:
                    logger.debug("엔드포인트 %s 시도 실패: %s", endpoint, exc)
                    continue

            logger.error("카메라 등록 실패: %s - 모든 엔드포인트 시도 완료", camera_id)
            return None

        except Exception as exc:
            logger.error("카메라 등록 오류 (%s): %s", camera_id, exc)
            return None

    async def _get_device_by_name(self, device_name: str) -> Optional[Dict[str, object]]:
        return await self._get_entity_by_name(f"device/name/{device_name}", "device")

    async def _delete_device_by_name(self, device_name: str) -> bool:
        return await self._delete_entity_by_name(
            f"device/name/{device_name}",
            success_log=f"✓ 기존 디바이스 삭제 완료: {device_name}",
        )

    # ── 이벤트 전송 ──────────────────────────────────────────────────────────

    async def _send_detection_event_payload(
        self,
        camera_id: str,
        event_data: Dict[str, Any],
        persist_on_failure: bool = True,
    ) -> bool:
        if camera_id not in self.devices:
            error_message = f"camera not registered: {camera_id}"
            logger.warning("등록되지 않은 카메라: %s", camera_id)
            if persist_on_failure:
                self._store_failed_detection_event(camera_id, event_data, error_message)
            return False

        outbox_row_id = (
            self._store_pending_event(camera_id, event_data) if persist_on_failure else None
        )

        device_name = f"camera-{camera_id}"
        fields = self._extract_event_fields(event_data)
        event_type = fields["event_type"]
        resource_name = self._map_event_type_to_resource(event_type)

        if self.message_bus_type == "redis":
            redis_ok = await asyncio.to_thread(
                self._publish_event_redis,
                device_name, resource_name, event_type,
                fields["confidence"], fields["x"], fields["y"],
                fields["width"], fields["height"], fields["object_id"],
                fields["timestamp"],
            )
            if redis_ok:
                logger.info("✓[%s] Redis 이벤트 전송: %s", camera_id, event_type)
                self._mark_outbox_sent(outbox_row_id)
                return True

        mqtt_ok = await asyncio.to_thread(
            self._publish_event_mqtt,
            device_name, resource_name, event_type,
            fields["confidence"], fields["x"], fields["y"],
            fields["width"], fields["height"], fields["object_id"],
            fields["timestamp"],
        )
        if mqtt_ok:
            logger.info("✓[%s] MQTT 이벤트 전송: %s", camera_id, event_type)
            self._mark_outbox_sent(outbox_row_id)
            return True

        if self.enable_rest_event_post:
            bundle = self._build_detection_payload_bundle(
                device_name, resource_name, event_type,
                fields["confidence"], fields["x"], fields["y"],
                fields["width"], fields["height"], fields["object_id"],
                fields["timestamp"],
            )
            base_event = {"event": bundle["event_payload"]["event"]}
            rest_ok = await self._post_event_via_rest(camera_id, event_type, base_event)
            if rest_ok:
                self._mark_outbox_sent(outbox_row_id)
                return True

        error_message = (
            f"EdgeX publish failed: camera={camera_id}, type={event_type}, "
            f"message_bus={self.message_bus_type}, rest={self.enable_rest_event_post}"
        )
        logger.warning(error_message)
        if persist_on_failure and outbox_row_id is None:
            self._store_failed_detection_event(camera_id, event_data, error_message)
        return False

    async def replay_detection_event(
        self,
        outbox_ref,
        camera_id: str,
        event_data: Dict[str, Any],
    ) -> bool:
        """저장된 outbox 이벤트를 EdgeX 로 재전송한다.

        Args:
            outbox_ref: (table, row_id) tuple 또는 int (하위 호환)
        """
        try:
            sent = await self._send_detection_event_payload(
                camera_id, event_data, persist_on_failure=False,
            )
            if sent:
                self._mark_outbox_sent(outbox_ref)
                return True
            self._mark_outbox_retry_failed(outbox_ref, "replay failed")
            return False
        except Exception as exc:
            self._mark_outbox_retry_failed(outbox_ref, str(exc))
            logger.error("Outbox replay 오류 (%s): %s", outbox_ref, exc)
            return False

    async def send_detection_event(self, camera_id: str, events: List) -> bool:
        """감지 이벤트 목록을 EdgeX 로 전송한다."""
        try:
            all_sent = True
            for event in events:
                sent = await self._send_detection_event_payload(camera_id, event)
                all_sent = all_sent and sent
            return all_sent
        except Exception as exc:
            logger.error("이벤트 전송 오류 (%s): %s", camera_id, exc)
            return False

    # ── 서비스/프로필 등록 ───────────────────────────────────────────────────

    async def register_device_service(self) -> bool:
        """Device Service 를 EdgeX 에 등록한다."""
        try:
            existing = await self._get_device_service_by_name(self.service_name)
            if existing:
                existing_base = existing.get("baseAddress", "")
                if existing_base and existing_base != self.base_url:
                    logger.warning(
                        "기존 Device Service baseAddress 불일치: %s -> %s",
                        existing_base, self.base_url,
                    )
                    if await self._delete_device_service_by_name(self.service_name):
                        logger.info("✓ 기존 Device Service 삭제 완료: %s", self.service_name)
                    else:
                        logger.warning("기존 Device Service 삭제 실패: %s", self.service_name)

            service_payload = {
                "apiVersion": "v2",
                "service": {
                    "name": self.service_name,
                    "description": "CCTV Detection Device Service",
                    "labels": ["cctv", "detection"],
                    "baseAddress": self.base_url,
                    "adminState": "UNLOCKED",
                },
            }
            result = await self._post_with_versioned_fallback(
                self.metadata_url, "deviceservice", service_payload, "Device Service 등록",
            )
            return result in {"success", "exists"}
        except Exception as exc:
            logger.error("Service 등록 오류: %s", exc)
            return False

    async def _get_device_service_by_name(
        self, service_name: str
    ) -> Optional[Dict[str, object]]:
        return await self._get_entity_by_name(
            f"deviceservice/name/{service_name}", "service"
        )

    async def _delete_device_service_by_name(self, service_name: str) -> bool:
        return await self._delete_entity_by_name(f"deviceservice/name/{service_name}")

    async def create_device_profile(self) -> bool:
        """CCTV 장치 프로필을 EdgeX 에 등록한다."""
        try:
            profile_payload = {
                "apiVersion": "v2",
                "profile": {
                    "name": self.PROFILE_NAME,
                    "description": "CCTV Camera Detection Profile",
                    "manufacturer": "CCTV",
                    "model": "Multi-Camera",
                    "deviceResources": [
                        {
                            "name": "helmet_detection",
                            "description": "헬멧 착용 감지",
                            "attributes": {"dataType": "String"},
                            "properties": {"valueType": "String", "readWrite": "R"},
                        },
                        {
                            "name": "fall_detection",
                            "description": "낙상 감지",
                            "attributes": {"dataType": "String"},
                            "properties": {"valueType": "String", "readWrite": "R"},
                        },
                        {
                            "name": "person_detection",
                            "description": "사람 감지",
                            "attributes": {"dataType": "String"},
                            "properties": {"valueType": "String", "readWrite": "R"},
                        },
                    ],
                },
            }
            result = await self._post_with_versioned_fallback(
                self.metadata_url, "deviceprofile", profile_payload, "Device Profile 생성",
            )
            return result in {"success", "exists"}
        except Exception as exc:
            logger.error("Profile 생성 오류: %s", exc)
            return False
