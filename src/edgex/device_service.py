"""
CCTV용 EdgeX 디바이스 서비스
CCTV 카메라를 EdgeX Foundry 장치로 관리
"""

import asyncio
import base64
import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import redis as redis_module
from urllib.parse import urlparse

import paho.mqtt.client as mqtt
import requests

try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Redis와 MQTT 연결 시도 간 최소 간격을 보장하기 위한 락과 타임스탬프
_redis_lock = threading.Lock()
_mqtt_lock = threading.Lock()


class CCTVDeviceService:
    """EdgeX CCTV 장치 서비스.

    CCTV 카메라를 EdgeX Foundry 장치로 관리하며, 탐지 이벤트를
    Redis Message Bus 또는 MQTT 채널로 발행한다.
    연결 실패 시 SQLite Outbox에 이벤트를 보관하고 복구 후 재전송한다.

    Attributes:
        metadata_url: EdgeX Core Metadata 서비스 URL
        data_url:     EdgeX Core Data 서비스 URL
        service_name: EdgeX 디바이스 서비스 이름
    """
    _redis_lock = _redis_lock
    _mqtt_lock = _mqtt_lock
    _redis_base_cooldown_sec: float = 5
    _mqtt_base_cooldown_sec: float = 5
    _max_cooldown_sec: float = 60    # 대기 상한선

    PROFILE_NAME = "CCTV-Camera-Profile"
    
    def __init__(self, config: Dict):
        """
        매개변수:
            config: {
                "coreMetadataUrl": "http://localhost:59881",
                "coreDataUrl": "http://localhost:59880",
                "deviceServiceName": "cctv-device-service",
                "baseUrl": "http://localhost:59999"
            }
        """
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
        self.outbox_db_path = Path(config.get("outboxDbPath", "data/edgex_outbox.db"))
        self.outbox_flush_batch_size = int(config.get("outboxFlushBatchSize", 100))
        self._mqtt_client: Optional[mqtt.Client] = None
        self._redis_client: Optional["redis_module.Redis"] = None
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

    # ── 데이터 카테고리 상수 ─────────────────────────────────
    # person : 사람 감지 이벤트 (헬멧, 낙상, 얼굴 인식 등)
    # camera : 카메라 단위 이벤트 (침입, 위험구역, 기타)
    # sensor : IoT 센서 데이터 (parser-python 쪽에서 발생)
    _PERSON_EVENT_TYPES = frozenset({
        "helmet", "head", "unsafe_behavior", "wearing_helmet",
        "fall_detected", "not_fall",
        "face_recognized", "face_unknown", "person",
    })
    _CAMERA_EVENT_TYPES = frozenset({
        "danger_zone", "intrusion", "other",
    })

    def _init_outbox(self) -> None:
        """Prepare the local SQLite outbox used for store-and-forward delivery."""
        if not self.enable_store_and_forward:
            return

        with self._outbox_connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    data_category TEXT NOT NULL DEFAULT 'camera',
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_attempt_at TEXT,
                    sent_at TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    last_error TEXT
                )
                """
            )
            # 기존 DB에 컬럼이 없으면 마이그레이션
            existing = {row[1] for row in conn.execute("PRAGMA table_info(detection_outbox)")}
            if "data_category" not in existing:
                conn.execute(
                    "ALTER TABLE detection_outbox "
                    "ADD COLUMN data_category TEXT NOT NULL DEFAULT 'camera'"
                )
                logger.info("detection_outbox: data_category 컬럼 추가 완료 (마이그레이션)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_detection_outbox_status_id "
                "ON detection_outbox(status, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_detection_outbox_category "
                "ON detection_outbox(data_category, status)"
            )
            conn.commit()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _outbox_connect(self):
        """부모 디렉토리를 보장한 뒤 SQLite 연결을 반환하는 컨텍스트 매니저."""
        self.outbox_db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.outbox_db_path)

    def _classify_event_category(self, event_type: str) -> str:
        """이벤트 타입으로 data_category 분류.

        Returns:
            'person'  — 사람 감지 이벤트 (헬멧/낙상/얼굴 인식 등)
            'camera'  — 카메라 단위 이벤트 (침입/위험구역 등)
        """
        normalized = (event_type or "").lower().strip()
        if normalized in self._PERSON_EVENT_TYPES:
            return "person"
        if normalized in self._CAMERA_EVENT_TYPES:
            return "camera"
        # 알 수 없는 타입은 카메라 이벤트로 처리
        return "camera"

    def _store_failed_detection_event(
        self,
        camera_id: str,
        event_data: Dict[str, Any],
        last_error: str,
    ) -> None:
        """Persist a failed event so Jetson can resend it after EdgeX/server recovery."""
        if not self.enable_store_and_forward:
            return

        event_type = ""
        if isinstance(event_data, dict):
            event_type = str(event_data.get("type") or event_data.get("event_type") or "")
        category = self._classify_event_category(event_type)

        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            conn.execute(
                """
                INSERT INTO detection_outbox (
                    camera_id, data_category, payload_json,
                    created_at, last_attempt_at, retry_count, status, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    category,
                    json.dumps(event_data, ensure_ascii=False),
                    now,
                    now,
                    1,
                    "pending",
                    last_error[:1000],
                ),
            )
            conn.commit()
            logger.debug(
                "[Outbox] 저장: camera=%s category=%s type=%s",
                camera_id, category, event_type,
            )

    def _store_pending_event(
        self,
        camera_id: str,
        event_data: Dict[str, Any],
    ) -> Optional[int]:
        """모든 이벤트를 pending으로 먼저 저장하고 row_id 반환."""
        if not self.enable_store_and_forward:
            return None

        event_type = ""
        if isinstance(event_data, dict):
            event_type = str(event_data.get("type") or event_data.get("event_type") or "")
        category = self._classify_event_category(event_type)

        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            cur = conn.execute(
                """
                INSERT INTO detection_outbox (
                    camera_id, data_category, payload_json,
                    created_at, last_attempt_at, retry_count, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    category,
                    json.dumps(event_data, ensure_ascii=False),
                    now,
                    now,
                    0,
                    "pending",
                ),
            )
            conn.commit()
            logger.debug(
                "[Outbox] pending 저장: camera=%s category=%s type=%s id=%s",
                camera_id, category, event_type, cur.lastrowid,
            )
            return cur.lastrowid

    def get_pending_detection_events(
        self,
        limit: Optional[int] = None,
        data_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return pending outbox rows in FIFO order for replay.

        Args:
            limit: 최대 반환 행 수 (기본: outbox_flush_batch_size)
            data_category: 'person' | 'camera' | None(전체)
        """
        if not self.enable_store_and_forward:
            return []

        fetch_limit = int(limit or self.outbox_flush_batch_size)
        with self._outbox_lock, self._outbox_connect() as conn:
            conn.row_factory = sqlite3.Row
            if data_category:
                rows = conn.execute(
                    """
                    SELECT id, camera_id, data_category, payload_json,
                           created_at, last_attempt_at, retry_count, status, last_error
                    FROM detection_outbox
                    WHERE status = 'pending' AND data_category = ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (data_category, fetch_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, camera_id, data_category, payload_json,
                           created_at, last_attempt_at, retry_count, status, last_error
                    FROM detection_outbox
                    WHERE status = 'pending'
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (fetch_limit,),
                ).fetchall()

        pending: List[Dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {}
            pending.append(
                {
                    "id": row["id"],
                    "camera_id": row["camera_id"],
                    "data_category": row["data_category"],
                    "event_data": payload,
                    "created_at": row["created_at"],
                    "last_attempt_at": row["last_attempt_at"],
                    "retry_count": row["retry_count"],
                    "status": row["status"],
                    "last_error": row["last_error"],
                }
            )
        return pending

    def _mark_outbox_sent(self, outbox_id: int) -> None:
        if not self.enable_store_and_forward:
            return
        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            conn.execute(
                """
                UPDATE detection_outbox
                SET status = 'sent', sent_at = ?, last_attempt_at = ?
                WHERE id = ?
                """,
                (now, now, outbox_id),
            )
            conn.commit()

    def _mark_outbox_retry_failed(self, outbox_id: int, last_error: str) -> None:
        if not self.enable_store_and_forward:
            return
        with self._outbox_lock, self._outbox_connect() as conn:
            conn.execute(
                """
                UPDATE detection_outbox
                SET retry_count = retry_count + 1,
                    last_attempt_at = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (self._utc_now_iso(), last_error[:1000], outbox_id),
            )
            conn.commit()

    @staticmethod
    def _to_bool(value: object) -> bool:
        """다양한 입력을 bool로 안전하게 변환"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "y"}:
                return True
            if normalized in {"0", "false", "no", "off", "n", ""}:
                return False
        return bool(value)

    def _describe_http_status(self, status_code: int) -> str:
        """HTTP 상태 코드에 대한 설명 반환"""
        status_map = {
            200: "성공: 요청 처리 완료",
            201: "생성됨: 리소스 생성 성공",
            202: "수락됨: 요청 수락, 처리 중",
            204: "본문 없음: 성공적으로 처리됨",
            207: "복합 상태: 부분 성공/실패 혼재",
            400: "잘못된 요청: 데이터 형식 오류 또는 잘못된 데이터",
            401: "인증 실패: JWT 토큰 누락 또는 유효하지 않음",
            403: "권한 없음: 접근 권한 없음",
            404: "리소스 없음: 요청한 리소스를 찾을 수 없음",
            405: "허용되지 않은 메서드",
            408: "요청 시간 초과",
            409: "충돌: 리소스 충돌(이미 존재 등)",
            415: "지원되지 않는 콘텐츠 타입",
            422: "처리 불가 엔터티: 검증 실패",
            423: "잠금됨: 디바이스 잠금 또는 운영 상태 비활성화",
            429: "요청 과다",
            500: "내부 서버 오류",
            502: "게이트웨이 오류",
            503: "서비스 사용 불가 또는 연결 제한",
            504: "게이트웨이 시간 초과",
        }
        return status_map.get(status_code, "알 수 없는 오류")

    def _versioned_endpoints(
        self,
        base_url: str,
        resource_path: str,
        include_legacy: bool = False,
    ) -> List[str]: # 버전별 엔드포인트 목록 생성 (v3 → v2 → v1 → 레거시)
        base = base_url.rstrip("/")
        resource = resource_path.lstrip("/")
        endpoints = [
            f"{base}/api/v3/{resource}",
            f"{base}/api/v2/{resource}",
            f"{base}/api/v1/{resource}",
        ]
        if include_legacy:
            endpoints.append(f"{base}/{resource}")
        return endpoints

    def _payload_for_endpoint(
        self, endpoint: str, payload: Dict[str, object]
    ) -> object:
        """엔드포인트 버전에 따라 페이로드 형식 조정 (v3=배열, v2/v1=객체)."""
        return [payload] if "/v3/" in endpoint else payload

    def _extract_multistatus_item(
        self, response: requests.Response
    ) -> Optional[Dict[str, object]]:
        """207 복합 상태 응답에서 첫 번째 항목 추출."""
        try:
            result = response.json()
        except ValueError:
            return None

        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            return result[0]
        return None

    def _response_status_code(self, response: requests.Response) -> int:
        """응답에서 실제 상태 코드 추출 (207 복합 상태 지원)."""
        if response.status_code != 207:
            return response.status_code

        item = self._extract_multistatus_item(response)
        if item and isinstance(item.get("statusCode"), int):
            return item["statusCode"]
        return response.status_code

    def _response_id(self, response: requests.Response) -> Optional[str]:
        """응답에서 생성된 리소스 ID 추출 (유연한 형식 지원)."""
        if response.status_code == 207:
            item = self._extract_multistatus_item(response)
            if item:
                value = item.get("id")
                return str(value) if value is not None else None

        try:
            data = response.json()
            if isinstance(data, dict):
                value = data.get("id")
                return str(value) if value is not None else None
        except ValueError:
            return None
        return None

    async def _request_get(self, endpoint: str, timeout: int = 5) -> Optional[requests.Response]:
        try:
            return await asyncio.to_thread(requests.get, endpoint, timeout=timeout)
        except requests.RequestException as error:
            logger.debug("GET 실패 (%s): %s", endpoint, error)
            return None

    async def _request_post(
        self,
        endpoint: str,
        payload: object,
        timeout: int = 10,
    ) -> Optional[requests.Response]:
        try:
            return await asyncio.to_thread(
                requests.post,
                endpoint,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.RequestException as error:
            logger.debug("POST 실패 (%s): %s", endpoint, error)
            return None

    async def _request_delete(self, endpoint: str, timeout: int = 10) -> Optional[requests.Response]:
        try:
            return await asyncio.to_thread(requests.delete, endpoint, timeout=timeout)
        except requests.RequestException as error:
            logger.debug("DELETE 실패 (%s): %s", endpoint, error)
            return None

    async def _post_with_versioned_fallback(
        self,
        base_url: str,
        resource_path: str,
        payload: Dict[str, object],
        operation_name: str,
    ) -> str:
        endpoints = self._versioned_endpoints(base_url, resource_path)

        for endpoint in endpoints:
            request_payload = self._payload_for_endpoint(endpoint, payload)
            response = await self._request_post(endpoint, request_payload, timeout=10)
            if response is None:
                logger.debug("%s 실패: 응답 없음 (%s)", operation_name, endpoint)
                continue

            status_code = self._response_status_code(response)
            if status_code in [200, 201]:
                logger.info("✓ %s: %s", operation_name, endpoint)
                return "success"
            if status_code == 409:
                logger.info("✓ %s 이미 존재: %s", operation_name, endpoint)
                return "exists"
            if status_code == 404:
                logger.debug("엔드포인트 없음: %s", endpoint)
                continue
            if response.status_code == 207:
                logger.warning("%s 실패: 207 응답 - %s", operation_name, response.text)
                continue

            logger.warning(
                "%s 실패: %s - %s", operation_name, status_code, self._describe_http_status(status_code)
            )
            logger.debug("응답: %s", response.text)

        return "failed"

    async def _probe_service_health(self, base_url: str, service_name: str) -> bool:
        endpoints = self._versioned_endpoints(
            base_url,
            "ping",
            include_legacy=True,
        )

        for endpoint in endpoints:
            response = await self._request_get(endpoint, timeout=5)
            if response and response.status_code == 200:
                logger.info("✓ EdgeX %s 연결됨 (%s)", service_name, endpoint)
                return True

        logger.warning("EdgeX %s 연결 실패 - 시도한 엔드포인트:", service_name)
        for endpoint in endpoints:
            logger.warning("  - %s", endpoint)
        return False

    async def _post_event_via_rest(
        self,
        camera_id: str,
        event_type: str,
        base_event: Dict[str, object],
    ) -> bool:
        endpoints = self._versioned_endpoints(self.data_url, "event")

        last_status = None
        last_text = None
        last_endpoint = None

        for endpoint in endpoints:
            last_endpoint = endpoint
            api_version = "v3" if "/v3/" in endpoint else ("v2" if "/v2/" in endpoint else "v1")
            event_data = {"apiVersion": api_version, **base_event}
            payload = [event_data] if api_version == "v3" else event_data

            response = await self._request_post(endpoint, payload, timeout=10)
            if response is None:
                continue

            status_code = self._response_status_code(response)
            last_status = status_code
            last_text = response.text

            if status_code in [200, 201]:
                logger.debug("[%s] EdgeX 이벤트 전송: %s", camera_id, event_type)
                return True
            if status_code == 404:
                logger.warning("엔드포인트 없음: %s", endpoint)
                continue
            if response.status_code == 207:
                logger.warning("Event 전송 실패 (%s): 207 응답 - %s", camera_id, response.text)
                continue

            logger.warning(
                "Event 전송 실패 (%s): %s - %s",
                camera_id, status_code, self._describe_http_status(status_code),
            )
            logger.warning("응답: %s", response.text)
            logger.warning("엔드포인트: %s", endpoint)

        logger.warning("이벤트 전송 실패 (%s) - 모든 엔드포인트 시도 완료", camera_id)
        if last_endpoint:
            logger.warning("마지막 엔드포인트: %s", last_endpoint)
        if last_status is not None:
            logger.warning(
                "마지막 상태 코드: %s - %s", last_status, self._describe_http_status(last_status)
            )
        if last_text:
            logger.warning("마지막 응답: %s", last_text)

        return False

    async def _get_entity_by_name(
        self,
        resource_path: str,
        container_key: str,
    ) -> Optional[Dict[str, object]]: # 이름으로 EdgeX 엔터티 조회 (성공 시 반환)
        endpoints = self._versioned_endpoints(self.metadata_url, resource_path)

        for endpoint in endpoints:
            response = await self._request_get(endpoint, timeout=5)
            if response is None:
                logger.debug("엔터티 조회 응답 없음: %s", endpoint)
                continue
            try:
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        nested = data.get(container_key)
                        if isinstance(nested, dict):
                            return nested
                        return data
                if response.status_code == 404:
                    continue
            except Exception:
                continue

        return None

    async def _delete_entity_by_name(
        self,
        resource_path: str,
        success_log: Optional[str] = None,
    ) -> bool: # EdgeX 장치를 이름으로 삭제 (성공 시 로그 기록)
        endpoints = self._versioned_endpoints(self.metadata_url, resource_path)

        for endpoint in endpoints:
            response = await self._request_delete(endpoint, timeout=10)
            if response is None:
                logger.debug("엔터티 삭제 응답 없음: %s", endpoint)
                continue
            try:
                if response.status_code in [200, 202, 204]:
                    if success_log:
                        logger.info(success_log)
                    return True
                if response.status_code == 404:
                    return True
            except Exception:
                continue

        return False

    def _map_event_type_to_resource(self, event_type: str) -> str: # 이벤트 유형을 EdgeX 리소스 이름으로 매핑
        if event_type in ["helmet", "head", "unsafe_behavior", "wearing_helmet"]:
            return "helmet_detection"
        if event_type in ["fall_detected", "not_fall"]:
            return "fall_detection"
        if event_type == "person":
            return "person_detection"
        return "helmet_detection"

    def _parse_rtsp_address_port(self, rtsp_source: str) -> Dict[str, str]:
        """RTSP URL에서 Address/Port를 안정적으로 추출"""
        default = {"Address": "localhost", "Port": "554"}
        if not isinstance(rtsp_source, str):
            return default

        source = rtsp_source.strip()
        if not source:
            return default

        try:
            parsed = urlparse(source if "://" in source else f"rtsp://{source}")
            if parsed.hostname:
                return {
                    "Address": parsed.hostname,
                    "Port": str(parsed.port or 554),
                }
        except Exception as exc:
            logger.debug("RTSP URL urlparse 실패, 수동 파싱으로 대체: %s", exc)

        host = source.split("://", 1)[-1].split("/", 1)[0]
        if ":" in host:
            parts = host.rsplit(":", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return {"Address": parts[0], "Port": parts[1]}
        return {"Address": host or "localhost", "Port": "554"}

    def _normalize_timestamp(self, timestamp: object) -> str: # 타임스탬프를 ISO 8601 문자열로 정규화 (유연한 입력 지원)
        now = lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if timestamp is None:
            return now()
        if isinstance(timestamp, str):
            normalized = timestamp.strip()
            return normalized if normalized else now()
        return str(timestamp)

    def _to_origin_nanos(self, timestamp: object) -> int: # 타임스탬프를 나노초 단위의 정수로 변환 (유연한 입력 지원)
        if isinstance(timestamp, (int, float)):
            return int(float(timestamp) * 1_000_000_000)

        if isinstance(timestamp, str):
            value = timestamp.strip()
            if value:
                try:
                    return int(float(value) * 1_000_000_000)
                except Exception as exc:
                    logger.debug("타임스탬프 float 변환 실패, ISO 파싱 시도: %s", exc)
                try:
                    normalized = value.replace("Z", "+00:00")
                    return int(datetime.fromisoformat(normalized).timestamp() * 1_000_000_000)
                except Exception as exc:
                    logger.debug("타임스탬프 ISO 파싱 실패, 현재 시각 사용: %s", exc)

        return int(time.time() * 1_000_000_000)

    def _extract_event_fields(self, event: object) -> Dict[str, object]: # 이벤트 객체에서 필드 추출 (유연한 구조 지원)
        if isinstance(event, dict):
            raw_event_type = event.get("type", "unknown")
            event_type = str(raw_event_type) if raw_event_type is not None else "unknown"
            confidence = event.get("confidence", 0.0)
            bbox = event.get("bbox", {}) or {}
            if not isinstance(bbox, dict):
                bbox = {}
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            width = bbox.get("width", 0)
            height = bbox.get("height", 0)
            object_id = event.get("object_id")
            timestamp = self._normalize_timestamp(event.get("timestamp"))
        else:
            event_type_attr = event.event_type if hasattr(event, "event_type") else "unknown"
            if hasattr(event_type_attr, "value"):
                event_type = str(event_type_attr.value)
            else:
                event_type = str(event_type_attr)
            confidence = event.confidence if hasattr(event, "confidence") else 0.0
            x = event.x if hasattr(event, "x") else 0
            y = event.y if hasattr(event, "y") else 0
            width = event.width if hasattr(event, "width") else 0
            height = event.height if hasattr(event, "height") else 0
            object_id = event.object_id if hasattr(event, "object_id") else None
            timestamp = self._normalize_timestamp(event.timestamp if hasattr(event, "timestamp") else None)

        return {
            "event_type": event_type,
            "confidence": confidence,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "object_id": object_id,
            "timestamp": timestamp,
        }

    def _build_detection_payload_bundle(
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
        timestamp: object,
    ) -> Dict[str, object]:
        normalized_timestamp = self._normalize_timestamp(timestamp)
        origin = self._to_origin_nanos(normalized_timestamp)
        value_payload = self._build_value_payload(
            event_type,
            confidence,
            x,
            y,
            width,
            height,
            object_id,
            normalized_timestamp,
            device_name,
            resource_name,
        )
        event_payload = self._build_event_payload(
            device_name,
            resource_name,
            origin,
            value_payload,
        )
        return {
            "timestamp": normalized_timestamp,
            "origin": origin,
            "value_payload": value_payload,
            "event_payload": event_payload,
            "envelope": self._build_envelope(event_payload),
        }

    def _build_value_payload(
        self,
        event_type: str,
        confidence: float,
        x: int,
        y: int,
        width: int,
        height: int,
        object_id: Optional[int],
        timestamp: str,
        device_name: str,
        resource_name: str,
    ) -> Dict[str, object]: # 이벤트 필드로 EdgeX 이벤트 페이로드의 value 필드 구성
        return {
            "type": event_type,
            "device": device_name,
            "resource": resource_name,
            "confidence": confidence,
            "bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
            "object_id": object_id,
            "timestamp": timestamp,
            "metadata": {
                "profile": self.PROFILE_NAME,
                "service": self.service_name,
                "version": "v1",
            },
        }

    def _build_event_payload(
        self,
        device_name: str,
        resource_name: str,
        origin: int,
        value_payload: Dict[str, object],
    ) -> Dict[str, object]: # EdgeX 이벤트 페이로드 구성 (유연한 구조 지원)
        event_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        return {
            "apiVersion": "v3",
            "requestId": request_id,
            "event": {
                "apiVersion": "v3",
                "id": event_id,
                "deviceName": device_name,
                "profileName": self.PROFILE_NAME,
                "sourceName": resource_name,
                "origin": origin,
                "readings": [
                    {
                        "origin": origin,
                        "deviceName": device_name,
                        "resourceName": resource_name,
                        "profileName": self.PROFILE_NAME,
                        "valueType": "String",
                        "value": json.dumps(value_payload, ensure_ascii=False),
                    }
                ],
            },
        }

    def _build_envelope(self, event_payload: Dict[str, object]) -> Dict[str, object]: # EdgeX 이벤트 페이로드를 감싸는 엔벨로프 구성
        return {
            "apiVersion": "v3",
            "receivedTopic": "",
            "correlationID": str(uuid.uuid4()),
            "requestID": event_payload.get("requestId", ""),
            "errorCode": 0,
            "payload": event_payload,
            "contentType": "application/json",
        }
    
    async def initialize(self):
        """EdgeX 연결 확인 (비동기 호환)"""
        try:
            self._init_outbox()
            await self._probe_service_health(self.metadata_url, "Core Metadata")
            await self._probe_service_health(self.data_url, "Core Data")
                    
        except Exception as exc:
            logger.error("EdgeX 연결 오류: %s", exc)
    
    async def add_camera(self, camera_id: str, rtsp_source: str) -> Optional[str]: # 카메라를 EdgeX 장치로 등록 (비동기 메서드)
        """
        카메라를 EdgeX 장치로 등록
        
        매개변수:
            camera_id: 카메라 ID (예: "camera_1")
            rtsp_source: RTSP URL (예: "rtsp://192.168.1.100:554/stream")
            
        반환값:
            device_id 또는 None
        """
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
                deleted = await self._delete_device_by_name(device_name)
                if not deleted:
                    logger.error("기존 디바이스 삭제 실패로 재등록 중단: %s", device_name)
                    return None
                await asyncio.sleep(0.3)

            rtsp_conn = self._parse_rtsp_address_port(rtsp_source)
            
            # Device 생성 페이로드
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
                            "URL": rtsp_source
                        }
                    },
                    "labels": [
                        "cctv",
                        f"camera_{camera_id}"
                    ]
                }
            }
            
            # Device 등록 (v3 → v2 → v1 폴백)
            endpoints = self._versioned_endpoints(self.metadata_url, "device")
            
            for endpoint in endpoints:
                try:
                    # EdgeX v3는 배열 형식 필요
                    payload = self._payload_for_endpoint(endpoint, device_payload)
                    response = await self._request_post(endpoint, payload, timeout=10)
                    if response is None:
                        continue
                    status_code = self._response_status_code(response)
                    
                    if status_code in [200, 201]:
                        device_id = self._response_id(response) or device_name
                        self.devices[camera_id] = device_id
                        logger.info("✓ 카메라 등록 성공: %s -> %s (ID: %s)", camera_id, device_name, device_id)
                        logger.debug("  RTSP: %s", rtsp_source)
                        logger.debug("  엔드포인트: %s", endpoint)
                        return device_id
                    elif status_code == 404:
                        logger.debug("엔드포인트 없음: %s", endpoint)
                        continue
                    elif status_code == 409:
                        existing = await self._get_device_by_name(device_name)
                        existing_service = (existing or {}).get("serviceName", "")
                        if existing_service == self.service_name:
                            self.devices[camera_id] = existing.get("id") if existing else device_name
                            logger.info("✓ 카메라 이미 존재: %s -> %s", camera_id, device_name)
                            logger.debug("  RTSP: %s", rtsp_source)
                            logger.debug("  엔드포인트: %s", endpoint)
                            return self.devices[camera_id]

                        logger.warning(
                            "기존 디바이스 서비스 불일치(충돌 응답): %s (%s -> %s)",
                            device_name, existing_service, self.service_name,
                        )
                        deleted = await self._delete_device_by_name(device_name)
                        if deleted:
                            await asyncio.sleep(0.3)
                            continue

                        logger.error("기존 디바이스 삭제 실패로 재등록 중단: %s", device_name)
                        return None
                    elif response.status_code == 207:
                        logger.warning("Device 등록 실패 (%s): 207 응답 - %s", camera_id, response.text)
                        continue
                    else:
                        logger.warning(
                            "Device 등록 실패 (%s): %s - %s",
                            camera_id, status_code, self._describe_http_status(status_code),
                        )
                        logger.warning("응답 내용: %s", response.text)
                        logger.warning("엔드포인트: %s", endpoint)
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

        # 전송 전에 outbox에 pending으로 먼저 저장 (replay=False인 경우만)
        outbox_row_id = self._store_pending_event(camera_id, event_data) if persist_on_failure else None

        device_name = f"camera-{camera_id}"
        event_fields = self._extract_event_fields(event_data)
        event_type = event_fields["event_type"]
        confidence = event_fields["confidence"]
        x = event_fields["x"]
        y = event_fields["y"]
        width = event_fields["width"]
        height = event_fields["height"]
        object_id = event_fields["object_id"]
        timestamp = event_fields["timestamp"]
        resource_name = self._map_event_type_to_resource(event_type)

        if self.message_bus_type == "redis":
            redis_ok = await asyncio.to_thread(
                self._publish_event_redis,
                device_name,
                resource_name,
                event_type,
                confidence,
                x,
                y,
                width,
                height,
                object_id,
                timestamp,
            )
            if redis_ok:
                logger.info("✓[%s] Redis 이벤트 전송: %s", camera_id, event_type)
                self._mark_outbox_sent(outbox_row_id)
                return True

        mqtt_ok = await asyncio.to_thread(
            self._publish_event_mqtt,
            device_name,
            resource_name,
            event_type,
            confidence,
            x,
            y,
            width,
            height,
            object_id,
            timestamp,
        )
        if mqtt_ok:
            logger.info("✓[%s] MQTT 이벤트 전송: %s", camera_id, event_type)
            self._mark_outbox_sent(outbox_row_id)
            return True

        if self.enable_rest_event_post:
            bundle = self._build_detection_payload_bundle(
                device_name,
                resource_name,
                event_type,
                confidence,
                x,
                y,
                width,
                height,
                object_id,
                timestamp,
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
        # outbox_row_id가 없는 경우(persist_on_failure=False)에만 _store_failed_detection_event 호출
        if persist_on_failure and outbox_row_id is None:
            self._store_failed_detection_event(camera_id, event_data, error_message)
        return False

    async def replay_detection_event(
        self,
        outbox_id: int,
        camera_id: str,
        event_data: Dict[str, Any],
    ) -> bool:
        """저장된 outbox 이벤트를 EdgeX로 다시 전송한다."""
        try:
            sent = await self._send_detection_event_payload(
                camera_id,
                event_data,
                persist_on_failure=False,
            )
            if sent:
                self._mark_outbox_sent(outbox_id)
                return True

            self._mark_outbox_retry_failed(outbox_id, "replay failed")
            return False
        except Exception as exc:
            self._mark_outbox_retry_failed(outbox_id, str(exc))
            logger.error("Outbox replay 오류 (%s): %s", outbox_id, exc)
            return False

    async def send_detection_event(self, camera_id: str, events: List) -> bool: # 감지 이벤트를 EdgeX Event로 전송 (비동기 메서드)
        """
        감지 이벤트를 EdgeX Event로 전송
        
        매개변수:
            camera_id: 카메라 ID
            events: DetectionEvent 리스트
            
        반환값:
            전송 성공 여부
        """
        try:
            all_sent = True
            for event in events:
                sent = await self._send_detection_event_payload(camera_id, event)
                all_sent = all_sent and sent

            return all_sent

        except Exception as exc:
            logger.error("이벤트 전송 오류 (%s): %s", camera_id, exc)
            return False

    # Redis 클라이언트를 초기화하고 연결을 확인합니다 (지수 백오프 재시도).
    def _ensure_redis_client(self) -> bool:
        with CCTVDeviceService._redis_lock:
            if self._redis_client:
                return True
            now = time.time()
            cooldown = min(
                CCTVDeviceService._redis_base_cooldown_sec
                * (2 ** self._redis_fail_count),
                CCTVDeviceService._max_cooldown_sec,
            )
            if now - self._redis_last_fail_time < cooldown:
                logger.debug(
                    "Redis 재연결 쿨다운 중 (%.1f초 대기, 실패 횟수=%d)",
                    cooldown - (now - self._redis_last_fail_time),
                    self._redis_fail_count,
                )
                return False
            try:
                client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=0,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                    decode_responses=True,
                )
                client.ping()
                self._redis_client = client
                self._redis_fail_count = 0   # 성공 시 초기화
                logger.info("✓ Redis 연결됨: %s:%s", self.redis_host, self.redis_port)
                return True
            except Exception as exc:
                self._redis_fail_count += 1
                self._redis_last_fail_time = now
                next_cooldown = min(
                    CCTVDeviceService._redis_base_cooldown_sec
                    * (2 ** self._redis_fail_count),
                    CCTVDeviceService._max_cooldown_sec,
                )
                logger.warning(
                    "Redis 연결 실패 (횟수=%d, 다음 재시도 %.0f초 후): %s",
                    self._redis_fail_count,
                    next_cooldown,
                    exc,
                )
                self._redis_client = None
                return False

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
    ) -> bool: # Redis를 통해 이벤트 발행 (EdgeX v3 형식의 envelope + payload 구조)
        if not self._ensure_redis_client():
            return False

        try:
            bundle = self._build_detection_payload_bundle(
                device_name,
                resource_name,
                event_type,
                confidence,
                x,
                y,
                width,
                height,
                object_id,
                timestamp,
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
                f"{topic_prefix}."
                f"{self.service_name}."
                f"{self.PROFILE_NAME}."
                f"{device_name}."
                f"{resource_name}"
            )
            publish_count = self._redis_client.publish(channel, json.dumps(envelope, ensure_ascii=False))

            if publish_count >= 0:
                logger.info("✓ Redis 발행 성공: %s (subscribers=%s)", channel, publish_count)
                return True

            logger.error("Redis 발행 실패: %s", channel)
            return False
        except Exception as exc:
            logger.error("Redis 전송 오류: %s", exc, exc_info=True)
            return False

    # MQTT 클라이언트를 초기화하고 연결을 확인합니다 (지수 백오프 재시도).
    def _ensure_mqtt_client(self) -> bool:
        with CCTVDeviceService._mqtt_lock:
            if self._mqtt_client:
                return True
            now = time.time()
            cooldown = min(
                CCTVDeviceService._mqtt_base_cooldown_sec
                * (2 ** self._mqtt_fail_count),
                CCTVDeviceService._max_cooldown_sec,
            )
            if now - self._mqtt_last_fail_time < cooldown:
                logger.debug(
                    "MQTT 재연결 쿨다운 중 (%.1f초 대기, 실패 횟수=%d)",
                    cooldown - (now - self._mqtt_last_fail_time),
                    self._mqtt_fail_count,
                )
                return False
            try:
                client = mqtt.Client()
                client.connect(self.mqtt_broker, self.mqtt_port, 60)
                client.loop_start()
                self._mqtt_client = client
                self._mqtt_fail_count = 0   # 성공 시 초기화
                logger.info("✓ MQTT 연결됨: %s:%d", self.mqtt_broker, self.mqtt_port)
                return True
            except Exception as exc:
                self._mqtt_fail_count += 1
                self._mqtt_last_fail_time = now
                next_cooldown = min(
                    CCTVDeviceService._mqtt_base_cooldown_sec
                    * (2 ** self._mqtt_fail_count),
                    CCTVDeviceService._max_cooldown_sec,
                )
                logger.warning(
                    "MQTT 연결 실패 (횟수=%d, 다음 재시도 %.0f초 후): %s",
                    self._mqtt_fail_count,
                    next_cooldown,
                    exc,
                )
                self._mqtt_client = None
                return False

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
        timestamp: str
    ) -> bool: # MQTT를 통해 이벤트 발행 (EdgeX v3 형식의 envelope + payload 구조)
        if not self._ensure_mqtt_client():
            return False

        try:
            logger.info("MQTT 이벤트 발행 시작: device=%s, resource=%s, type=%s", device_name, resource_name, event_type)

            bundle = self._build_detection_payload_bundle(
                device_name,
                resource_name,
                event_type,
                confidence,
                x,
                y,
                width,
                height,
                object_id,
                timestamp,
            )
            envelope = bundle["envelope"]

            topic = f"{self.mqtt_topic_prefix}/{self.service_name}/{device_name}/{resource_name}"
            logger.info("MQTT 토픽: %s", topic)

            result = self._mqtt_client.publish(topic, json.dumps(envelope), qos=0)
            
            if result.rc == 0:
                logger.info("✓ MQTT 발행 성공: %s (mid=%s)", topic, result.mid)
                return True
            else:
                logger.error("MQTT 발행 실패: %s (rc=%s)", topic, result.rc)
                return False
        except Exception as exc:
            logger.error("MQTT 전송 오류: %s", exc, exc_info=True)
            return False
    
    async def register_device_service(self) -> bool:
        """
        Device Service를 EdgeX에 등록
        """
        try:
            existing_service = await self._get_device_service_by_name(self.service_name)
            if existing_service:
                existing_base = existing_service.get("baseAddress", "")
                if existing_base and existing_base != self.base_url:
                    logger.warning(
                        f"기존 Device Service baseAddress 불일치: {existing_base} -> {self.base_url}"
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
                    "adminState": "UNLOCKED"
                }
            }
            
            result = await self._post_with_versioned_fallback(
                self.metadata_url,
                "deviceservice",
                service_payload,
                "Device Service 등록",
            )
            return result in {"success", "exists"}
            
        except Exception as exc:
            logger.error("Service 등록 오류: %s", exc)
            return False

    async def _get_device_service_by_name(self, service_name: str) -> Optional[Dict[str, object]]:
        # 이름으로 Device Service 조회
        return await self._get_entity_by_name(f"deviceservice/name/{service_name}", "service")

    async def _delete_device_service_by_name(self, service_name: str) -> bool: # 이름으로 Device Service 삭제
        return await self._delete_entity_by_name(f"deviceservice/name/{service_name}")
    
    async def create_device_profile(self) -> bool:
        """
        CCTV 장치 프로필 생성 (필요시)
        """
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
                            "properties": {
                                "valueType": "String",
                                "readWrite": "R"
                            }
                        },
                        {
                            "name": "fall_detection",
                            "description": "낙상 감지",
                            "attributes": {"dataType": "String"},
                            "properties": {
                                "valueType": "String",
                                "readWrite": "R"
                            }
                        },
                        {
                            "name": "person_detection",
                            "description": "사람 감지",
                            "attributes": {"dataType": "String"},
                            "properties": {
                                "valueType": "String",
                                "readWrite": "R"
                            }
                        }
                    ]
                }
            }
            
            result = await self._post_with_versioned_fallback(
                self.metadata_url,
                "deviceprofile",
                profile_payload,
                "Device Profile 생성",
            )
            return result in {"success", "exists"}
                
        except Exception as exc:
            logger.error("Profile 생성 오류: %s", exc)
            return False

    def publish_device_event(
        self,
        device_id: str,
        device_type: str,
        resource_name: str,
        event_data: Dict
    ) -> bool:
        """
        범용 디바이스 이벤트 발행 메서드
        
        다양한 디바이스 타입 (CCTV, 열화상, 센서 등)을 지원하는 통합 인터페이스
        
        매개변수:
            device_id: 디바이스 ID (예: camera-1, thermal-1, sensor-1)
            device_type: 디바이스 타입 (예: cctv, thermal, sensor)
            resource_name: 리소스명 (예: helmet_detection, temperature, motion)
            event_data: 이벤트 데이터 딕셔너리
                {
                    "type": "detection type",
                    "confidence": 0.95,
                    "value": "measurement value",
                    "bbox": {"x": 100, "y": 200, "width": 300, "height": 400},  # 선택사항
                    "object_id": 1,  # 선택사항
                    "timestamp": "2026-02-05T06:00:00Z"
                }
        
        반환값:
            발행 성공 여부
        """
        if not self._ensure_mqtt_client():
            return False

        try:
            logger.info("범용 디바이스 이벤트 발행: %s/%s", device_id, resource_name)
            
            timestamp_raw = event_data.get("timestamp")
            timestamp = self._normalize_timestamp(timestamp_raw)
            origin = self._to_origin_nanos(timestamp_raw if timestamp_raw is not None else timestamp)

            # 표준화된 메시지 포맷 (모든 디바이스 타입에 공통)
            payload_value = {
                "type": event_data.get("type", "unknown"),
                "device": device_id,
                "device_type": device_type,
                "resource": resource_name,
                "confidence": event_data.get("confidence", 0.0),
                "value": event_data.get("value"),
                "bbox": event_data.get("bbox"),  # 선택사항 (detection 타입만 해당)
                "object_id": event_data.get("object_id"),  # 선택사항
                "timestamp": timestamp,
                "metadata": {
                    "service": self.service_name,
                    "version": "v1",
                    "device_type": device_type
                }
            }
            event_payload = self._build_event_payload(device_id, resource_name, origin, payload_value)
            envelope = self._build_envelope(event_payload)

            # 확장성 있는 토픽 구조: edgex/events/device/{service}/{device_type}/{device_id}/{resource}
            topic = f"{self.mqtt_topic_prefix}/{self.service_name}/{device_type}/{device_id}/{resource_name}"
            logger.info("MQTT 토픽: %s", topic)

            result = self._mqtt_client.publish(topic, json.dumps(envelope, ensure_ascii=False), qos=0)
            
            if result.rc == 0:
                logger.info("✓ 범용 디바이스 이벤트 발행 성공: %s (mid=%s)", topic, result.mid)
                return True
            else:
                logger.error("범용 디바이스 이벤트 발행 실패: %s (rc=%s)", topic, result.rc)
                return False
        except Exception as exc:
            logger.error("범용 디바이스 이벤트 발행 오류: %s", exc, exc_info=True)
            return False

    def close(self) -> None:
        """열려 있는 메시지 버스 연결 정리"""
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
