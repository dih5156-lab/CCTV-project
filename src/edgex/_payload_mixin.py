"""
EdgeX 페이로드 빌더 믹스인

EdgeX 이벤트 페이로드 구성, 타임스탬프 변환, 이벤트 필드 추출 등
데이터 변환 로직을 담당한다.
각 메서드는 self.PROFILE_NAME, self.service_name 에 의존한다.
"""

import json
import logging
import time
import uuid
from typing import Dict, Optional
from urllib.parse import urlparse

from ..time_utils import coerce_timestamp_seconds, now_kst_iso

logger = logging.getLogger(__name__)


class _PayloadMixin:
    """EdgeX 페이로드 구성 및 데이터 변환 유틸리티 믹스인."""

    # ── 범용 유틸리티 ────────────────────────────────────────────────────────

    @staticmethod
    def _to_bool(value: object) -> bool:
        """다양한 입력을 bool로 안전하게 변환."""
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

    def _parse_rtsp_address_port(self, rtsp_source: str) -> Dict[str, str]:
        """RTSP URL 에서 Address/Port 를 안정적으로 추출."""
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

    # ── 이벤트 타입 → 리소스 이름 매핑 ──────────────────────────────────────

    def _map_event_type_to_resource(self, event_type: str) -> str:
        """이벤트 유형을 EdgeX 리소스 이름으로 매핑."""
        if event_type in ["helmet", "head", "unsafe_behavior", "wearing_helmet"]:
            return "helmet_detection"
        if event_type in ["fall_detected", "not_fall"]:
            return "fall_detection"
        if event_type == "person":
            return "person_detection"
        return "helmet_detection"

    # ── 타임스탬프 변환 ───────────────────────────────────────────────────────

    def _normalize_timestamp(self, timestamp: object) -> str:
        """타임스탬프를 ISO 8601 문자열로 정규화 (유연한 입력 지원)."""
        if timestamp is None:
            return now_kst_iso()
        if isinstance(timestamp, str):
            normalized = timestamp.strip()
            return normalized if normalized else now_kst_iso()
        return str(timestamp)

    def _to_origin_nanos(self, timestamp: object) -> int:
        """타임스탬프를 나노초 단위의 정수로 변환 (유연한 입력 지원)."""
        seconds = coerce_timestamp_seconds(timestamp, fallback=time.time())
        return int(seconds * 1_000_000_000)

    # ── 이벤트 필드 추출 ─────────────────────────────────────────────────────

    def _extract_event_fields(self, event: object) -> Dict[str, object]:
        """이벤트 객체에서 필드 추출 (dict / dataclass 양쪽 지원)."""
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
            event_type = (
                str(event_type_attr.value)
                if hasattr(event_type_attr, "value")
                else str(event_type_attr)
            )
            confidence = event.confidence if hasattr(event, "confidence") else 0.0
            x = event.x if hasattr(event, "x") else 0
            y = event.y if hasattr(event, "y") else 0
            width = event.width if hasattr(event, "width") else 0
            height = event.height if hasattr(event, "height") else 0
            object_id = event.object_id if hasattr(event, "object_id") else None
            timestamp = self._normalize_timestamp(
                event.timestamp if hasattr(event, "timestamp") else None
            )

        return {
            "event_type": event_type,
            "confidence": confidence,
            "x": x, "y": y, "width": width, "height": height,
            "object_id": object_id,
            "timestamp": timestamp,
        }

    # ── 페이로드 빌더 ────────────────────────────────────────────────────────

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
        """감지 이벤트의 모든 페이로드 컴포넌트를 한 번에 조립."""
        normalized_timestamp = self._normalize_timestamp(timestamp)
        origin = self._to_origin_nanos(normalized_timestamp)
        value_payload = self._build_value_payload(
            event_type, confidence, x, y, width, height,
            object_id, normalized_timestamp, device_name, resource_name,
        )
        event_payload = self._build_event_payload(
            device_name, resource_name, origin, value_payload,
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
    ) -> Dict[str, object]:
        """EdgeX 이벤트 페이로드의 value 필드 구성."""
        return {
            "type": event_type,
            "device": device_name,
            "resource": resource_name,
            "confidence": confidence,
            "bbox": {"x": x, "y": y, "width": width, "height": height},
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
    ) -> Dict[str, object]:
        """EdgeX 이벤트 페이로드 구성 (readings 포함)."""
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

    def _build_envelope(
        self, event_payload: Dict[str, object]
    ) -> Dict[str, object]:
        """EdgeX 이벤트 페이로드를 감싸는 envelope 구성."""
        return {
            "apiVersion": "v3",
            "receivedTopic": "",
            "correlationID": str(uuid.uuid4()),
            "requestID": event_payload.get("requestId", ""),
            "errorCode": 0,
            "payload": event_payload,
            "contentType": "application/json",
        }
