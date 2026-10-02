"""외부 MQTT 입력을 내부 표준 이벤트로 정규화하는 서비스."""

from __future__ import annotations

import binascii
import json
import logging
import os
import signal
import sqlite3
import threading
import time
from base64 import b64decode, b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from ..canonical_event import build_canonical_event, build_event_id
from ..config import AppConfig, ExternalIngestConfig, MqttConfig
from ..event_ingest_store import EventIngestStore
from ..event_validation import EventValidationError, validate_event_identity
from ..protocols import MqttEventPublisher, MqttTopicSubscriber
from ..sensor_observation import build_sensor_observation
from ..time_utils import now_kst_iso

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return now_kst_iso()


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decode_base64_text(value: Any) -> Optional[str]:
    if not value:
        return None
    try:
        decoded = b64decode(str(value), validate=True)
    except (ValueError, binascii.Error):
        return None
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _extract_lora_timestamp(raw_payload: Dict[str, Any]) -> Any:
    rx_metadata = raw_payload.get("rx_metadata")
    if isinstance(rx_metadata, list) and rx_metadata:
        first_meta = rx_metadata[0] or {}
        if isinstance(first_meta, dict):
            for key in ("gw_recv_time", "time", "timestamp"):
                if key in first_meta:
                    return first_meta[key]
    return None


def _lora_spec(raw_payload: Dict[str, Any], gateway_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "message_id": raw_payload.get("message_id"),
        "app_eui": raw_payload.get("app_eui"),
        "dev_eui": raw_payload.get("dev_eui"),
        "f_port": raw_payload.get("f_port"),
        "f_cnt_up": raw_payload.get("f_cnt_up"),
        "is_confirmed": raw_payload.get("is_confirmed"),
        "gateway": gateway_info,
        "channel_plan": gateway_info.get("channel_plan"),
    }


def _lora_telemetry(first_rx_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "modulation": first_rx_meta.get("modulation"),
        "data_rate": first_rx_meta.get("data_rate"),
        "coding_rate": first_rx_meta.get("coding_rate"),
        "frequency": first_rx_meta.get("frequency"),
        "channel": first_rx_meta.get("channel"),
        "rssi": first_rx_meta.get("rssi"),
        "snr": first_rx_meta.get("snr"),
        "gateway_time": first_rx_meta.get("gw_recv_time"),
    }


def _canonical_gateway_fields(
    first_rx_meta: Dict[str, Any],
    gateway_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "gw_eui": gateway_info.get("gw_eui"),
        "latitude": gateway_info.get("latitude"),
        "longitude": gateway_info.get("longitude"),
        "altitude": gateway_info.get("altitude"),
        "channel_plan": gateway_info.get("channel_plan"),
        "modulation": first_rx_meta.get("modulation"),
        "data_rate": first_rx_meta.get("data_rate"),
        "coding_rate": first_rx_meta.get("coding_rate"),
        "channel": first_rx_meta.get("channel"),
        "frequency": first_rx_meta.get("frequency"),
        "rssi": first_rx_meta.get("rssi"),
        "snr": first_rx_meta.get("snr"),
        "received_at": first_rx_meta.get("gw_recv_time"),
    }


def normalize_external_event(
    raw_payload: Dict[str, Any], topic: str, *, received_at: Optional[str] = None
) -> Dict[str, Any]:
    """외부 입력 payload를 프로젝트 내부 이벤트 형식으로 정규화한다."""
    if not isinstance(raw_payload, dict):
        raise EventValidationError([{"field": "payload", "code": "mapping_required"}])
    for field in ("decoded", "telemetry"):
        if (
            field in raw_payload
            and raw_payload[field] is not None
            and not isinstance(raw_payload[field], dict)
        ):
            raise EventValidationError([{"field": field, "code": "mapping_required"}])
    received_at = received_at if received_at is not None else _utc_now_iso()
    rx_metadata = raw_payload.get("rx_metadata")
    first_rx_meta = rx_metadata[0] if isinstance(rx_metadata, list) and rx_metadata else {}
    first_rx_meta = first_rx_meta if isinstance(first_rx_meta, dict) else {}
    gateway_info = first_rx_meta.get("gateway_info", {})
    gateway_info = gateway_info if isinstance(gateway_info, dict) else {}
    is_lora_uplink = "dev_eui" in raw_payload or "app_eui" in raw_payload or "f_port" in raw_payload

    camera_id = next(
        (
            raw_payload[key]
            for key in (
                "camera_id", "cameraId", "source_id", "device_id",
                "deviceId", "dev_eui", "camera",
            )
            if key in raw_payload
        ),
        None,
    )
    event_type = (
        raw_payload.get("type")
        or raw_payload.get("event_type")
        or raw_payload.get("eventType")
        or raw_payload.get("label_hint")
        or raw_payload.get("sensor_type")
        or raw_payload.get("status")
        or ("lora_uplink" if is_lora_uplink else None)
        or "external_input"
    )
    sensor_type = raw_payload.get("sensor_type") or ("lora" if is_lora_uplink else None)
    payload_text = _decode_base64_text(raw_payload.get("payload")) if is_lora_uplink else None
    timestamp_source = next(
        (key for key in ("occurred_at", "timestamp", "time", "ts") if key in raw_payload),
        None,
    )
    if timestamp_source is not None:
        timestamp = raw_payload[timestamp_source]
    elif any(key in first_rx_meta for key in ("gw_recv_time", "time", "timestamp")):
        timestamp = _extract_lora_timestamp(raw_payload)
        timestamp_source = "rx_metadata[0]"
    else:
        timestamp = received_at
        timestamp_source = "received_at"
    validate_event_identity(raw_payload, device_id=camera_id, occurred_at=timestamp)
    message = (
        raw_payload.get("message")
        or raw_payload.get("display_message")
        or raw_payload.get("label")
        or raw_payload.get("description")
    )

    confidence = _coerce_float(
        raw_payload.get("confidence")
        or raw_payload.get("score")
        or raw_payload.get("probability")
    )
    normalized: Dict[str, Any] = {
        "camera_id": str(camera_id),
        "type": str(event_type),
        "timestamp": str(timestamp),
        "received_at": received_at,
        "occurred_at_source": timestamp_source,
        "source": "external_mqtt",
        "source_type": "mqtt",
        "confidence": confidence,
        "metadata": {
            "topic": topic,
            "source_id": raw_payload.get("source_id") or raw_payload.get("device_id") or raw_payload.get("deviceId") or raw_payload.get("dev_eui"),
            "sensor_type": sensor_type,
            "spec": raw_payload.get("spec") or _lora_spec(raw_payload, gateway_info),
            "telemetry": raw_payload.get("telemetry") or _lora_telemetry(first_rx_meta),
            "image_path": raw_payload.get("image_path"),
            "image_url": raw_payload.get("image_url"),
            "image_ref": raw_payload.get("image_ref"),
            "payload_base64": raw_payload.get("payload"),
            "payload_text": payload_text,
            "width": raw_payload.get("width"),
            "height": raw_payload.get("height"),
            "raw_payload": raw_payload,
        },
    }
    normalized.update(
        build_canonical_event(
            camera_id=str(camera_id),
            event_type=str(event_type),
            message_type="external_event",
            message_id=raw_payload.get("message_id"),
            occurred_at=timestamp,
            source="external_mqtt",
            source_type="mqtt",
            severity=raw_payload.get("severity"),
            confidence=confidence,
            message=message,
            display_message=raw_payload.get("display_message") or message,
            tts_message=raw_payload.get("tts_message") or message,
            device={
                "camera_id": str(camera_id),
                "device_id": raw_payload.get("device_id") or raw_payload.get("deviceId"),
                "app_eui": raw_payload.get("app_eui"),
                "dev_eui": raw_payload.get("dev_eui"),
                "f_port": raw_payload.get("f_port"),
                "f_cnt_up": raw_payload.get("f_cnt_up"),
            },
            gateway=_canonical_gateway_fields(first_rx_meta, gateway_info),
            decoded=raw_payload.get("decoded") or raw_payload.get("telemetry") or {},
            raw={
                "payload_base64": raw_payload.get("payload"),
                "payload_text": payload_text,
                "rx_metadata": rx_metadata,
                "raw_payload": raw_payload,
            },
        )
    )
    normalized["event_id"] = build_event_id(
        camera_id=str(camera_id),
        event_type=str(event_type),
        occurred_at=timestamp,
        message_id=raw_payload.get("message_id"),
        payload=raw_payload,
    )
    observation = build_sensor_observation(
        raw_payload,
        namespace="external_mqtt",
        source_record_id=normalized["event_id"],
        device_id=str(camera_id),
        received_at=received_at,
    )
    if observation is not None:
        normalized["observation"] = observation
    return normalized


class IngestEventRepository:
    """외부 수신 이벤트를 SQLite에 저장한다."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    raw_payload TEXT NOT NULL,
                    normalized_payload TEXT NOT NULL,
                    republished INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(ingest_events)")
            }
            if "event_id" not in columns:
                conn.execute(
                    "ALTER TABLE ingest_events ADD COLUMN event_id TEXT"
                )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_ingest_events_event_id ON ingest_events(event_id)"
            )
            conn.commit()

    def save(self, *, topic: str, raw_payload: Any, normalized_payload: Dict[str, Any], republished: bool) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO ingest_events (
                    event_id, received_at, topic, raw_payload, normalized_payload, republished
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_payload.get("event_id"),
                    _utc_now_iso(),
                    topic,
                    json.dumps(raw_payload, ensure_ascii=False),
                    json.dumps(normalized_payload, ensure_ascii=False),
                    1 if republished else 0,
                ),
            )
            conn.commit()


@dataclass
class IngestStats:
    received_count: int = 0
    parse_fail_count: int = 0
    republish_count: int = 0
    quarantined_count: int = 0


class ExternalIngestService:
    """외부 MQTT 입력을 수신하고 정규화해 저장한다."""

    def __init__(
        self,
        ingest_config: ExternalIngestConfig,
        output_config: Optional[MqttConfig] = None,
        *,
        ingest_store: Optional[EventIngestStore] = None,
    ) -> None:
        self.ingest_config = ingest_config
        self.output_config = output_config or MqttConfig()
        durable_path = os.environ.get("AIOT_INGEST_DB", "").strip()
        self._ingest_store = ingest_store
        if self._ingest_store is None and durable_path:
            self._ingest_store = EventIngestStore(durable_path)
        if (
            self._ingest_store is not None
            and ingest_config.republish_enabled
            and os.environ.get("MQTT_EVENT_OUTBOX_DB")
        ):
            raise ValueError(
                "Common ingest and legacy MQTT outbox replay cannot run together"
            )
        self._repo = IngestEventRepository(ingest_config.db_path)
        self._stats = IngestStats()
        self._durable_lock = threading.Lock()
        self._running = False
        self._publisher: Optional[MqttEventPublisher] = None
        if ingest_config.republish_enabled:
            self._publisher = MqttEventPublisher(
                broker=self.output_config.broker,
                port=self.output_config.port,
                topic_prefix=self.output_config.topic_prefix,
                client_id_prefix=self.output_config.client_id_prefix,
                qos=self.output_config.qos,
                retain=self.output_config.retain,
            )
        self._subscriber = MqttTopicSubscriber(
            broker=ingest_config.mqtt_broker,
            port=ingest_config.mqtt_port,
            topics=ingest_config.topics,
            client_id_prefix=ingest_config.client_id_prefix,
            client_id=ingest_config.mqtt_client_id,
            username=ingest_config.mqtt_username,
            password=ingest_config.mqtt_password,
            message_handler=self.handle_message,
            durable=self._ingest_store is not None,
        )

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "ExternalIngestService":
        return cls(config.external_ingest, config.mqtt)

    def handle_message(self, topic: str, payload_bytes: bytes) -> None:
        self._stats.received_count += 1
        if self._ingest_store is not None:
            with self._durable_lock:
                self._handle_durable_message(topic, payload_bytes)
            return
        received_at = _utc_now_iso()
        preview = payload_bytes[:160].decode("utf-8", errors="replace")
        logger.debug(
            "External ingest raw payload 수신: topic=%s payload_len=%d preview=%r",
            topic,
            len(payload_bytes),
            preview,
        )
        try:
            raw_payload = json.loads(payload_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._stats.parse_fail_count += 1
            logger.error(
                "External ingest JSON 파싱 실패: topic=%s payload_len=%d preview=%r",
                topic,
                len(payload_bytes),
                preview,
            )
            self._quarantine(
                topic=topic,
                raw_payload={"payload_base64": b64encode(payload_bytes).decode("ascii")},
                received_at=received_at,
                errors=[{"field": "payload", "code": "invalid_json"}],
            )
            return

        try:
            normalized = normalize_external_event(
                raw_payload, topic, received_at=received_at
            )
        except EventValidationError as exc:
            self._quarantine(
                topic=topic, raw_payload=raw_payload,
                received_at=received_at, errors=exc.errors,
            )
            return
        republished = False
        if self._publisher:
            republished = self._publisher.publish_event(normalized)
            if republished:
                self._stats.republish_count += 1

        self._repo.save(
            topic=topic,
            raw_payload=raw_payload,
            normalized_payload=normalized,
            republished=republished,
        )
        logger.info(
            "External ingest 수신: topic=%s camera_id=%s type=%s republished=%s",
            topic,
            normalized["camera_id"],
            normalized["type"],
            republished,
        )

    def _handle_durable_message(self, topic: str, payload_bytes: bytes) -> None:
        destination = None
        if self._publisher is not None:
            destination = json.dumps(
                {
                    "transport": "mqtt",
                    "broker": self.output_config.broker,
                    "port": self.output_config.port,
                    "topic_prefix": self.output_config.topic_prefix,
                    "qos": self.output_config.qos,
                    "retain": self.output_config.retain,
                },
                sort_keys=True,
            )
        ingest_id = self._ingest_store.accept_message(
            "external_mqtt", topic, payload_bytes, destination=destination
        )
        try:
            self._process_durable_receipt(ingest_id, recovery=False)
        except Exception as exc:
            self._ingest_store.retry_event(ingest_id, type(exc).__name__)
            raise

    def _process_durable_receipt(self, ingest_id: str, *, recovery: bool) -> None:
        store = self._ingest_store
        receipt = store.get_event_receipt(ingest_id)
        if receipt["stage"] != "pending":
            return
        envelope = receipt["raw"]
        try:
            raw_payload = json.loads(
                b64decode(envelope["payload_base64"]).decode("utf-8")
            )
            json.dumps(raw_payload, allow_nan=False)
        except (ValueError, UnicodeDecodeError):
            self._stats.parse_fail_count += 1
            store.reject_event(
                ingest_id,
                [{"field": "payload", "code": "invalid_json"}],
                converter_version="external-v1",
            )
            self._stats.quarantined_count += 1
            return
        try:
            normalized = normalize_external_event(
                raw_payload, envelope["topic"], received_at=receipt["received_at"]
            )
            json.dumps(normalized, allow_nan=False)
        except (EventValidationError, ValueError) as exc:
            errors = (
                exc.errors if isinstance(exc, EventValidationError)
                else [{"field": "payload", "code": "non_finite_normalized_value"}]
            )
            store.reject_event(ingest_id, errors, converter_version="external-v1")
            self._stats.quarantined_count += 1
            return
        status, job_id = store.record_event(
            ingest_id, normalized, converter_version="external-v1", recovery=recovery
        )
        if status == "conflict":
            logger.warning("External ingest ID conflict: ingest_id=%s", ingest_id)
        if recovery or job_id is None or self._publisher is None:
            return
        if not store.begin_live_delivery(job_id):
            return
        try:
            submitted = bool(self._publisher.publish_event(normalized))
            error = None if submitted else "publisher_returned_false"
        except Exception as exc:
            submitted, error = False, type(exc).__name__
        store.finish_live_delivery(job_id, submitted=submitted, error=error)
        if submitted:
            self._stats.republish_count += 1
        else:
            logger.warning(
                "External ingest delivery uncertain: ingest_id=%s error=%s", ingest_id, error
            )

    def recover_pending_once(self, limit: int = 100) -> int:
        """Resume normalization only. Old alarms are never automatically published."""
        if self._ingest_store is None:
            return 0
        processed = 0
        for ingest_id in self._ingest_store.pending_events("external_mqtt", limit):
            try:
                with self._durable_lock:
                    self._process_durable_receipt(ingest_id, recovery=True)
                processed += 1
            except Exception as exc:
                self._ingest_store.retry_event(ingest_id, type(exc).__name__)
                logger.warning("External ingest recovery deferred: ingest_id=%s", ingest_id)
        return processed

    def _quarantine(
        self, *, topic: str, raw_payload: Any, received_at: str, errors: list[dict[str, str]]
    ) -> None:
        rejected = {
            "event_id": f"rejected_{uuid4().hex}",
            "message_type": "invalid_external_input",
            "received_at": received_at,
            "validation": {"status": "invalid", "errors": errors},
        }
        self._repo.save(
            topic=topic, raw_payload=raw_payload,
            normalized_payload=rejected, republished=False,
        )
        self._stats.quarantined_count += 1
        logger.warning("External ingest 입력 격리: topic=%s errors=%s", topic, errors)

    def start(self) -> None:
        if self._ingest_store is None:
            self._repo.init()
        else:
            self.recover_pending_once()
        attempts = 0
        while not self._subscriber.connect():
            attempts += 1
            if attempts >= 30:
                raise RuntimeError("외부 MQTT 브로커 연결 실패")
            time.sleep(1.0)
        self._running = True

    def run_forever(self) -> None:
        self.start()
        logger.info(
            "External ingest 실행 중: broker=%s:%d topics=%s republish=%s",
            self.ingest_config.mqtt_broker,
            self.ingest_config.mqtt_port,
            ",".join(self.ingest_config.topics),
            self.ingest_config.republish_enabled,
        )
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        try:
            while self._running:
                self._subscriber.reconnect_if_needed()
                self.recover_pending_once()
                time.sleep(1.0)
        finally:
            self.stop()

    def _signal_handler(self, signum, frame) -> None:
        logger.info("External ingest 종료 신호 수신 (signum=%d)", signum)
        self._running = False

    def stop(self) -> None:
        self._subscriber.disconnect()
        if self._publisher:
            self._publisher.disconnect()
        self._running = False

    def get_stats(self) -> Dict[str, int]:
        return {
            "received_count": self._stats.received_count,
            "parse_fail_count": self._stats.parse_fail_count,
            "republish_count": self._stats.republish_count,
            "quarantined_count": self._stats.quarantined_count,
        }
