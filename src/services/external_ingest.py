"""외부 MQTT 입력을 내부 표준 이벤트로 정규화하는 서비스."""

from __future__ import annotations

import binascii
import json
import logging
import signal
import sqlite3
import threading
import time
from base64 import b64decode
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppConfig, ExternalIngestConfig, MqttConfig
from ..protocols import MqttEventPublisher, MqttTopicSubscriber

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _extract_lora_timestamp(raw_payload: Dict[str, Any]) -> Optional[str]:
    rx_metadata = raw_payload.get("rx_metadata")
    if isinstance(rx_metadata, list) and rx_metadata:
        first_meta = rx_metadata[0] or {}
        if isinstance(first_meta, dict):
            gw_recv_time = first_meta.get("gw_recv_time")
            if gw_recv_time:
                return str(gw_recv_time)
            meta_time = first_meta.get("time")
            if meta_time is not None:
                return str(meta_time)
            meta_timestamp = first_meta.get("timestamp")
            if meta_timestamp is not None:
                return str(meta_timestamp)
    return None


def normalize_external_event(raw_payload: Dict[str, Any], topic: str) -> Dict[str, Any]:
    """외부 입력 payload를 프로젝트 내부 이벤트 형식으로 정규화한다."""
    rx_metadata = raw_payload.get("rx_metadata")
    first_rx_meta = rx_metadata[0] if isinstance(rx_metadata, list) and rx_metadata else {}
    gateway_info = first_rx_meta.get("gateway_info", {}) if isinstance(first_rx_meta, dict) else {}
    is_lora_uplink = "dev_eui" in raw_payload or "app_eui" in raw_payload or "f_port" in raw_payload

    camera_id = (
        raw_payload.get("camera_id")
        or raw_payload.get("cameraId")
        or raw_payload.get("source_id")
        or raw_payload.get("device_id")
        or raw_payload.get("deviceId")
        or raw_payload.get("dev_eui")
        or raw_payload.get("camera")
        or "unknown"
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
    timestamp = (
        raw_payload.get("timestamp")
        or raw_payload.get("time")
        or raw_payload.get("ts")
        or _extract_lora_timestamp(raw_payload)
        or _utc_now_iso()
    )

    normalized = {
        "camera_id": str(camera_id),
        "type": str(event_type),
        "timestamp": str(timestamp),
        "source": "external_mqtt",
        "source_type": "mqtt",
        "confidence": _coerce_float(
            raw_payload.get("confidence")
            or raw_payload.get("score")
            or raw_payload.get("probability")
        ),
        "metadata": {
            "topic": topic,
            "source_id": raw_payload.get("source_id") or raw_payload.get("device_id") or raw_payload.get("deviceId") or raw_payload.get("dev_eui"),
            "sensor_type": sensor_type,
            "spec": raw_payload.get("spec") or {
                "message_id": raw_payload.get("message_id"),
                "app_eui": raw_payload.get("app_eui"),
                "dev_eui": raw_payload.get("dev_eui"),
                "f_port": raw_payload.get("f_port"),
                "f_cnt_up": raw_payload.get("f_cnt_up"),
                "is_confirmed": raw_payload.get("is_confirmed"),
                "gateway": gateway_info,
                "channel_plan": gateway_info.get("channel_plan") if isinstance(gateway_info, dict) else None,
            },
            "telemetry": raw_payload.get("telemetry") or {
                "modulation": first_rx_meta.get("modulation") if isinstance(first_rx_meta, dict) else None,
                "data_rate": first_rx_meta.get("data_rate") if isinstance(first_rx_meta, dict) else None,
                "coding_rate": first_rx_meta.get("coding_rate") if isinstance(first_rx_meta, dict) else None,
                "frequency": first_rx_meta.get("frequency") if isinstance(first_rx_meta, dict) else None,
                "channel": first_rx_meta.get("channel") if isinstance(first_rx_meta, dict) else None,
                "rssi": first_rx_meta.get("rssi") if isinstance(first_rx_meta, dict) else None,
                "snr": first_rx_meta.get("snr") if isinstance(first_rx_meta, dict) else None,
                "gateway_time": first_rx_meta.get("gw_recv_time") if isinstance(first_rx_meta, dict) else None,
            },
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
                    received_at TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    raw_payload TEXT NOT NULL,
                    normalized_payload TEXT NOT NULL,
                    republished INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()

    def save(self, *, topic: str, raw_payload: Dict[str, Any], normalized_payload: Dict[str, Any], republished: bool) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO ingest_events (
                    received_at, topic, raw_payload, normalized_payload, republished
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
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


class ExternalIngestService:
    """외부 MQTT 입력을 수신하고 정규화해 저장한다."""

    def __init__(
        self,
        ingest_config: ExternalIngestConfig,
        output_config: Optional[MqttConfig] = None,
    ) -> None:
        self.ingest_config = ingest_config
        self.output_config = output_config or MqttConfig()
        self._repo = IngestEventRepository(ingest_config.db_path)
        self._stats = IngestStats()
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
        )

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "ExternalIngestService":
        return cls(config.external_ingest, config.mqtt)

    def handle_message(self, topic: str, payload_bytes: bytes) -> None:
        self._stats.received_count += 1
        preview = payload_bytes[:160].decode("utf-8", errors="replace")
        logger.debug(
            "External ingest raw payload 수신: topic=%s payload_len=%d preview=%r",
            topic,
            len(payload_bytes),
            preview,
        )
        try:
            raw_payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            self._stats.parse_fail_count += 1
            logger.error(
                "External ingest JSON 파싱 실패: topic=%s payload_len=%d preview=%r",
                topic,
                len(payload_bytes),
                preview,
            )
            return

        normalized = normalize_external_event(raw_payload, topic)
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

    def start(self) -> None:
        self._repo.init()
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
        }
