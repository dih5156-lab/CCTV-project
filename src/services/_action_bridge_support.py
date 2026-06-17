"""ActionBridge 지원 타입과 내부 헬퍼."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

from ..canonical_event import (
    get_payload_camera_id,
    get_payload_confidence,
    get_payload_display_message,
    get_payload_event_id,
    get_payload_event_type,
    get_payload_severity,
    get_payload_tts_message,
)
from ..event_priority import event_priority, event_risk_level
from ..event_routing import decide_alert_forward
from ..time_utils import now_kst_iso
from .cctv_metrics import device_commands as _device_commands

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    """카메라 사이트별 조치 제어 방식."""

    AUTO = "auto"
    MANUAL = "manual"


class AlarmDevice(str, Enum):
    """사이트에 연결된 알람 장치 종류."""

    SPEAKER = "speaker"
    SIREN = "siren"
    SIGNBOARD = "signboard"


@dataclass
class SiteConfig:
    """IoT 플랫폼 사이트(현장) 설정."""

    site_id: str
    site_name: str
    site_nickname: str = ""
    camera_ids: List[str] = field(default_factory=list)
    control_mode: ControlMode = ControlMode.AUTO
    alarm_devices: List[AlarmDevice] = field(
        default_factory=lambda: [AlarmDevice.SPEAKER, AlarmDevice.SIGNBOARD]
    )
    confidence_threshold: Optional[float] = None
    display_message: str = ""
    tts_message: str = ""

    def to_dict(self) -> Dict:
        return {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "site_nickname": self.site_nickname,
            "camera_ids": self.camera_ids,
            "control_mode": self.control_mode.value,
            "alarm_devices": [device.value for device in self.alarm_devices],
            "confidence_threshold": self.confidence_threshold,
            "display_message": self.display_message,
            "tts_message": self.tts_message,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SiteConfig":
        threshold = data.get("confidence_threshold")
        if threshold in ("", None):
            threshold = None
        else:
            threshold = max(0.0, min(float(threshold), 1.0))
        return cls(
            site_id=data["site_id"],
            site_name=data.get("site_name", ""),
            site_nickname=data.get("site_nickname", ""),
            camera_ids=data.get("camera_ids", []),
            control_mode=ControlMode(data.get("control_mode", "auto")),
            alarm_devices=[
                AlarmDevice(device)
                for device in data.get("alarm_devices", ["speaker", "signboard"])
            ],
            confidence_threshold=threshold,
            display_message=str(data.get("display_message", "") or ""),
            tts_message=str(data.get("tts_message", "") or ""),
        )


class _EventRepo:
    """SQLite 이벤트 CRUD 전담 헬퍼."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id     TEXT    NOT NULL,
                    received_at  TEXT    NOT NULL,
                    topic        TEXT    NOT NULL,
                    camera_id    TEXT,
                    event_type   TEXT,
                    confidence   REAL,
                    severity     TEXT,
                    alarm_played INTEGER DEFAULT 0,
                    http_sent    INTEGER DEFAULT 0,
                    payload_json TEXT    NOT NULL
                )
                """
            )
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(action_events)")
            }
            if "event_id" not in columns:
                conn.execute(
                    "ALTER TABLE action_events ADD COLUMN event_id TEXT"
                )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_action_events_event_id ON action_events(event_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_events_camera_id ON action_events(camera_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_events_received_at ON action_events(received_at)"
            )
            conn.commit()

    def save(
        self,
        topic: str,
        payload: Dict,
        alarm_played: bool,
        http_sent: bool,
    ) -> None:
        try:
            event_id = get_payload_event_id(payload)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO action_events
                        (event_id, received_at, topic, camera_id, event_type, confidence,
                         severity, alarm_played, http_sent, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        now_kst_iso(),
                        topic,
                        get_payload_camera_id(payload),
                        get_payload_event_type(payload),
                        get_payload_confidence(payload),
                        get_payload_severity(payload),
                        int(alarm_played),
                        int(http_sent),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error("DB 저장 오류: %s", exc)

    def list_recent(self, limit: int = 20) -> List[Dict]:
        """최근 Action Layer 처리 이력을 최신순으로 반환한다."""
        safe_limit = max(1, min(int(limit), 100))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT id, event_id, received_at, topic, camera_id, event_type,
                           confidence, severity, alarm_played, http_sent, payload_json
                    FROM action_events
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.error("DB 조회 오류: %s", exc)
            return []

        items: List[Dict] = []
        for row in rows:
            items.append(self._row_to_dict(row))
        return items

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        return {
            "id": row["id"],
            "event_id": row["event_id"],
            "received_at": row["received_at"],
            "topic": row["topic"],
            "camera_id": row["camera_id"],
            "event_type": row["event_type"],
            "confidence": row["confidence"],
            "severity": row["severity"],
            "alarm_played": bool(row["alarm_played"]),
            "http_sent": bool(row["http_sent"]),
            "payload": payload,
            "priority": event_priority(payload),
            "risk_level": event_risk_level(payload),
        }


class _SiteRegistry:
    """사이트 설정과 수동 승인 큐를 관리한다."""

    def __init__(
        self,
        default_mode: ControlMode,
        initial_sites: Optional[List[SiteConfig]] = None,
    ) -> None:
        self.default_mode: ControlMode = default_mode
        self.default_alarm_devices: List[AlarmDevice] = list(AlarmDevice)
        self.default_confidence_threshold: Optional[float] = None
        self.default_display_message: str = ""
        self.default_tts_message: str = ""
        self._sites: Dict[str, SiteConfig] = {
            site.site_id: site for site in (initial_sites or [])
        }
        self._pending: Dict[str, Dict] = {}
        self._pending_lock = Lock()

    def add(self, site: SiteConfig) -> None:
        self._sites[site.site_id] = site
        logger.info(
            "사이트 등록: %s (%s) mode=%s",
            site.site_id,
            site.site_name,
            site.control_mode.value,
        )

    def remove(self, site_id: str) -> bool:
        if site_id in self._sites:
            del self._sites[site_id]
            logger.info("사이트 제거: %s", site_id)
            return True
        return False

    def list_all(self) -> List[Dict]:
        return [site.to_dict() for site in self._sites.values()]

    def find_by_camera(self, camera_id: str) -> Optional[SiteConfig]:
        return next(
            (site for site in self._sites.values() if camera_id in site.camera_ids),
            None,
        )

    def set_mode(self, mode: ControlMode, site_id: Optional[str] = None) -> None:
        if site_id:
            site = self._sites.get(site_id)
            if site:
                site.control_mode = mode
                logger.info("사이트 모드 변경: %s → %s", site_id, mode.value)
            else:
                logger.warning("set_mode: 사이트 없음 (%s)", site_id)
        else:
            self.default_mode = mode
            logger.info("전역 기본 모드 변경 → %s", mode.value)

    def default_settings(self) -> Dict:
        return {
            "mode": self.default_mode.value,
            "alarm_devices": [device.value for device in self.default_alarm_devices],
            "confidence_threshold": self.default_confidence_threshold,
            "display_message": self.default_display_message,
            "tts_message": self.default_tts_message,
        }

    def set_default_action_settings(
        self,
        *,
        alarm_devices: Optional[List[AlarmDevice]] = None,
        confidence_threshold: Optional[float] = None,
        display_message: Optional[str] = None,
        tts_message: Optional[str] = None,
    ) -> None:
        if alarm_devices is not None:
            self.default_alarm_devices = alarm_devices
        self.default_confidence_threshold = confidence_threshold
        if display_message is not None:
            self.default_display_message = display_message
        if tts_message is not None:
            self.default_tts_message = tts_message

    def resolve_alarm_devices(self, camera_id: str) -> List[AlarmDevice]:
        site = self.find_by_camera(camera_id)
        return site.alarm_devices if site else self.default_alarm_devices

    def resolve_action_settings(self, camera_id: str) -> Dict:
        site = self.find_by_camera(camera_id)
        if site:
            return {
                "site": site,
                "site_id": site.site_id,
                "confidence_threshold": site.confidence_threshold,
                "display_message": site.display_message,
                "tts_message": site.tts_message,
            }
        return {
            "site": None,
            "site_id": None,
            "confidence_threshold": self.default_confidence_threshold,
            "display_message": self.default_display_message,
            "tts_message": self.default_tts_message,
        }

    def resolve_mode(self, camera_id: str) -> Tuple[ControlMode, Optional[str]]:
        site = self.find_by_camera(camera_id)
        if site:
            return site.control_mode, site.site_id
        return self.default_mode, None

    def push_pending(
        self,
        event_id: str,
        topic: str,
        payload: Dict,
        site_id: Optional[str],
    ) -> None:
        with self._pending_lock:
            self._pending[event_id] = {
                "payload": payload,
                "topic": topic,
                "queued_at": now_kst_iso(),
                "site_id": site_id,
            }
        logger.info(
            "[수동 대기] event_id=%s camera=%s type=%s site=%s",
            event_id,
            payload.get("camera_id"),
            payload.get("type"),
            site_id,
        )

    def pop_pending(self, event_id: str) -> Optional[Dict]:
        with self._pending_lock:
            return self._pending.pop(event_id, None)

    def list_pending(self) -> List[Dict]:
        with self._pending_lock:
            return [
                {
                    "event_id": event_id,
                    "queued_at": info.get("queued_at"),
                    "site_id": info.get("site_id"),
                    "camera_id": get_payload_camera_id(info["payload"]),
                    "event_type": get_payload_event_type(info["payload"]),
                    "confidence": get_payload_confidence(info["payload"]),
                    "severity": get_payload_severity(info["payload"]),
                    "priority": event_priority(info["payload"]),
                    "risk_level": event_risk_level(info["payload"]),
                    "display_message": get_payload_display_message(info["payload"]),
                    "tts_message": get_payload_tts_message(info["payload"]),
                    "topic": info.get("topic"),
                }
                for event_id, info in self._pending.items()
            ]


class _AlarmCoordinator:
    """알람 토픽, 쿨다운, 재생 잠금을 관리한다."""

    _COOLDOWN_EXEMPT: frozenset = frozenset({"head", "fall_detected"})
    _DEVICE_OUTPUT_SUPPRESSED: frozenset = frozenset({"person"})

    def __init__(
        self,
        alarm_topics: Set[str],
        alarm_cooldown_seconds: int,
    ) -> None:
        self.alarm_topics = alarm_topics
        self.alarm_cooldown_seconds = max(1, int(alarm_cooldown_seconds))
        self._last_alarm_ts: Dict[Tuple[str, str], float] = {}
        self._block_until: Dict[str, float] = {}
        self._lock = Lock()

    @staticmethod
    def _mqtt_topic_matches(pattern: str, topic: str) -> bool:
        pat_parts = pattern.split("/")
        top_parts = topic.split("/")
        pattern_index = topic_index = 0
        while pattern_index < len(pat_parts) and topic_index < len(top_parts):
            if pat_parts[pattern_index] == "#":
                return True
            if (
                pat_parts[pattern_index] == "+"
                or pat_parts[pattern_index] == top_parts[topic_index]
            ):
                pattern_index += 1
                topic_index += 1
            else:
                return False
        return pattern_index == len(pat_parts) and topic_index == len(top_parts)

    def should_alarm(self, topic: str, payload: Dict) -> bool:
        event_type = get_payload_event_type(payload).lower()
        severity = get_payload_severity(payload).lower()
        if event_type in self._DEVICE_OUTPUT_SUPPRESSED:
            return False
        return (
            topic == "rest/inbound"                                               # REST API로 직접 수신한 이벤트
            or event_type in self._COOLDOWN_EXEMPT
            or any(self._mqtt_topic_matches(pattern, topic) for pattern in self.alarm_topics)
            or severity == "critical"
        )

    @staticmethod
    def is_demo_event(payload: Dict) -> bool:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return False
        return metadata.get("demo") is True or metadata.get("source") == "public-demo-ui"

    def try_acquire_slot(self, camera_id: str, event_type: str, *, force: bool = False) -> bool:
        if force:
            logger.info("데모 이벤트 - 알람 쿨다운 우회: camera=%s type=%s", camera_id, event_type)
            return True
        now = time.time()
        with self._lock:
            block_until = self._block_until.get(camera_id, 0.0)
            if now < block_until:
                remaining = int(block_until - now)
                logger.info("재생 잠금 중 - 스킵 (camera=%s, 남은 %d초)", camera_id, remaining)
                return False

            if event_type not in self._COOLDOWN_EXEMPT:
                key = (camera_id, event_type)
                last_ts = self._last_alarm_ts.get(key, 0.0)
                if now - last_ts < self.alarm_cooldown_seconds:
                    logger.info("알람 쿨다운 - 스킵 (camera=%s, type=%s)", camera_id, event_type)
                    return False

            key = (camera_id, event_type)
            self._last_alarm_ts[key] = now
            self._block_until[camera_id] = now + self.alarm_cooldown_seconds
        return True


class _ActionExecutor:
    """디바이스 알람 실행, HTTP 전송, 저장소 기록을 묶어 처리한다."""

    def __init__(
        self,
        *,
        repo: _EventRepo,
        alarm: _AlarmCoordinator,
        forwarder,
        speaker,
        signboard,
        siren,
        resolve_devices,
        publish_status,
        build_display_text,
        alarm_device_enum,
    ) -> None:
        self._repo = repo
        self._alarm = alarm
        self._forwarder = forwarder
        self._speaker = speaker
        self._signboard = signboard
        self._siren = siren
        self._resolve_devices = resolve_devices
        self._publish_status = publish_status
        self._build_display_text = build_display_text
        self._alarm_device_enum = alarm_device_enum

    def execute(self, topic: str, payload: Dict) -> None:
        """단일 이벤트에 대한 장치 조치와 외부 전송을 수행한다."""
        camera_id = get_payload_camera_id(payload)
        event_type = get_payload_event_type(payload).lower()
        severity = get_payload_severity(payload).lower()
        display_message = get_payload_display_message(payload)
        tts_message = get_payload_tts_message(payload)

        alarm_played = False
        if self._alarm.should_alarm(topic, payload) and self._alarm.try_acquire_slot(
            camera_id, event_type, force=self._alarm.is_demo_event(payload)
        ):
            devices = self._resolve_devices(camera_id)

            if self._alarm_device_enum.SPEAKER in devices:
                speaker_ok = self._speaker.play(
                    event_type,
                    severity,
                    camera_id,
                    text=tts_message,
                )
                if speaker_ok:
                    alarm_played = True
                _device_commands.labels(
                    device="speaker", status="ok" if speaker_ok else "skip"
                ).inc()

            if self._alarm_device_enum.SIGNBOARD in devices:
                signboard_ok = self._signboard.display(
                    text=display_message
                    or self._build_display_text(event_type, severity, camera_id),
                    title="경고!",
                    class_name=event_type,
                )
                if signboard_ok:
                    alarm_played = True
                _device_commands.labels(
                    device="signboard", status="ok" if signboard_ok else "skip"
                ).inc()

            if self._alarm_device_enum.SIREN in devices:
                siren_ok = self._siren.trigger(event_type, camera_id)
                if siren_ok:
                    alarm_played = True
                _device_commands.labels(device="siren", status="ok" if siren_ok else "skip").inc()

        forward_decision = decide_alert_forward(
            payload,
            has_targets=self._forwarder.has_targets,
        )
        if forward_decision.reason == "already_stored":
            logger.debug("중복 Alert 저장 방지를 위해 HTTP forward 생략: camera=%s type=%s", camera_id, event_type)
        elif forward_decision.should_forward:
            self._forwarder.forward(topic, payload)
        http_sent = forward_decision.http_sent

        self._repo.save(topic, payload, alarm_played=alarm_played, http_sent=http_sent)
        self._publish_status(
            "events/executed",
            {
                "camera_id": camera_id,
                "event_type": event_type,
                "severity": severity,
                "status": "executed",
                "alarm_played": alarm_played,
                "http_sent": http_sent,
                "devices": [device.value for device in self._resolve_devices(camera_id)],
            },
        )
