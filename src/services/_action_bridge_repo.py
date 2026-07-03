"""ActionBridge SQLite 이벤트 저장소."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

from ..canonical_event import (
    get_payload_camera_id,
    get_payload_confidence,
    get_payload_event_id,
    get_payload_event_type,
    get_payload_severity,
)
from ..event_priority import event_priority, event_risk_level, event_risk_score
from ..time_utils import now_kst_iso

logger = logging.getLogger(__name__)


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
            columns = {row[1] for row in conn.execute("PRAGMA table_info(action_events)")}
            if "event_id" not in columns:
                conn.execute("ALTER TABLE action_events ADD COLUMN event_id TEXT")
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

        return [self._row_to_dict(row) for row in rows]

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
            "risk_score": event_risk_score(payload),
        }
