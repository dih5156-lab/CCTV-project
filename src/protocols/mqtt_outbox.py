"""SQLite backed MQTT event outbox.

The outbox is intentionally small: store before publish, mark sent after a
successful broker handoff, and replay pending rows in FIFO order.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..canonical_event import get_payload_event_id

logger = logging.getLogger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mqtt_event_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    destination_name TEXT NOT NULL,
    topic TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at_ms INTEGER NOT NULL,
    last_attempt_ms INTEGER,
    sent_at_ms INTEGER,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_mqtt_outbox_event_dest
    ON mqtt_event_outbox(event_id, destination_name);
CREATE INDEX IF NOT EXISTS idx_mqtt_outbox_status_id
    ON mqtt_event_outbox(status, id);
"""


class MqttEventOutbox:
    """Store-and-forward queue for MQTT event publishes."""

    def __init__(self, db_path: str | Path, *, destination_name: str = "mqtt") -> None:
        self.path = Path(db_path)
        self.destination_name = destination_name
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_CREATE_TABLE_SQL)
            conn.commit()

    def save_pending(self, topic: str, payload: Dict) -> int:
        """Insert a pending row if it does not already exist."""
        event_id = get_payload_event_id(payload)
        now_ms = int(time.time() * 1000)
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO mqtt_event_outbox (
                    event_id, destination_name, topic, payload_json, status,
                    created_at_ms, last_attempt_ms
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    event_id,
                    self.destination_name,
                    topic,
                    payload_json,
                    now_ms,
                    now_ms,
                ),
            )
            conn.commit()
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                """
                SELECT id
                FROM mqtt_event_outbox
                WHERE event_id = ? AND destination_name = ?
                """,
                (event_id, self.destination_name),
            ).fetchone()
            return int(row["id"]) if row else 0

    def mark_sent(self, row_id: int) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE mqtt_event_outbox
                SET status = 'sent',
                    sent_at_ms = ?,
                    last_attempt_ms = ?
                WHERE id = ?
                """,
                (now_ms, now_ms, row_id),
            )
            conn.commit()

    def mark_retry_failed(self, row_id: int, error: str) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE mqtt_event_outbox
                SET retry_count = retry_count + 1,
                    last_attempt_ms = ?,
                    last_error = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now_ms, error[:1000], row_id),
            )
            conn.commit()

    def get_pending(self, *, limit: int = 100, max_retry: Optional[int] = None) -> List[dict]:
        retry_filter = ""
        params: list[object] = [self.destination_name]
        if max_retry is not None:
            retry_filter = "AND retry_count < ?"
            params.append(int(max_retry))
        params.append(max(1, int(limit)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, topic, payload_json, retry_count
                FROM mqtt_event_outbox
                WHERE status = 'pending'
                  AND destination_name = ?
                  {retry_filter}
                ORDER BY id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()

        pending: List[dict] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {}
            pending.append(
                {
                    "id": int(row["id"]),
                    "topic": row["topic"],
                    "payload": payload,
                    "retry_count": int(row["retry_count"]),
                }
            )
        return pending

    def pending_count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM mqtt_event_outbox
                WHERE status = 'pending' AND destination_name = ?
                """,
                (self.destination_name,),
            ).fetchone()
            return int(row["count"])
