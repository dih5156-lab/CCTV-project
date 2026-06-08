"""SQLite outbox for external HTTP event forwarding."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..canonical_event import get_payload_event_id

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS http_event_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    target_name TEXT NOT NULL,
    target_url TEXT NOT NULL,
    body_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at_ms INTEGER NOT NULL,
    last_attempt_ms INTEGER,
    sent_at_ms INTEGER,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_http_outbox_event_target
    ON http_event_outbox(event_id, target_name, target_url);
CREATE INDEX IF NOT EXISTS idx_http_outbox_status_id
    ON http_event_outbox(status, id);
"""


class HttpEventOutbox:
    """Store-and-forward queue for external HTTP event deliveries."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
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

    @staticmethod
    def _event_id_from_body(body: Dict) -> str:
        event = body.get("event") if isinstance(body.get("event"), dict) else body
        return get_payload_event_id(event)

    def save_pending(self, target_name: str, target_url: str, body: Dict) -> int:
        event_id = self._event_id_from_body(body)
        body_json = json.dumps(body, ensure_ascii=False)
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO http_event_outbox (
                    event_id, target_name, target_url, body_json, status,
                    created_at_ms, last_attempt_ms
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                (event_id, target_name, target_url, body_json, now_ms, now_ms),
            )
            conn.commit()
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                """
                SELECT id
                FROM http_event_outbox
                WHERE event_id = ? AND target_name = ? AND target_url = ?
                """,
                (event_id, target_name, target_url),
            ).fetchone()
            return int(row["id"]) if row else 0

    def mark_sent(self, row_id: int) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE http_event_outbox
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
                UPDATE http_event_outbox
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
        params: list[object] = []
        if max_retry is not None:
            retry_filter = "AND retry_count < ?"
            params.append(int(max_retry))
        params.append(max(1, int(limit)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, target_name, target_url, body_json, retry_count
                FROM http_event_outbox
                WHERE status = 'pending'
                  {retry_filter}
                ORDER BY id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()

        pending: List[dict] = []
        for row in rows:
            try:
                body = json.loads(row["body_json"])
            except json.JSONDecodeError:
                body = {}
            pending.append(
                {
                    "id": int(row["id"]),
                    "target_name": row["target_name"],
                    "target_url": row["target_url"],
                    "body": body,
                    "retry_count": int(row["retry_count"]),
                }
            )
        return pending

    def pending_count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM http_event_outbox WHERE status = 'pending'"
            ).fetchone()
            return int(row["count"])
