"""Event review storage for false-positive/true-positive labeling."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..time_utils import now_kst_iso

_VALID_STATUSES = frozenset({"true_positive", "false_positive", "uncertain"})


class EventReviewStore:
    """Small SQLite store that keeps operator review labels separate from events."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(
            db_path
            or os.environ.get("EVENT_REVIEW_DB", "data/runtime/event_reviews.db")
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    reviewed_at TEXT NOT NULL,
                    reviewer TEXT,
                    status TEXT NOT NULL,
                    category TEXT,
                    note TEXT,
                    camera_id TEXT,
                    event_type TEXT,
                    event_timestamp REAL,
                    object_id TEXT,
                    event_context_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_reviews_status ON event_reviews(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_reviews_type ON event_reviews(event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_reviews_reviewed_at ON event_reviews(reviewed_at)"
            )
            conn.commit()

    @staticmethod
    def _normalize_status(status: str) -> str:
        normalized = str(status or "").strip().lower()
        if normalized not in _VALID_STATUSES:
            raise ValueError(f"invalid review status: {status}")
        return normalized

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "event_id": row["event_id"],
            "reviewed_at": row["reviewed_at"],
            "reviewer": row["reviewer"],
            "status": row["status"],
            "category": row["category"],
            "note": row["note"],
            "camera_id": row["camera_id"],
            "event_type": row["event_type"],
            "event_timestamp": row["event_timestamp"],
            "object_id": row["object_id"],
        }

    def upsert(
        self,
        *,
        event_id: str,
        status: str,
        reviewer: Optional[str] = None,
        category: Optional[str] = None,
        note: Optional[str] = None,
        event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event_id = str(event_id or "").strip()
        if not event_id:
            raise ValueError("event_id is required")
        normalized_status = self._normalize_status(status)
        event = dict(event or {})
        reviewed_at = now_kst_iso()
        object_id = event.get("object_id")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO event_reviews (
                    event_id, reviewed_at, reviewer, status, category, note,
                    camera_id, event_type, event_timestamp, object_id, event_context_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    reviewed_at=excluded.reviewed_at,
                    reviewer=excluded.reviewer,
                    status=excluded.status,
                    category=excluded.category,
                    note=excluded.note,
                    camera_id=excluded.camera_id,
                    event_type=excluded.event_type,
                    event_timestamp=excluded.event_timestamp,
                    object_id=excluded.object_id,
                    event_context_json=excluded.event_context_json
                """,
                (
                    event_id,
                    reviewed_at,
                    reviewer,
                    normalized_status,
                    category,
                    note,
                    event.get("camera_id"),
                    event.get("event_type") or event.get("type"),
                    event.get("timestamp"),
                    None if object_id is None else str(object_id),
                    json.dumps(event, ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM event_reviews WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def get_many(self, event_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        ids = [str(event_id) for event_id in event_ids if event_id]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM event_reviews WHERE event_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {row["event_id"]: self._row_to_dict(row) for row in rows}

    def summary(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
            by_status = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM event_reviews GROUP BY status"
                ).fetchall()
            }
            by_type = [
                {"event_type": row["event_type"] or "unknown", "count": row["count"]}
                for row in conn.execute(
                    """
                    SELECT event_type, COUNT(*) AS count
                    FROM event_reviews
                    GROUP BY event_type
                    ORDER BY count DESC
                    LIMIT 20
                    """
                ).fetchall()
            ]
            recent = [
                self._row_to_dict(row)
                for row in conn.execute(
                    "SELECT * FROM event_reviews ORDER BY reviewed_at DESC LIMIT 20"
                ).fetchall()
            ]
        return {
            "total": int(total),
            "by_status": {
                "true_positive": int(by_status.get("true_positive", 0)),
                "false_positive": int(by_status.get("false_positive", 0)),
                "uncertain": int(by_status.get("uncertain", 0)),
            },
            "by_event_type": by_type,
            "recent": recent,
        }
