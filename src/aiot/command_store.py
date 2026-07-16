from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ClaimResult:
    is_new: bool


@dataclass(frozen=True)
class CommandRecord:
    request_id: str
    message_type: str
    status: str
    expires_at: str
    result_payload: Optional[dict[str, Any]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _without_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_secrets(item)
            for key, item in value.items()
            if key not in {"upload_url", "authorization", "token"}
        }
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


class CommandStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS aiot_command_inbox (
                request_id TEXT PRIMARY KEY,
                message_type TEXT NOT NULL,
                status TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def claim(
        self, request_id: str, message_type: str, expires_at: datetime
    ) -> ClaimResult:
        now = _utc_now_iso()
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO aiot_command_inbox (
                    request_id, message_type, status, expires_at,
                    result_json, created_at, updated_at
                ) VALUES (?, ?, 'received', ?, NULL, ?, ?)
                """,
                (request_id, message_type, expires_at.isoformat(), now, now),
            )
            self._conn.commit()
            return ClaimResult(is_new=cursor.rowcount == 1)

    def update(
        self,
        request_id: str,
        status: str,
        result_payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        result_json = None
        if result_payload is not None:
            result_json = json.dumps(_without_secrets(result_payload), ensure_ascii=False)
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE aiot_command_inbox
                SET status = ?, result_json = ?, updated_at = ?
                WHERE request_id = ?
                """,
                (status, result_json, _utc_now_iso(), request_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(request_id)
            self._conn.commit()

    def get(self, request_id: str) -> Optional[CommandRecord]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT request_id, message_type, status, expires_at, result_json
                FROM aiot_command_inbox WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        result_payload = json.loads(row["result_json"]) if row["result_json"] else None
        return CommandRecord(
            request_id=row["request_id"],
            message_type=row["message_type"],
            status=row["status"],
            expires_at=row["expires_at"],
            result_payload=result_payload,
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

