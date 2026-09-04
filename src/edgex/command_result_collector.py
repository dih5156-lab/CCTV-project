"""EdgeX 장치 결과를 공통 SQLite 감사 저장소에 기록한다."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class CommandResultStore:
    """장치 Command 결과를 request_id 기준으로 저장하는 저장소."""

    def __init__(self, db_path: str | Path) -> None:
        """결과 저장소 경로를 초기화하고 테이블을 준비한다."""
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """부모 디렉터리를 준비한 뒤 SQLite 연결을 반환한다."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        """장치 결과 감사 기록 테이블을 생성한다."""
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS command_results (
                    request_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_code TEXT,
                    topic TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    received_at TEXT NOT NULL
                )
                """
            )

    def record(
        self,
        topic: str,
        payload: Mapping[str, Any],
        received_at: Optional[str] = None,
    ) -> bool:
        """유효한 장치 결과를 request_id 기준으로 저장하거나 갱신한다."""
        request_id = str(payload.get("request_id") or "").strip()
        event_id = str(payload.get("event_id") or "").strip()
        device_id = str(payload.get("device_id") or "").strip()
        status = str(payload.get("status") or "").strip()
        if not request_id or not event_id or not device_id or not status:
            return False

        timestamp = received_at or datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO command_results
                    (request_id, event_id, device_id, status, error_code,
                     topic, payload_json, received_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    status=excluded.status,
                    error_code=excluded.error_code,
                    topic=excluded.topic,
                    payload_json=excluded.payload_json,
                    received_at=excluded.received_at
                """,
                (
                    request_id,
                    event_id,
                    device_id,
                    status,
                    payload.get("error_code"),
                    topic,
                    json.dumps(dict(payload), ensure_ascii=False),
                    timestamp,
                ),
            )
        return True

    def get(self, request_id: str) -> Dict[str, Any]:
        """request_id에 해당하는 최신 장치 결과를 반환한다."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM command_results WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return dict(row) if row else {}

    def list_recent(
        self,
        limit: int = 100,
        *,
        device_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """최근 장치 결과를 선택 조건과 함께 수신 시각 역순으로 반환한다."""
        conditions = []
        parameters: list[Any] = []
        if device_id:
            conditions.append("device_id = ?")
            parameters.append(device_id)
        if status:
            conditions.append("status = ?")
            parameters.append(status)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.append(max(1, int(limit)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM command_results{where} ORDER BY received_at DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]
