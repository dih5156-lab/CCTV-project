"""외형 검색 조건 저장소."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List

from ..storage import SQLiteDatabase
from ..time_utils import now_kst_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_conditions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);
"""


class AppearanceConditionStore:
    """SQLite 기반 외형 검색 조건 저장소."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        database = SQLiteDatabase(self._db_path)
        conn = database.connect()
        try:
            try:
                conn.execute(_SCHEMA)
                conn.commit()
            except sqlite3.OperationalError as exc:
                if "readonly" not in str(exc).lower():
                    raise
            yield conn
        finally:
            conn.close()

    def list_all(self) -> List[dict]:
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM search_conditions ORDER BY created_at"
                ).fetchall()
        except sqlite3.Error:
            return []

        conditions: List[dict] = []
        for row in rows:
            try:
                conditions.append(self._row_to_dict(row))
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
        return conditions

    def create(self, *, condition_id: str, name: str, payload: dict, enabled: bool) -> dict:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO search_conditions (id, name, payload, enabled, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    condition_id,
                    name,
                    json.dumps(payload),
                    int(enabled),
                    now_kst_iso(),
                ),
            )
            conn.commit()

        return {
            "id": condition_id,
            "name": name,
            "enabled": enabled,
            **payload,
        }

    def delete(self, condition_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM search_conditions WHERE id = ?", (condition_id,)
            )
            conn.commit()
            return bool(cur.rowcount)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        entry = json.loads(row["payload"])
        entry["id"] = row["id"]
        entry["name"] = row["name"]
        entry["enabled"] = bool(row["enabled"])
        return entry
