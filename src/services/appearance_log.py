"""외형 감지 기록 저장소 — SQLite 기반 인물 외형 로그.

실시간 감지된 인물의 외형 속성(상의/하의/헬멧 색상, 헬멧 착용 여부, 소지품, 성별, 나이 등)을
SQLite에 기록하고, 조건부 검색을 지원한다.

사용:
    log = AppearanceLog("data/appearance.db")
    log.insert(camera_id="cam_01", track_id=3, upper_color="black", ...)
    results = log.search(upper_color="black", gender="male",
                         time_from="2026-04-13 14:00:00")
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional

from ..canonical_event import build_event_id

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS appearance_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT,
    timestamp   REAL    NOT NULL,
    camera_id   TEXT    NOT NULL,
    track_id    INTEGER,
    upper_color TEXT,
    lower_color TEXT,
    has_helmet  INTEGER DEFAULT 0,
    helmet_color TEXT,
    has_backpack  INTEGER DEFAULT 0,
    has_handbag   INTEGER DEFAULT 0,
    has_suitcase  INTEGER DEFAULT 0,
    gender      TEXT,
    age_group   TEXT,
    face_name   TEXT,
    attribute_backend TEXT,
    crop_path   TEXT,
    bbox_x      INTEGER,
    bbox_y      INTEGER,
    bbox_w      INTEGER,
    bbox_h      INTEGER
);

CREATE INDEX IF NOT EXISTS idx_appearance_ts ON appearance_log(timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_appearance_event_id ON appearance_log(event_id);
CREATE INDEX IF NOT EXISTS idx_appearance_camera ON appearance_log(camera_id);
CREATE INDEX IF NOT EXISTS idx_appearance_upper ON appearance_log(upper_color);
CREATE INDEX IF NOT EXISTS idx_appearance_lower ON appearance_log(lower_color);
CREATE INDEX IF NOT EXISTS idx_appearance_has_helmet ON appearance_log(has_helmet);
CREATE INDEX IF NOT EXISTS idx_appearance_gender ON appearance_log(gender);
"""

# 삽입 쿨다운: 동일 track의 중복 기록 방지 (초)
_INSERT_COOLDOWN = 3.0


class AppearanceLog:
    """SQLite 기반 외형 감지 기록 저장소."""

    def __init__(self, db_path: str = "data/appearance.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._last_insert: Dict[str, float] = {}  # "cam:track" → timestamp

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._ensure_columns()
        self._conn.commit()
        logger.info("외형 로그 DB 초기화: %s", db_path)

    def _ensure_columns(self) -> None:
        """기존 DB를 새 외형 스키마로 점진 마이그레이션한다."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(appearance_log)").fetchall()
        }
        if "has_helmet" not in existing:
            self._conn.execute(
                "ALTER TABLE appearance_log ADD COLUMN has_helmet INTEGER DEFAULT 0"
            )
        if "helmet_color" not in existing:
            self._conn.execute(
                "ALTER TABLE appearance_log ADD COLUMN helmet_color TEXT"
            )
        if "event_id" not in existing:
            self._conn.execute(
                "ALTER TABLE appearance_log ADD COLUMN event_id TEXT"
            )
        if "attribute_backend" not in existing:
            self._conn.execute(
                "ALTER TABLE appearance_log ADD COLUMN attribute_backend TEXT"
            )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_appearance_event_id ON appearance_log(event_id)"
        )

    # ── 기록 ─────────────────────────────────────────────────────────

    def insert(
        self,
        camera_id: str,
        event_id: Optional[str] = None,
        track_id: Optional[int] = None,
        upper_color: Optional[str] = None,
        lower_color: Optional[str] = None,
        has_helmet: bool = False,
        helmet_color: Optional[str] = None,
        has_backpack: bool = False,
        has_handbag: bool = False,
        has_suitcase: bool = False,
        gender: Optional[str] = None,
        age_group: Optional[str] = None,
        face_name: Optional[str] = None,
        attribute_backend: Optional[str] = None,
        crop_path: Optional[str] = None,
        bbox_x: int = 0,
        bbox_y: int = 0,
        bbox_w: int = 0,
        bbox_h: int = 0,
        timestamp: Optional[float] = None,
    ) -> bool:
        """외형 기록 1건을 삽입한다. 쿨다운 내 중복은 무시."""
        now = timestamp or time.time()
        cooldown_key = f"{camera_id}:{track_id}"
        resolved_event_id = event_id or build_event_id(
            camera_id=camera_id,
            event_type="appearance_log",
            occurred_at=now,
            object_id=track_id,
            payload={
                "upper_color": upper_color,
                "lower_color": lower_color,
                "helmet_color": helmet_color,
                "face_name": face_name,
            },
        )

        with self._lock:
            last = self._last_insert.get(cooldown_key, 0.0)
            if now - last < _INSERT_COOLDOWN:
                return False
            self._last_insert[cooldown_key] = now

            try:
                self._conn.execute(
                    """INSERT OR IGNORE INTO appearance_log
                       (event_id, timestamp, camera_id, track_id,
                       upper_color, lower_color, has_helmet, helmet_color,
                       has_backpack, has_handbag, has_suitcase,
                       gender, age_group, face_name, attribute_backend, crop_path,
                       bbox_x, bbox_y, bbox_w, bbox_h)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        resolved_event_id, now, camera_id, track_id,
                        upper_color, lower_color, int(has_helmet), helmet_color,
                        int(has_backpack), int(has_handbag), int(has_suitcase),
                        gender, age_group, face_name, attribute_backend, crop_path,
                        bbox_x, bbox_y, bbox_w, bbox_h,
                    ),
                )
                self._conn.commit()
                return True
            except Exception as exc:
                logger.error("외형 로그 삽입 실패: %s", exc)
                return False

    # ── 검색 ─────────────────────────────────────────────────────────

    def search(
        self,
        camera_id: Optional[str] = None,
        upper_color: Optional[str] = None,
        lower_color: Optional[str] = None,
        has_helmet: Optional[bool] = None,
        helmet_color: Optional[str] = None,
        has_backpack: Optional[bool] = None,
        has_handbag: Optional[bool] = None,
        has_suitcase: Optional[bool] = None,
        gender: Optional[str] = None,
        age_group: Optional[str] = None,
        face_name: Optional[str] = None,
        time_from: Optional[float] = None,
        time_to: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """조건에 맞는 외형 기록을 검색한다."""
        clauses: List[str] = []
        params: List[object] = []

        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if upper_color:
            clauses.append("upper_color = ?")
            params.append(upper_color)
        if lower_color:
            clauses.append("lower_color = ?")
            params.append(lower_color)
        if has_helmet is not None:
            clauses.append("has_helmet = ?")
            params.append(int(has_helmet))
        if helmet_color:
            clauses.append("helmet_color = ?")
            params.append(helmet_color)
        if has_backpack is not None:
            clauses.append("has_backpack = ?")
            params.append(int(has_backpack))
        if has_handbag is not None:
            clauses.append("has_handbag = ?")
            params.append(int(has_handbag))
        if has_suitcase is not None:
            clauses.append("has_suitcase = ?")
            params.append(int(has_suitcase))
        if gender:
            clauses.append("gender = ?")
            params.append(gender)
        if age_group:
            clauses.append("age_group = ?")
            params.append(age_group)
        if face_name:
            clauses.append("face_name LIKE ?")
            params.append(f"%{face_name}%")
        if time_from is not None:
            clauses.append("timestamp >= ?")
            params.append(time_from)
        if time_to is not None:
            clauses.append("timestamp <= ?")
            params.append(time_to)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        # limit/offset을 정수로 직접 삽입 (SQL injection 안전: int 강제 변환)
        sql = (
            f"SELECT * FROM appearance_log{where}"
            f" ORDER BY timestamp DESC"
            f" LIMIT {int(limit)} OFFSET {int(offset)}"
        )

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def count(self, **kwargs) -> int:
        """검색 조건에 맞는 총 레코드 수를 반환한다."""
        clauses: List[str] = []
        params: List[object] = []

        for key in (
            "camera_id", "upper_color", "lower_color", "helmet_color",
            "gender", "age_group",
        ):
            val = kwargs.get(key)
            if val:
                clauses.append(f"{key} = ?")
                params.append(val)

        for key in ("has_helmet", "has_backpack", "has_handbag", "has_suitcase"):
            val = kwargs.get(key)
            if val is not None:
                clauses.append(f"{key} = ?")
                params.append(int(val))

        if kwargs.get("face_name"):
            clauses.append("face_name LIKE ?")
            params.append(f"%{kwargs['face_name']}%")
        if kwargs.get("time_from") is not None:
            clauses.append("timestamp >= ?")
            params.append(kwargs["time_from"])
        if kwargs.get("time_to") is not None:
            clauses.append("timestamp <= ?")
            params.append(kwargs["time_to"])

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT COUNT(*) FROM appearance_log{where}"

        with self._lock:
            return self._conn.execute(sql, params).fetchone()[0]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        return {
            "id": row["id"],
            "event_id": row["event_id"],
            "timestamp": row["timestamp"],
            "camera_id": row["camera_id"],
            "track_id": row["track_id"],
            "upper_color": row["upper_color"],
            "lower_color": row["lower_color"],
            "has_helmet": bool(row["has_helmet"]),
            "helmet_color": row["helmet_color"],
            "has_backpack": bool(row["has_backpack"]),
            "has_handbag": bool(row["has_handbag"]),
            "has_suitcase": bool(row["has_suitcase"]),
            "gender": row["gender"],
            "age_group": row["age_group"],
            "face_name": row["face_name"],
            "attribute_backend": row["attribute_backend"],
            "crop_path": row["crop_path"],
            "bbox_x": row["bbox_x"],
            "bbox_y": row["bbox_y"],
            "bbox_w": row["bbox_w"],
            "bbox_h": row["bbox_h"],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
