"""
database/edgex_outbox.py
========================
EdgeX Core Data 전송 실패 시 로컬 SQLite에 저장하는 아웃박스(Outbox) 모듈.

동작 원리:
  1. EdgeXForwarder 가 Core Data POST 전에 outbox 에 'pending' 으로 저장
  2. POST 성공 시 'sent' 로 업데이트
  3. 실패 시 'pending' 유지 → 백그라운드 워커가 주기적으로 재시도
  4. MAX_RETRY 초과 또는 TTL 만료 시 'failed' 로 처리

환경변수:
  EDGEX_OUTBOX_DB  : SQLite 파일 경로 (기본: /data/runtime/event_outbox.db)
                     로컬 개발 시 ./data/runtime/event_outbox.db 로 자동 fallback
"""

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
# 컨테이너: /data/runtime/event_outbox.db
# 로컬 개발: {프로젝트루트}/data/runtime/event_outbox.db (fallback)
_DEFAULT_DB_PATH = os.environ.get(
    "EDGEX_OUTBOX_DB",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "runtime",
        "event_outbox.db",
    ),
)
_MAX_RETRY = int(os.environ.get("EDGEX_OUTBOX_MAX_RETRY", "20"))
_TTL_SECONDS = int(os.environ.get("EDGEX_OUTBOX_TTL_SECONDS", str(60 * 60 * 24)))  # 24h

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS event_outbox (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id         TEXT,
    source_service   TEXT    NOT NULL,
    data_category    TEXT    NOT NULL,
    destination_type TEXT    NOT NULL,
    destination_name TEXT    NOT NULL,
    camera_id        TEXT,
    device_id        TEXT,
    table_name       TEXT,
    core_data_url    TEXT,
    payload_json     TEXT    NOT NULL,
    created_at       TEXT,
    status           TEXT    NOT NULL DEFAULT 'pending',
    created_at_ms    INTEGER NOT NULL,
    expire_at        TEXT,
    expire_at_ms     INTEGER,
    last_attempt_at  TEXT,
    last_attempt_ms  INTEGER,
    sent_at          TEXT,
    sent_at_ms       INTEGER,
    retry_count      INTEGER NOT NULL DEFAULT 0,
    last_error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_status
    ON event_outbox (status, id);
CREATE INDEX IF NOT EXISTS idx_outbox_created
    ON event_outbox (created_at_ms);
CREATE INDEX IF NOT EXISTS idx_outbox_category
    ON event_outbox (data_category, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_event_dest
    ON event_outbox (event_id, destination_name);
"""


def _resolve_db_path(path: str) -> str:
    """경로의 디렉터리가 없으면 자동 생성하고 경로를 반환."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    return path


class EdgeXOutbox:
    """SQLite 기반 EdgeX 아웃박스.

    thread-safe: 내부적으로 threading.Lock 사용.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = _resolve_db_path(db_path or _DEFAULT_DB_PATH)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        logger.info("[Outbox] SQLite 아웃박스 초기화: %s", self._db_path)

    # ──────────────────────────────────────────
    # 내부 초기화
    # ──────────────────────────────────────────

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")  # 동시 쓰기/읽기 성능 향상
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(_CREATE_TABLE_SQL)
        self._ensure_event_outbox_columns(conn)
        self._migrate_legacy_table(conn)
        conn.commit()
        self._conn = conn

    def _cursor(self):
        return self._conn.cursor()

    def _ensure_event_outbox_columns(self, conn: sqlite3.Connection) -> None:
        """다른 서비스가 먼저 만든 event_outbox에도 필요한 컬럼을 추가한다."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(event_outbox)")}
        additions = {
            "event_id": "TEXT",
            "source_service": "TEXT NOT NULL DEFAULT 'aiot-parser'",
            "data_category": "TEXT NOT NULL DEFAULT 'sensor'",
            "destination_type": "TEXT NOT NULL DEFAULT 'http'",
            "destination_name": "TEXT NOT NULL DEFAULT 'edgex-core-data'",
            "camera_id": "TEXT",
            "device_id": "TEXT",
            "table_name": "TEXT",
            "core_data_url": "TEXT",
            "payload_json": "TEXT",
            "created_at": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'pending'",
            "created_at_ms": "INTEGER",
            "expire_at": "TEXT",
            "expire_at_ms": "INTEGER",
            "last_attempt_at": "TEXT",
            "last_attempt_ms": "INTEGER",
            "sent_at": "TEXT",
            "sent_at_ms": "INTEGER",
            "retry_count": "INTEGER NOT NULL DEFAULT 0",
            "last_error": "TEXT",
        }
        for column, definition in additions.items():
            if column not in cols:
                try:
                    conn.execute(
                        f"ALTER TABLE event_outbox ADD COLUMN {column} {definition}"
                    )
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise

    def _migrate_legacy_table(self, conn: sqlite3.Connection) -> None:
        """기존 edgex_outbox 테이블이 있으면 공통 event_outbox로 복사한다."""
        exists = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name='edgex_outbox'
            """
        ).fetchone()
        if not exists:
            return

        rows = conn.execute(
            """
            SELECT id, device_id, table_name, core_data_url, edgex_event_json,
                   status, created_at_ms, sent_at_ms, retry_count
            FROM edgex_outbox
            """
        ).fetchall()
        for row in rows:
            event_id = self._event_id_from_payload(row[4])
            created_at_ms = int(row[6])
            conn.execute(
                """
                INSERT OR IGNORE INTO event_outbox (
                    event_id, source_service, data_category, destination_type,
                    destination_name, device_id, table_name, core_data_url,
                    payload_json, status, created_at_ms, expire_at_ms,
                    last_attempt_ms, sent_at_ms, retry_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    "aiot-parser",
                    "sensor",
                    "http",
                    "edgex-core-data",
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    created_at_ms,
                    created_at_ms + (_TTL_SECONDS * 1000),
                    row[7],
                    row[7],
                    row[8],
                ),
            )

    @staticmethod
    def _event_id_from_payload(payload_json: str) -> str:
        try:
            payload = json.loads(payload_json)
            event = payload.get("event") if isinstance(payload, dict) else None
            if isinstance(event, dict) and event.get("id"):
                return str(event["id"])
        except Exception:
            pass
        return str(uuid.uuid4())

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def save_pending(
        self,
        device_id: str,
        table_name: str,
        core_data_url: str,
        edgex_event: dict,
    ) -> int:
        """아웃박스에 'pending' 상태로 저장하고 row id 반환."""
        now_ms = int(time.time() * 1000)
        event_json = json.dumps(edgex_event, ensure_ascii=False)
        event_id = self._event_id_from_payload(event_json)
        with self._lock:
            try:
                cur = self._cursor()
                cur.execute(
                    """
                    INSERT OR IGNORE INTO event_outbox
                        (event_id, source_service, data_category, destination_type,
                         destination_name, device_id, table_name, core_data_url,
                         payload_json, status, created_at_ms, expire_at_ms,
                         last_attempt_ms)
                    VALUES (?, 'aiot-parser', 'sensor', 'http',
                            'edgex-core-data', ?, ?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (
                        event_id,
                        device_id,
                        table_name,
                        core_data_url,
                        event_json,
                        now_ms,
                        now_ms + (_TTL_SECONDS * 1000),
                        now_ms,
                    ),
                )
                self._conn.commit()
                row_id = cur.lastrowid
                if row_id is None or row_id == 0:
                    row = self._conn.execute(
                        """
                        SELECT id FROM event_outbox
                        WHERE event_id = ? AND destination_name = ?
                        """,
                        (event_id, "edgex-core-data"),
                    ).fetchone()
                    row_id = int(row[0]) if row else 0
            except sqlite3.Error as exc:
                logger.error("[Outbox] save_pending DB 저장 에러: %s", exc)
                return 0
        logger.debug(
            "[Outbox] 저장 id=%s device=%s table=%s", row_id, device_id, table_name
        )
        return int(row_id)

    def mark_sent(self, row_id: int) -> None:
        """전송 성공 → 'sent' 상태로 업데이트."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            try:
                self._conn.execute(
                    """
                    UPDATE event_outbox
                    SET status='sent', sent_at_ms=?, last_attempt_ms=?
                    WHERE id=?
                    """,
                    (now_ms, now_ms, row_id),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.error("[Outbox] mark_sent 상태 업데이트 에러: %s", exc)

    def mark_failed(self, row_id: int) -> None:
        """최대 재시도 초과 → 'failed' 상태로 업데이트."""
        with self._lock:
            self._conn.execute(
                "UPDATE event_outbox SET status='failed' WHERE id=?",
                (row_id,),
            )
            self._conn.commit()

    def increment_retry(self, row_id: int) -> None:
        """재시도 횟수 증가."""
        with self._lock:
            self._conn.execute(
                """
                UPDATE event_outbox
                SET retry_count = retry_count + 1,
                    last_attempt_ms = ?
                WHERE id=?
                """,
                (int(time.time() * 1000), row_id),
            )
            self._conn.commit()

    def get_pending(self, limit: int = 50) -> List[dict]:
        """재전송 대상 'pending' 행 목록 반환 (생성 순, TTL·재시도 필터 적용)."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            cur = self._cursor()
            cur.execute(
                """
                SELECT id, device_id, table_name, core_data_url,
                       payload_json, retry_count, created_at_ms
                FROM event_outbox
                WHERE status = 'pending'
                  AND retry_count < ?
                  AND (expire_at_ms IS NULL OR expire_at_ms > ?)
                  AND data_category = 'sensor'
                ORDER BY created_at_ms ASC
                LIMIT ?
                """,
                (_MAX_RETRY, now_ms, limit),
            )
            rows = cur.fetchall()

        result = []
        for row in rows:
            try:
                event = json.loads(row[4])
            except Exception:
                event = {}
            result.append(
                {
                    "id": row[0],
                    "device_id": row[1],
                    "table_name": row[2],
                    "core_data_url": row[3],
                    "edgex_event": event,
                    "retry_count": row[5],
                    "created_at_ms": row[6],
                }
            )
        return result

    def pending_count(self) -> int:
        """현재 pending 건수 반환."""
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT COUNT(*) FROM event_outbox WHERE status='pending'")
            return cur.fetchone()[0]

    def expire_old_failed(self) -> int:
        """TTL 초과한 pending 항목을 'failed' 로 일괄 처리하고 처리 건수 반환."""
        cutoff_ms = int((time.time() - _TTL_SECONDS) * 1000)
        with self._lock:
            cur = self._cursor()
            cur.execute(
                """
                UPDATE event_outbox
                SET status = 'failed'
                WHERE status = 'pending'
                  AND (expire_at_ms <= ? OR retry_count >= ?)
                """,
                (cutoff_ms, _MAX_RETRY),
            )
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
