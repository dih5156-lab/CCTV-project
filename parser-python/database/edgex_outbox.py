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
  EDGEX_OUTBOX_DB  : SQLite 파일 경로 (기본: /data/edgex_outbox.db)
                     로컬 개발 시 ./edgex_outbox.db 로 자동 fallback
"""

import json
import logging
import os
import sqlite3
import threading
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
# 컨테이너: /data/edgex_outbox.db
# 로컬 개발: {프로젝트루트}/data/edgex_outbox.db (fallback)
_DEFAULT_DB_PATH = os.environ.get(
    "EDGEX_OUTBOX_DB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "edgex_outbox.db"),
)
_MAX_RETRY = int(os.environ.get("EDGEX_OUTBOX_MAX_RETRY", "20"))
_TTL_SECONDS = int(os.environ.get("EDGEX_OUTBOX_TTL_SECONDS", str(60 * 60 * 24)))  # 24h

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS edgex_outbox (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id        TEXT    NOT NULL,
    table_name       TEXT    NOT NULL,
    core_data_url    TEXT    NOT NULL,
    edgex_event_json TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'pending',
    created_at_ms    INTEGER NOT NULL,
    sent_at_ms       INTEGER,
    retry_count      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_outbox_status
    ON edgex_outbox (status);
CREATE INDEX IF NOT EXISTS idx_outbox_created
    ON edgex_outbox (created_at_ms);
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
        conn.execute("PRAGMA journal_mode=WAL;")   # 동시 쓰기/읽기 성능 향상
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(_CREATE_TABLE_SQL)
        conn.commit()
        self._conn = conn

    def _cursor(self):
        return self._conn.cursor()

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
        with self._lock:
            cur = self._cursor()
            cur.execute(
                """
                INSERT INTO edgex_outbox
                    (device_id, table_name, core_data_url, edgex_event_json,
                     status, created_at_ms)
                VALUES (?, ?, ?, ?, 'pending', ?)
                """,
                (device_id, table_name, core_data_url, event_json, now_ms),
            )
            self._conn.commit()
            row_id = cur.lastrowid
        logger.debug("[Outbox] 저장 id=%s device=%s table=%s", row_id, device_id, table_name)
        return row_id

    def mark_sent(self, row_id: int) -> None:
        """전송 성공 → 'sent' 상태로 업데이트."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            self._conn.execute(
                "UPDATE edgex_outbox SET status='sent', sent_at_ms=? WHERE id=?",
                (now_ms, row_id),
            )
            self._conn.commit()

    def mark_failed(self, row_id: int) -> None:
        """최대 재시도 초과 → 'failed' 상태로 업데이트."""
        with self._lock:
            self._conn.execute(
                "UPDATE edgex_outbox SET status='failed' WHERE id=?",
                (row_id,),
            )
            self._conn.commit()

    def increment_retry(self, row_id: int) -> None:
        """재시도 횟수 증가."""
        with self._lock:
            self._conn.execute(
                "UPDATE edgex_outbox SET retry_count = retry_count + 1 WHERE id=?",
                (row_id,),
            )
            self._conn.commit()

    def get_pending(self, limit: int = 50) -> List[dict]:
        """재전송 대상 'pending' 행 목록 반환 (생성 순, TTL·재시도 필터 적용)."""
        cutoff_ms = int((time.time() - _TTL_SECONDS) * 1000)
        with self._lock:
            cur = self._cursor()
            cur.execute(
                """
                SELECT id, device_id, table_name, core_data_url,
                       edgex_event_json, retry_count, created_at_ms
                FROM edgex_outbox
                WHERE status = 'pending'
                  AND retry_count < ?
                  AND created_at_ms > ?
                ORDER BY created_at_ms ASC
                LIMIT ?
                """,
                (_MAX_RETRY, cutoff_ms, limit),
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
                    "id":              row[0],
                    "device_id":       row[1],
                    "table_name":      row[2],
                    "core_data_url":   row[3],
                    "edgex_event":     event,
                    "retry_count":     row[5],
                    "created_at_ms":   row[6],
                }
            )
        return result

    def pending_count(self) -> int:
        """현재 pending 건수 반환."""
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT COUNT(*) FROM edgex_outbox WHERE status='pending'")
            return cur.fetchone()[0]

    def expire_old_failed(self) -> int:
        """TTL 초과한 pending 항목을 'failed' 로 일괄 처리하고 처리 건수 반환."""
        cutoff_ms = int((time.time() - _TTL_SECONDS) * 1000)
        with self._lock:
            cur = self._cursor()
            cur.execute(
                """
                UPDATE edgex_outbox
                SET status = 'failed'
                WHERE status = 'pending'
                  AND (created_at_ms <= ? OR retry_count >= ?)
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
