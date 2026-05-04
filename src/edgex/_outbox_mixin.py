"""
EdgeX Outbox Mixin — SQLite store-and-forward 로직

CCTVDeviceService 에 믹스인으로 포함되며, 전송 실패 이벤트를 공통
event_outbox SQLite 테이블에 저장하고 복구 후 재전송하는 기능을 담당한다.
각 메서드는 self.enable_store_and_forward, self.outbox_db_path,
self.outbox_flush_batch_size, self._outbox_lock 에 의존한다.
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..canonical_event import get_payload_event_id

logger = logging.getLogger(__name__)

_OUTBOX_TTL_DAYS = 7


class _OutboxMixin:
    """SQLite 기반 store-and-forward outbox 기능을 제공하는 믹스인."""

    # ── 데이터 카테고리 상수 ─────────────────────────────────────────────────
    _PERSON_EVENT_TYPES = frozenset({
        "helmet", "head", "unsafe_behavior", "wearing_helmet",
        "fall_detected", "not_fall",
        "face_recognized", "face_unknown", "person",
    })
    _ZONE_EVENT_TYPES = frozenset({
        "danger_zone", "zone_entered", "zone_dwelling",
        "zone_object_detected", "crowd_warning",
        "intrusion",
    })
    _SENSOR_EVENT_TYPES = frozenset({
        "tilt_alert", "temperature_alert", "vibration_alert",
        "sensor_data",
    })
    _CAMERA_EVENT_TYPES = frozenset({
        "other",
    })

    _OUTBOX_TABLE = "event_outbox"
    _LEGACY_OUTBOX_TABLES = ("detection_outbox", "sensor_outbox", "zone_outbox")

    @classmethod
    def _table_for_category(cls, category: str) -> str:
        return cls._OUTBOX_TABLE

    # ── 유틸리티 ────────────────────────────────────────────────────────────

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _outbox_connect(self):
        """부모 디렉토리를 보장한 뒤 SQLite 연결을 반환."""
        self.outbox_db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.outbox_db_path), timeout=15)

    # ── 이벤트 분류 ──────────────────────────────────────────────────────────

    def _classify_event_category(self, event_type: str) -> str:
        """이벤트 타입으로 data_category 분류.

        Returns:
            'person' | 'zone' | 'sensor' | 'camera'
        """
        normalized = (event_type or "").lower().strip()
        if normalized in self._PERSON_EVENT_TYPES:
            return "person"
        if normalized in self._ZONE_EVENT_TYPES:
            return "zone"
        if normalized in self._SENSOR_EVENT_TYPES:
            return "sensor"
        if normalized in self._CAMERA_EVENT_TYPES:
            return "camera"
        return "camera"

    # ── DB 초기화 ────────────────────────────────────────────────────────────

    def _init_outbox(self) -> None:
        """Prepare the local SQLite outbox used for store-and-forward delivery."""
        if not self.enable_store_and_forward:
            return

        self.outbox_db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            conn = sqlite3.connect(str(self.outbox_db_path), timeout=30)
        except Exception as exc:
            logger.warning("[Outbox] DB 파일 열기 실패 — store-and-forward 비활성화: %s", exc)
            self.enable_store_and_forward = False
            return

        try:
            with conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS event_outbox (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT,
                        source_service TEXT NOT NULL,
                        data_category TEXT NOT NULL DEFAULT 'camera',
                        destination_type TEXT NOT NULL DEFAULT 'edgex',
                        destination_name TEXT NOT NULL DEFAULT 'edgex-core',
                        camera_id TEXT,
                        device_id TEXT,
                        table_name TEXT,
                        core_data_url TEXT,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        created_at_ms INTEGER,
                        expire_at TEXT,
                        expire_at_ms INTEGER,
                        last_attempt_at TEXT,
                        last_attempt_ms INTEGER,
                        sent_at TEXT,
                        sent_at_ms INTEGER,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'pending',
                        last_error TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_event_outbox_status_id "
                    "ON event_outbox(status, id)"
                )
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_event_outbox_event_dest "
                    "ON event_outbox(event_id, destination_name)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_event_outbox_category "
                    "ON event_outbox(data_category, status)"
                )
                self._ensure_event_outbox_columns(conn)
                self._migrate_legacy_outbox_tables(conn)
                conn.commit()
        except Exception as exc:
            logger.warning("[Outbox] DB 초기화 실패 — store-and-forward 비활성화: %s", exc)
            self.enable_store_and_forward = False
        finally:
            conn.close()

    def _ensure_event_outbox_columns(self, conn: sqlite3.Connection) -> None:
        """다른 서비스가 먼저 만든 event_outbox에도 필요한 컬럼을 추가한다."""
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(event_outbox)")
        }
        additions = {
            "event_id": "TEXT",
            "source_service": "TEXT NOT NULL DEFAULT 'cctv-edgex-adapter'",
            "data_category": "TEXT NOT NULL DEFAULT 'camera'",
            "destination_type": "TEXT NOT NULL DEFAULT 'edgex'",
            "destination_name": "TEXT NOT NULL DEFAULT 'edgex-core'",
            "camera_id": "TEXT",
            "device_id": "TEXT",
            "table_name": "TEXT",
            "core_data_url": "TEXT",
            "payload_json": "TEXT",
            "created_at": "TEXT",
            "created_at_ms": "INTEGER",
            "expire_at": "TEXT",
            "expire_at_ms": "INTEGER",
            "last_attempt_at": "TEXT",
            "last_attempt_ms": "INTEGER",
            "sent_at": "TEXT",
            "sent_at_ms": "INTEGER",
            "retry_count": "INTEGER NOT NULL DEFAULT 0",
            "status": "TEXT NOT NULL DEFAULT 'pending'",
            "last_error": "TEXT",
        }
        for column, definition in additions.items():
            if column not in cols:
                conn.execute(f"ALTER TABLE event_outbox ADD COLUMN {column} {definition}")

    def _migrate_legacy_outbox_tables(self, conn: sqlite3.Connection) -> None:
        """기존 카테고리별 outbox 테이블을 공통 event_outbox로 복사한다."""
        for table in self._LEGACY_OUTBOX_TABLES:
            exists = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type='table' AND name=?
                """,
                (table,),
            ).fetchone()
            if not exists:
                continue

            cols = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})")
            }
            if not {"camera_id", "payload_json", "created_at", "status"}.issubset(cols):
                continue
            if "event_id" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN event_id TEXT")
            if "destination_name" not in cols:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN destination_name TEXT NOT NULL DEFAULT 'edgex-core'"
                )
            if "data_category" not in cols:
                default_category = (
                    "sensor" if table == "sensor_outbox"
                    else "zone" if table == "zone_outbox"
                    else "camera"
                )
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN data_category TEXT NOT NULL DEFAULT '{default_category}'"
                )
            if "expire_at" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN expire_at TEXT")
            if "last_attempt_at" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN last_attempt_at TEXT")
            if "sent_at" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN sent_at TEXT")
            if "retry_count" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
            if "last_error" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN last_error TEXT")

            rows = conn.execute(
                f"""
                SELECT id,
                       event_id,
                       COALESCE(destination_name, 'edgex-core') AS destination_name,
                       camera_id,
                       COALESCE(data_category, 'camera') AS data_category,
                       payload_json,
                       created_at,
                       expire_at,
                       last_attempt_at,
                       sent_at,
                       retry_count,
                       status,
                       last_error
                FROM {table}
                """
            ).fetchall()
            migrated = 0
            for row in rows:
                event_id = row[1]
                if not event_id:
                    try:
                        event_id = get_payload_event_id(json.loads(row[5]))
                    except Exception:
                        event_id = None
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO event_outbox (
                        event_id, source_service, data_category, destination_type,
                        destination_name, camera_id, payload_json, created_at,
                        created_at_ms, expire_at, expire_at_ms, last_attempt_at,
                        last_attempt_ms, sent_at, sent_at_ms, retry_count,
                        status, last_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        "cctv-edgex-adapter",
                        row[4],
                        "edgex",
                        row[2],
                        row[3],
                        row[5],
                        row[6],
                        int(time.time() * 1000),
                        row[7],
                        None,
                        row[8],
                        None,
                        row[9],
                        None,
                        row[10],
                        row[11],
                        row[12],
                    ),
                )
                migrated += int(cur.rowcount or 0)
            if migrated:
                logger.info("%s → event_outbox 마이그레이션: %d건", table, migrated)

    # ── 이벤트 저장 ──────────────────────────────────────────────────────────

    def _store_failed_detection_event(
        self,
        camera_id: str,
        event_data: Dict[str, Any],
        last_error: str,
    ) -> None:
        """전송 실패 이벤트를 outbox 에 저장 (status='pending')."""
        if not self.enable_store_and_forward:
            return

        event_type = ""
        if isinstance(event_data, dict):
            event_type = str(event_data.get("type") or event_data.get("event_type") or "")
        category = self._classify_event_category(event_type)

        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            now_ms = int(time.time() * 1000)
            event_id = get_payload_event_id(event_data)
            expire_at = datetime.now(timezone.utc).replace(
                microsecond=0
            ).isoformat()
            conn.execute(
                """
                INSERT OR IGNORE INTO event_outbox (
                    event_id, source_service, data_category, destination_type,
                    destination_name, camera_id, payload_json, created_at,
                    created_at_ms, expire_at, expire_at_ms, last_attempt_at,
                    last_attempt_ms, retry_count, status, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime(?, '+7 days'), ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    "cctv-edgex-adapter",
                    category,
                    "edgex",
                    "edgex-core",
                    camera_id,
                    json.dumps(event_data, ensure_ascii=False),
                    now,
                    now_ms,
                    now,
                    now_ms + (_OUTBOX_TTL_DAYS * 24 * 60 * 60 * 1000),
                    now,
                    now_ms,
                    1,
                    "pending",
                    last_error[:1000],
                ),
            )
            conn.commit()
            logger.debug(
                "[Outbox:event_outbox] 저장: camera=%s category=%s type=%s",
                camera_id, category, event_type,
            )

    def _store_pending_event(
        self,
        camera_id: str,
        event_data: Dict[str, Any],
    ) -> Optional[tuple]:
        """모든 이벤트를 pending 으로 먼저 저장하고 (table, row_id) 반환."""
        if not self.enable_store_and_forward:
            return None

        event_type = ""
        if isinstance(event_data, dict):
            event_type = str(event_data.get("type") or event_data.get("event_type") or "")
        category = self._classify_event_category(event_type)

        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            now_ms = int(time.time() * 1000)
            event_id = get_payload_event_id(event_data)
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO event_outbox (
                    event_id, source_service, data_category, destination_type,
                    destination_name, camera_id, payload_json, created_at,
                    created_at_ms, expire_at, expire_at_ms, last_attempt_at,
                    last_attempt_ms, retry_count, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime(?, '+7 days'), ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    "cctv-edgex-adapter",
                    category,
                    "edgex",
                    "edgex-core",
                    camera_id,
                    json.dumps(event_data, ensure_ascii=False),
                    now,
                    now_ms,
                    now,
                    now_ms + (_OUTBOX_TTL_DAYS * 24 * 60 * 60 * 1000),
                    now,
                    now_ms,
                    0,
                    "pending",
                ),
            )
            conn.commit()
            if cur.lastrowid is None or cur.lastrowid == 0:
                row = conn.execute(
                    "SELECT id FROM event_outbox WHERE event_id = ? AND destination_name = ?",
                    (event_id, "edgex-core"),
                ).fetchone()
                if row:
                    return ("event_outbox", int(row[0]))
            logger.debug(
                "[Outbox:event_outbox] pending 저장: camera=%s category=%s type=%s id=%s",
                camera_id, category, event_type, cur.lastrowid,
            )
            return ("event_outbox", cur.lastrowid)

    # ── 이벤트 조회 ──────────────────────────────────────────────────────────

    def get_pending_detection_events(
        self,
        limit: Optional[int] = None,
        data_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return pending outbox rows in FIFO order for replay.

        Args:
            limit: 최대 반환 행 수 (기본: outbox_flush_batch_size)
            data_category: 'person' | 'zone' | 'sensor' | 'camera' | None(전체)
        """
        if not self.enable_store_and_forward:
            return []

        fetch_limit = int(limit or self.outbox_flush_batch_size)
        with self._outbox_lock, self._outbox_connect() as conn:
            conn.row_factory = sqlite3.Row
            if data_category:
                rows = conn.execute(
                    """
                    SELECT id, camera_id, data_category, payload_json,
                           event_id, destination_name, created_at, expire_at,
                           last_attempt_at, retry_count, status, last_error
                    FROM event_outbox
                    WHERE status = 'pending' AND data_category = ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (data_category, fetch_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, camera_id, data_category, payload_json,
                           event_id, destination_name, created_at, expire_at,
                           last_attempt_at, retry_count, status, last_error
                    FROM event_outbox
                    WHERE status = 'pending'
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (fetch_limit,),
                ).fetchall()

        pending: List[Dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {}
            pending.append(
                {
                    "id": row["id"],
                    "_table": "event_outbox",
                    "camera_id": row["camera_id"],
                    "event_id": row["event_id"],
                    "destination_name": row["destination_name"],
                    "data_category": row["data_category"],
                    "event_data": payload,
                    "created_at": row["created_at"],
                    "expire_at": row["expire_at"],
                    "last_attempt_at": row["last_attempt_at"],
                    "retry_count": row["retry_count"],
                    "status": row["status"],
                    "last_error": row["last_error"],
                }
            )
        return pending

    def expire_pending_detection_events(self) -> int:
        """TTL 이 지난 pending outbox 행을 만료 상태로 전환한다."""
        if not self.enable_store_and_forward:
            return 0

        expired = 0
        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            cur = conn.execute(
                """
                UPDATE event_outbox
                SET status = 'expired',
                    last_attempt_at = ?,
                    last_attempt_ms = ?,
                    last_error = COALESCE(last_error, 'ttl expired')
                WHERE status = 'pending'
                  AND expire_at IS NOT NULL
                  AND datetime(expire_at) <= datetime(?)
                """,
                (now, int(time.time() * 1000), now),
            )
            expired += int(cur.rowcount or 0)
            conn.commit()
        return expired

    # ── Outbox 상태 갱신 ─────────────────────────────────────────────────────

    def _mark_outbox_sent(self, outbox_ref) -> None:
        """outbox_ref: (table, row_id) tuple 또는 int (하위 호환)."""
        if not self.enable_store_and_forward or outbox_ref is None:
            return
        if isinstance(outbox_ref, tuple):
            table, outbox_id = outbox_ref
        else:
            table, outbox_id = "event_outbox", int(outbox_ref)
        if table in self._LEGACY_OUTBOX_TABLES:
            table = "event_outbox"
        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            conn.execute(
                f"""
                UPDATE {table}
                SET status = 'sent',
                    sent_at = ?,
                    sent_at_ms = ?,
                    last_attempt_at = ?,
                    last_attempt_ms = ?
                WHERE id = ?
                """,
                (now, int(time.time() * 1000), now, int(time.time() * 1000), outbox_id),
            )
            conn.commit()

    def _mark_outbox_retry_failed(self, outbox_ref, last_error: str) -> None:
        """outbox_ref: (table, row_id) tuple 또는 int (하위 호환)."""
        if not self.enable_store_and_forward or outbox_ref is None:
            return
        if isinstance(outbox_ref, tuple):
            table, outbox_id = outbox_ref
        else:
            table, outbox_id = "event_outbox", int(outbox_ref)
        if table in self._LEGACY_OUTBOX_TABLES:
            table = "event_outbox"
        with self._outbox_lock, self._outbox_connect() as conn:
            conn.execute(
                f"""
                UPDATE {table}
                SET retry_count = retry_count + 1,
                    last_attempt_at = ?,
                    last_attempt_ms = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (self._utc_now_iso(), int(time.time() * 1000), last_error[:1000], outbox_id),
            )
            conn.commit()
