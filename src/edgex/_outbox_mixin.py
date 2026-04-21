"""
EdgeX Outbox Mixin — SQLite store-and-forward 로직

CCTVDeviceService 에 믹스인으로 포함되며, 전송 실패 이벤트를 SQLite 에
저장하고 복구 후 재전송하는 기능을 담당한다.
각 메서드는 self.enable_store_and_forward, self.outbox_db_path,
self.outbox_flush_batch_size, self._outbox_lock 에 의존한다.
"""

import json
import logging
import sqlite3
import threading
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

    # ── 카테고리 → 테이블 매핑 ───────────────────────────────────────────────
    _CATEGORY_TABLE_MAP: Dict[str, str] = {
        "person": "detection_outbox",
        "camera": "detection_outbox",
        "sensor": "sensor_outbox",
        "zone":   "zone_outbox",
    }
    _OUTBOX_TABLES = ("detection_outbox", "sensor_outbox", "zone_outbox")

    @classmethod
    def _table_for_category(cls, category: str) -> str:
        return cls._CATEGORY_TABLE_MAP.get(category, "detection_outbox")

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
                conn.execute("PRAGMA journal_mode=OFF")
                conn.execute("PRAGMA synchronous=OFF")

                for table in self._OUTBOX_TABLES:
                    conn.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {table} (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            event_id TEXT,
                            destination_name TEXT NOT NULL DEFAULT 'edgex-core',
                            camera_id TEXT NOT NULL,
                            data_category TEXT NOT NULL DEFAULT 'camera',
                            payload_json TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            expire_at TEXT,
                            last_attempt_at TEXT,
                            sent_at TEXT,
                            retry_count INTEGER NOT NULL DEFAULT 0,
                            status TEXT NOT NULL DEFAULT 'pending',
                            last_error TEXT
                        )
                        """
                    )
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{table}_status_id "
                        f"ON {table}(status, id)"
                    )
                    conn.execute(
                        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_event_dest "
                        f"ON {table}(event_id, destination_name)"
                    )
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{table}_category "
                        f"ON {table}(data_category, status)"
                    )

                # 기존 DB에 data_category 컬럼 없으면 마이그레이션
                existing = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(detection_outbox)")
                }
                if "data_category" not in existing:
                    conn.execute(
                        "ALTER TABLE detection_outbox "
                        "ADD COLUMN data_category TEXT NOT NULL DEFAULT 'camera'"
                    )
                    logger.info("detection_outbox: data_category 컬럼 추가 완료 (마이그레이션)")
                if "event_id" not in existing:
                    conn.execute(
                        "ALTER TABLE detection_outbox ADD COLUMN event_id TEXT"
                    )
                if "destination_name" not in existing:
                    conn.execute(
                        "ALTER TABLE detection_outbox ADD COLUMN destination_name TEXT NOT NULL DEFAULT 'edgex-core'"
                    )
                if "expire_at" not in existing:
                    conn.execute(
                        "ALTER TABLE detection_outbox ADD COLUMN expire_at TEXT"
                    )

                for table in self._OUTBOX_TABLES:
                    cols = {
                        row[1]
                        for row in conn.execute(f"PRAGMA table_info({table})")
                    }
                    if "event_id" not in cols:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN event_id TEXT")
                    if "destination_name" not in cols:
                        conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN destination_name TEXT NOT NULL DEFAULT 'edgex-core'"
                        )
                    if "expire_at" not in cols:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN expire_at TEXT")
                    conn.execute(
                        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_event_dest "
                        f"ON {table}(event_id, destination_name)"
                    )

                # 기존 detection_outbox에서 sensor/zone 데이터를 새 테이블로 이관
                for cat, target in (("sensor", "sensor_outbox"), ("zone", "zone_outbox")):
                    migrated = conn.execute(
                        f"""
                        INSERT INTO {target}
                            (camera_id, data_category, payload_json,
                             created_at, last_attempt_at, sent_at,
                             retry_count, status, last_error)
                        SELECT camera_id, data_category, payload_json,
                               created_at, last_attempt_at, sent_at,
                               retry_count, status, last_error
                        FROM detection_outbox
                        WHERE data_category = ?
                        """,
                        (cat,),
                    ).rowcount
                    if migrated:
                        conn.execute(
                            "DELETE FROM detection_outbox WHERE data_category = ?",
                            (cat,),
                        )
                        logger.info(
                            "detection_outbox → %s 마이그레이션: %d건 이관",
                            target, migrated,
                        )

                conn.commit()
        except Exception as exc:
            logger.warning("[Outbox] DB 초기화 실패 — store-and-forward 비활성화: %s", exc)
            self.enable_store_and_forward = False
        finally:
            conn.close()

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
        table = self._table_for_category(category)

        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            event_id = get_payload_event_id(event_data)
            expire_at = datetime.now(timezone.utc).replace(
                microsecond=0
            ).isoformat()
            conn.execute(
                f"""
                INSERT OR IGNORE INTO {table} (
                    event_id, destination_name, camera_id, data_category, payload_json,
                    created_at, expire_at, last_attempt_at, retry_count, status, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, datetime(?, '+{_OUTBOX_TTL_DAYS} days'), ?, ?, ?, ?)
                """,
                (
                    event_id,
                    "edgex-core",
                    camera_id,
                    category,
                    json.dumps(event_data, ensure_ascii=False),
                    now,
                    now,
                    now,
                    1,
                    "pending",
                    last_error[:1000],
                ),
            )
            conn.commit()
            logger.debug(
                "[Outbox:%s] 저장: camera=%s category=%s type=%s",
                table, camera_id, category, event_type,
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
        table = self._table_for_category(category)

        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            event_id = get_payload_event_id(event_data)
            cur = conn.execute(
                f"""
                INSERT OR IGNORE INTO {table} (
                    event_id, destination_name, camera_id, data_category, payload_json,
                    created_at, expire_at, last_attempt_at, retry_count, status
                ) VALUES (?, ?, ?, ?, ?, ?, datetime(?, '+{_OUTBOX_TTL_DAYS} days'), ?, ?, ?)
                """,
                (
                    event_id,
                    "edgex-core",
                    camera_id,
                    category,
                    json.dumps(event_data, ensure_ascii=False),
                    now,
                    now,
                    now,
                    0,
                    "pending",
                ),
            )
            conn.commit()
            if cur.lastrowid is None or cur.lastrowid == 0:
                row = conn.execute(
                    f"SELECT id FROM {table} WHERE event_id = ? AND destination_name = ?",
                    (event_id, "edgex-core"),
                ).fetchone()
                if row:
                    return (table, int(row[0]))
            logger.debug(
                "[Outbox:%s] pending 저장: camera=%s category=%s type=%s id=%s",
                table, camera_id, category, event_type, cur.lastrowid,
            )
            return (table, cur.lastrowid)

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
        tables = (
            [self._table_for_category(data_category)]
            if data_category
            else list(self._OUTBOX_TABLES)
        )

        all_rows: list = []
        with self._outbox_lock, self._outbox_connect() as conn:
            conn.row_factory = sqlite3.Row
            for table in tables:
                if data_category:
                    rows = conn.execute(
                        f"""
                        SELECT id, camera_id, data_category, payload_json,
                               event_id, destination_name, created_at, expire_at,
                               last_attempt_at, retry_count, status, last_error
                        FROM {table}
                        WHERE status = 'pending' AND data_category = ?
                        ORDER BY id ASC
                        LIMIT ?
                        """,
                        (data_category, fetch_limit - len(all_rows)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        f"""
                        SELECT id, camera_id, data_category, payload_json,
                               event_id, destination_name, created_at, expire_at,
                               last_attempt_at, retry_count, status, last_error
                        FROM {table}
                        WHERE status = 'pending'
                        ORDER BY id ASC
                        LIMIT ?
                        """,
                        (fetch_limit - len(all_rows),),
                    ).fetchall()
                for r in rows:
                    all_rows.append((table, r))
                if len(all_rows) >= fetch_limit:
                    break

        pending: List[Dict[str, Any]] = []
        for table, row in all_rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {}
            pending.append(
                {
                    "id": row["id"],
                    "_table": table,
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
            for table in self._OUTBOX_TABLES:
                cur = conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'expired',
                        last_attempt_at = ?,
                        last_error = COALESCE(last_error, 'ttl expired')
                    WHERE status = 'pending'
                      AND expire_at IS NOT NULL
                      AND datetime(expire_at) <= datetime(?)
                    """,
                    (now, now),
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
            table, outbox_id = "detection_outbox", int(outbox_ref)
        with self._outbox_lock, self._outbox_connect() as conn:
            now = self._utc_now_iso()
            conn.execute(
                f"""
                UPDATE {table}
                SET status = 'sent', sent_at = ?, last_attempt_at = ?
                WHERE id = ?
                """,
                (now, now, outbox_id),
            )
            conn.commit()

    def _mark_outbox_retry_failed(self, outbox_ref, last_error: str) -> None:
        """outbox_ref: (table, row_id) tuple 또는 int (하위 호환)."""
        if not self.enable_store_and_forward or outbox_ref is None:
            return
        if isinstance(outbox_ref, tuple):
            table, outbox_id = outbox_ref
        else:
            table, outbox_id = "detection_outbox", int(outbox_ref)
        with self._outbox_lock, self._outbox_connect() as conn:
            conn.execute(
                f"""
                UPDATE {table}
                SET retry_count = retry_count + 1,
                    last_attempt_at = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (self._utc_now_iso(), last_error[:1000], outbox_id),
            )
            conn.commit()
