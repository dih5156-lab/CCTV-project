"""ActionBridge SQLite 이벤트 저장소."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from ..canonical_event import (
    get_payload_camera_id,
    get_payload_confidence,
    get_payload_event_id,
    get_payload_event_type,
    get_payload_severity,
)
from ..event_priority import event_priority, event_risk_level, event_risk_score
from ..time_utils import now_kst_iso

logger = logging.getLogger(__name__)


def _device_result_status(results: list[Dict]) -> str:
    """미응답을 완료로 세지 않고 고정 대상의 부분 실패를 집계한다."""
    statuses = {row["status"] for row in results}
    # 발행하지 않은 만료/보류를 성공으로 합치지 않고 물리 대상별 차이를 유지한다.
    if statuses in ({"expired"}, {"held"}):
        return next(iter(statuses))
    if "uncertain" in statuses or ("failed" in statuses and statuses & {"pending", "accepted"}):
        return "uncertain"
    if statuses & {"pending", "accepted"}:
        return "pending"
    if "queued" in statuses:
        return "queued"
    if statuses == {"acknowledged"}:
        return "acknowledged"
    if statuses == {"simulated"}:
        return "simulated"
    if "failed" in statuses:
        return "failed" if statuses == {"failed"} else "partial_failure"
    if statuses & {"expired", "held"}:
        return "partial_failure"
    return "uncertain"


class _EventRepo:
    """SQLite 이벤트 CRUD 전담 헬퍼."""

    def __init__(self, db_path: Path, response_timeout_seconds: float = 30.0) -> None:
        self.db_path = db_path
        self._response_timeout_seconds = float(response_timeout_seconds)
        if not math.isfinite(self._response_timeout_seconds) or self._response_timeout_seconds <= 0:
            raise ValueError("장치 응답 제한시간은 유한한 양수여야 합니다")

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id     TEXT    NOT NULL,
                    received_at  TEXT    NOT NULL,
                    topic        TEXT    NOT NULL,
                    camera_id    TEXT,
                    event_type   TEXT,
                    confidence   REAL,
                    severity     TEXT,
                    alarm_played INTEGER DEFAULT 0,
                    http_sent    INTEGER DEFAULT 0,
                    payload_json TEXT    NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(action_events)")}
            if "event_id" not in columns:
                conn.execute("ALTER TABLE action_events ADD COLUMN event_id TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_action_events_event_id ON action_events(event_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_events_camera_id ON action_events(camera_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_events_received_at ON action_events(received_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_commands (
                    command_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    site_id TEXT,
                    event_id TEXT,
                    requested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            # 재시작/재수신으로 대기 시간을 늘리지 않도록 최초 발행 기한을 별도 보존한다.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(action_commands)")}
            if "response_deadline" not in columns:
                conn.execute("ALTER TABLE action_commands ADD COLUMN response_deadline REAL")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_action_commands_deadline "
                         "ON action_commands(response_deadline) WHERE response_deadline IS NOT NULL")
            for command_id, requested_at, raw in conn.execute(
                "SELECT command_id, requested_at, payload_json FROM action_commands "
                "WHERE response_deadline IS NULL AND status IN ('pending', 'accepted', 'uncertain')"
            ).fetchall():
                try:
                    if any(result.get("status") in {"pending", "accepted"}
                           for result in json.loads(raw).get("device_results", [])):
                        deadline = datetime.fromisoformat(requested_at).timestamp() + self._response_timeout_seconds
                        conn.execute("UPDATE action_commands SET response_deadline=? WHERE command_id=?",
                                     (deadline, command_id))
                except (TypeError, ValueError):
                    logger.warning("기존 명령의 응답 기한 복구 불가: %s", command_id)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_commands_updated_at ON action_commands(updated_at)"
            )
            conn.commit()

    def record_command(self, command_id: str, topic: str, status: str, payload: Dict, message: str = "") -> Dict | None:
        """명령별 장치 결과를 원자적으로 병합하고 실제 저장 상태를 반환한다."""
        now = now_kst_iso()
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                # 읽기와 병합을 같은 쓰기 트랜잭션에 묶어 동시 응답의 유실을 막는다.
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT payload_json, response_deadline, requested_at FROM action_commands "
                                   "WHERE command_id=?", (command_id,)).fetchone()
                previous = json.loads(row[0]) if row else {}
                response_deadline = row[1] if row else None
                incoming = payload.get("device_results", [])
                if topic.startswith("edgex/results/") and payload.get("request_id") and payload.get("device_id"):
                    incoming = [payload]
                if incoming or previous.get("device_results"):
                    expected = previous.get("target_request_ids", payload.get("target_request_ids"))
                    if (previous.get("event_id") and payload.get("event_id")
                            and previous["event_id"] != payload["event_id"]):
                        logger.warning("명령 이벤트 ID 충돌: %s", command_id)
                        return None
                    if (expected is not None and payload.get("target_request_ids") is not None
                            and set(expected) != set(payload["target_request_ids"])):
                        logger.warning("명령 대상 집합 변경 거부: %s", command_id)
                        return None
                    results = {value["request_id"]: value for value in previous.get("device_results", [])}
                    for value in incoming:
                        request_id = value["request_id"]
                        if expected is not None and request_id not in expected:
                            logger.warning("등록되지 않은 장치 결과 무시: %s", request_id)
                            return None
                        current = results.get(request_id, {})
                        if current.get("device_id") and current["device_id"] != value.get("device_id"):
                            logger.warning("요청의 물리 장치 ID 충돌: %s", request_id)
                            return None
                        # 늦은 발행 상태나 실패 응답이 이미 확인한 성공을 지우지 않게 한다.
                        if current.get("status") == "acknowledged":
                            # 명령 ACK는 유지하되 물리 자원 점유/종료 증거는 별도로 갱신한다.
                            current.update({key: value[key] for key in ("queue_state", "resource_busy") if key in value})
                            if value["status"] != "acknowledged":
                                continue
                        if (value["status"] == "queued" and current.get("status") not in {None, "queued"}) or (
                            current.get("status") in {"failed", "uncertain", "simulated", "expired", "held"}
                            and value["status"] in {"queued", "pending", "accepted"}
                        ):
                            continue
                        # 응답이 늦어도 기한 초과 이력은 남기고 현재 결과만 갱신한다.
                        results[request_id] = {**current, **value}
                    for result in results.values():
                        # 발행 실패/이력 없음은 실제 미출력 증거가 아니므로 자동 재시도를 열지 않는다.
                        result["retry_allowed"] = False
                        result["retry_reason"] = ("already_completed" if result["status"] in
                                                  {"acknowledged", "simulated"} else "requires_reconciliation")
                    payload = {**previous, **payload, "device_results": list(results.values())}
                    if expected is not None:
                        payload["target_request_ids"] = expected
                    status = _device_result_status(payload["device_results"])
                    waiting = any(value["status"] in {"pending", "accepted"} for value in results.values())
                    if waiting and response_deadline is None:
                        requested_at = row[2] if row else now
                        # 큐 대기 시간을 장치 응답 제한시간으로 소비하지 않는다.
                        old_results = previous.get("device_results", [])
                        if any(value["status"] == "queued" for value in old_results) and not any(
                            value["status"] in {"pending", "accepted"} for value in old_results
                        ):
                            requested_at = now
                        response_deadline = datetime.fromisoformat(requested_at).timestamp() + self._response_timeout_seconds
                    elif not waiting:
                        response_deadline = None
                conn.execute(
                    """
                    INSERT INTO action_commands
                        (command_id, topic, status, message, site_id, event_id,
                         requested_at, updated_at, payload_json, response_deadline)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(command_id) DO UPDATE SET
                        status=excluded.status, message=excluded.message,
                        event_id=COALESCE(action_commands.event_id, excluded.event_id),
                        updated_at=excluded.updated_at, payload_json=excluded.payload_json,
                        response_deadline=excluded.response_deadline
                    """,
                    (
                        command_id, topic, status, message,
                        payload.get("site_id"), payload.get("event_id"),
                        now, now, json.dumps(payload, ensure_ascii=False), response_deadline,
                    ),
                )
                conn.commit()
                return {"status": status, "payload": payload}
        except sqlite3.Error as exc:
            logger.error("명령 상태 저장 오류: %s", exc)

    def expire_command_responses(self) -> List[Dict]:
        """기한이 지난 미응답만 영속 보류하며 장치 실행이나 명령 재발행은 하지 않는다."""
        now = now_kst_iso()
        updates = []
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                # 주기 점검의 연결은 즉시 닫고 실제 응답과 같은 잠금으로 완료 결과를 보호한다.
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    "SELECT command_id, payload_json FROM action_commands "
                    "WHERE response_deadline <= ?", (datetime.fromisoformat(now).timestamp(),),
                ).fetchall()
                for command_id, raw in rows:
                    payload = json.loads(raw)
                    timed_out = []
                    for result in payload.get("device_results", []):
                        if result["status"] in {"pending", "accepted"}:
                            result.update(status="uncertain", error_code="response_timeout",
                                          response_timed_out_at=now, retry_allowed=False,
                                          retry_reason="requires_reconciliation")
                            timed_out.append(dict(result))
                    if not timed_out:
                        continue
                    status = _device_result_status(payload["device_results"])
                    conn.execute(
                        "UPDATE action_commands SET status=?, message=?, updated_at=?, payload_json=?, "
                        "response_deadline=NULL WHERE command_id=?",
                        (status, "response_timeout", now, json.dumps(payload, ensure_ascii=False), command_id),
                    )
                    updates.extend({**result, "command_id": command_id, "command_status": status,
                                    "device_results": payload["device_results"]} for result in timed_out)
                conn.commit()
            return updates
        except (sqlite3.Error, TypeError, ValueError) as exc:
            logger.error("장치 미응답 상태 저장 오류: %s", exc)
            return []

    def get_device_result(self, command_id: str, request_id: str) -> Dict | None:
        """이미 저장된 개별 응답을 조회해 큐 저장 실패 후 결과를 다시 대조한다."""
        # DB 소실을 빈 이력으로 숨기지 않도록 조회에서는 새 파일 생성을 금지한다.
        with closing(sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            row = conn.execute("SELECT payload_json FROM action_commands WHERE command_id=?",
                               (command_id,)).fetchone()
        if row is None:
            return None
        return next((result for result in json.loads(row[0]).get("device_results", [])
                     if result.get("request_id") == request_id), None)

    def list_commands(self, limit: int = 50) -> List[Dict]:
        safe_limit = max(1, min(int(limit), 200))
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT command_id, topic, status, message, site_id, event_id,
                              requested_at, updated_at, payload_json, response_deadline
                       FROM action_commands ORDER BY updated_at DESC LIMIT ?""",
                    (safe_limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.error("명령 상태 조회 오류: %s", exc)
            return []
        result = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                payload = {}
            result.append({
                "command_id": row["command_id"], "topic": row["topic"],
                "status": row["status"], "message": row["message"],
                "site_id": row["site_id"], "event_id": row["event_id"],
                "requested_at": row["requested_at"], "updated_at": row["updated_at"],
                "payload": payload,
                "response_deadline": (datetime.fromtimestamp(row["response_deadline"], timezone.utc).isoformat()
                                      if row["response_deadline"] is not None else None),
            })
        return result

    def save(
        self,
        topic: str,
        payload: Dict,
        alarm_played: bool,
        http_sent: bool,
    ) -> None:
        try:
            event_id = get_payload_event_id(payload)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO action_events
                        (event_id, received_at, topic, camera_id, event_type, confidence,
                         severity, alarm_played, http_sent, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        now_kst_iso(),
                        topic,
                        get_payload_camera_id(payload),
                        get_payload_event_type(payload),
                        get_payload_confidence(payload),
                        get_payload_severity(payload),
                        int(alarm_played),
                        int(http_sent),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error("DB 저장 오류: %s", exc)

    def list_recent(self, limit: int = 20) -> List[Dict]:
        """최근 Action Layer 처리 이력을 최신순으로 반환한다."""
        safe_limit = max(1, min(int(limit), 100))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT id, event_id, received_at, topic, camera_id, event_type,
                           confidence, severity, alarm_played, http_sent, payload_json
                    FROM action_events
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.error("DB 조회 오류: %s", exc)
            return []

        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        return {
            "id": row["id"],
            "event_id": row["event_id"],
            "received_at": row["received_at"],
            "topic": row["topic"],
            "camera_id": row["camera_id"],
            "event_type": row["event_type"],
            "confidence": row["confidence"],
            "severity": row["severity"],
            "alarm_played": bool(row["alarm_played"]),
            "http_sent": bool(row["http_sent"]),
            "payload": payload,
            "priority": event_priority(payload),
            "risk_level": event_risk_level(payload),
            "risk_score": event_risk_score(payload),
        }
