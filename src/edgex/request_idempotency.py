"""동시 요청과 재시작 후 재수신의 물리 중복 실행을 방지한다."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Generic, Mapping, TypeVar

from src.services.command_identifiers import (
    build_device_command_id,
    resolve_logical_command_id,
)

ResultType = TypeVar("ResultType")


class RequestIdempotencyError(Exception):
    """재실행할 수 없는 요청의 표준 상태와 원인을 전달한다."""

    def __init__(self, error_code: str, status: str = "uncertain") -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.status = status


class RequestIdempotencyCache(Generic[ResultType]):
    """메모리에서 동시 실행을 합치고 영속 기록으로 재시작 경계를 보호한다."""

    def __init__(self, max_entries: int = 4096, database_path: str | Path | None = None,
                 wait_timeout_seconds: float = 30.0) -> None:
        self._max_entries = max(1, int(max_entries))
        self._wait_timeout_seconds = float(wait_timeout_seconds)
        if not math.isfinite(self._wait_timeout_seconds) or self._wait_timeout_seconds <= 0:
            raise ValueError("중복 요청 대기 제한시간은 유한한 양수여야 합니다")
        self._entries: OrderedDict[str, tuple[str, ResultType]] = OrderedDict()
        self._inflight: dict[str, tuple[str, Future[ResultType]]] = {}
        self._lock = Lock()
        # shortcut: 경로 없는 직접 생성은 메모리 전용, 운영 러너는 반드시 영속 경로를 전달한다.
        self._database_path = Path(database_path) if database_path is not None else None
        if self._database_path is not None:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self._database_path, timeout=5)) as conn, conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS device_requests (
                    request_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                )""")

    def _connect(self) -> sqlite3.Connection:
        # 실행 중 DB가 사라지면 새 빈 기록으로 진행하지 않고 출력을 보류한다.
        conn = sqlite3.connect(self._database_path.resolve().as_uri() + "?mode=rw", uri=True, timeout=5)
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _claim(self, request_id: str, fingerprint: str, request: Mapping[str, Any]) -> str | None:
        """실행 전에 원자적으로 선점하며 미완료 기록은 소유권을 다시 가져오지 않는다."""
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT request_hash, result_json FROM device_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None:
                # 과거 논리 ID는 물리 대상/재전송 ID가 바뀌어도 재실행 근거로 삼지 않는다.
                aliases = (resolve_logical_command_id(request_id, request.get("device_id")),
                           build_device_command_id(request.get("event_id"), request.get("device")))
                legacy = conn.execute(
                    "SELECT 1 FROM device_requests WHERE request_id IN (?, ?) AND request_hash = ''",
                    aliases,
                ).fetchone()
                stored_hash = "" if legacy else fingerprint
                conn.execute("INSERT INTO device_requests(request_id, request_hash) VALUES (?, ?)",
                             (request_id, stored_hash))
                if not legacy:
                    return None
                row = (stored_hash, None)
        # 보류된 실제 요청 ID도 commit한 뒤 반환해 실행 없는 조회/재시작에서 확인하게 한다.
        stored_hash, result_json = row
        if stored_hash == "":
            raise RequestIdempotencyError("legacy_execution_unconfirmed")
        if stored_hash != fingerprint:
            raise RequestIdempotencyError("request_id_conflict", "failed")
        if result_json is None:
            raise RequestIdempotencyError("execution_unconfirmed")
        return result_json

    def _finish(self, request_id: str, fingerprint: str, result: ResultType) -> None:
        """장치 응답을 확정 저장한 뒤에만 성공 결과를 반환한다."""
        with closing(self._connect()) as conn, conn:
            updated = conn.execute(
                "UPDATE device_requests SET result_json = ?, completed_at = CURRENT_TIMESTAMP "
                "WHERE request_id = ? AND request_hash = ? AND result_json IS NULL",
                (json.dumps(asdict(result), ensure_ascii=False, allow_nan=False), request_id, fingerprint),
            ).rowcount
            if updated != 1:
                raise RequestIdempotencyError("request_store_unavailable")

    def lookup(self, request_id: str) -> dict[str, Any]:
        """장치 호출·선점 없이 실행 이력만 읽으며 기록 없음도 재실행 허가로 쓰지 않는다."""
        result = {"request_id": request_id, "status": "unknown", "error_code": "request_not_found"}
        try:
            if self._database_path is not None:
                # 조회가 새 빈 DB를 만들어 이력 삭제를 숨기지 않게 읽기 전용으로 연다.
                with closing(sqlite3.connect(self._database_path.resolve().as_uri() + "?mode=ro",
                                             uri=True, timeout=5)) as conn:
                    row = conn.execute("SELECT request_hash, result_json FROM device_requests WHERE request_id=?",
                                       (request_id,)).fetchone()
                if row is not None:
                    if row[0] == "":
                        result.update(status="uncertain", error_code="legacy_execution_unconfirmed")
                    elif row[1] is None:
                        result.update(status="uncertain", error_code="execution_unconfirmed")
                    else:
                        stored = json.loads(row[1])
                        if not isinstance(stored, dict) or stored.get("request_id") != request_id:
                            raise ValueError("실행 이력 형식 불일치")
                        result = {key: stored[key] for key in
                                  ("request_id", "event_id", "device_id", "status", "error_code")}
                        result.update({key: stored[key] for key in
                                       ("execution_completed", "resource_idle", "completion_evidence")
                                       if key in stored})
                if result["status"] == "unknown" or result.get("error_code") == "execution_unconfirmed":
                    # 실행 전 대기/재시작 보류도 조회하되 수신함이 없는 구버전 DB는 그대로 읽는다.
                    with closing(sqlite3.connect(self._database_path.resolve().as_uri() + "?mode=ro",
                                                 uri=True, timeout=5)) as conn:
                        table = conn.execute("SELECT 1 FROM sqlite_master WHERE name='device_mqtt_inbox'").fetchone()
                        inbox = conn.execute("SELECT event_id,device_id,error FROM device_mqtt_inbox "
                                             "WHERE request_id=? AND is_status=0 ORDER BY sequence DESC LIMIT 1",
                                             (request_id,)).fetchone() if table else None
                    if inbox:
                        result.update(event_id=inbox[0], device_id=inbox[1], status="uncertain",
                                      error_code=inbox[2] or "device_execution_pending")
            else:
                with self._lock:
                    cached = self._entries.get(request_id)
                    if cached is not None:
                        result = asdict(cached[1])
                    elif request_id in self._inflight:
                        result.update(status="uncertain", error_code="execution_unconfirmed")
        except (sqlite3.Error, OSError, TypeError, ValueError, KeyError):
            result.update(status="uncertain", error_code="request_store_unavailable")
        return {**result, "durable": self._database_path is not None,
                "retry_allowed": False,
                "retry_reason": ("already_completed" if result["status"] in
                                 {"acknowledged", "simulated"} else "requires_reconciliation")}

    def complete(self, request_id: str, *, event_id: str, device_id: str,
                 result_type: type[ResultType], evidence: str) -> ResultType | None:
        """원래 ACK와 동일한 요청의 종료 증거를 확정 저장한 뒤 반환한다."""
        # 최초 ACK 저장 중에는 다음 유휴 확인에서 다시 시도해 메모리/DB 역전을 막는다.
        with self._lock:
            if request_id in self._inflight:
                return None
            cached = self._entries.get(request_id)
            if self._database_path is not None:
                with closing(self._connect()) as conn, conn:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute("SELECT result_json FROM device_requests WHERE request_id=?",
                                       (request_id,)).fetchone()
                    if row is None or row[0] is None:
                        return None
                    result = result_type(**json.loads(row[0]))
                    if (result.status, result.event_id, result.device_id) != ("acknowledged", event_id, device_id):
                        return None
                    result = replace(result, execution_completed=True, resource_idle=True,
                                     completion_evidence=evidence)
                    conn.execute("UPDATE device_requests SET result_json=? WHERE request_id=?",
                                 (json.dumps(asdict(result), ensure_ascii=False, allow_nan=False), request_id))
            else:
                if cached is None:
                    return None
                result = cached[1]
                if (result.status, result.event_id, result.device_id) != ("acknowledged", event_id, device_id):
                    return None
                result = replace(result, execution_completed=True, resource_idle=True,
                                 completion_evidence=evidence)
            if cached is not None:
                self._entries[request_id] = (cached[0], result)
            return result

    def run(
        self, request_id: str, execute: Callable[[], ResultType], *,
        request: Mapping[str, Any] | None = None, result_type: type[ResultType] | None = None,
    ) -> ResultType:
        """고정 내용이 같은 요청만 합치며 저장 오류나 불확실 실행은 재시도하지 않는다."""
        try:
            # 재전송마다 바뀌는 issued_at/attempt_id/source는 고정 명령 내용에서 제외한다.
            content = {key: request.get(key) for key in
                       ("event_id", "device", "device_id", "action", "payload", "dry_run")} if request else {}
            fingerprint = hashlib.sha256(json.dumps(
                content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            raise RequestIdempotencyError("invalid_payload", "failed") from None
        if self._database_path is not None and (request is None or result_type is None):
            raise ValueError("영속 실행에는 요청 내용과 결과 형식이 필요합니다")
        with self._lock:
            cached = self._entries.get(request_id)
            inflight = self._inflight.get(request_id)
            if (cached or inflight) and (cached or inflight)[0] != fingerprint:
                raise RequestIdempotencyError("request_id_conflict", "failed")
            if cached is not None:
                self._entries.move_to_end(request_id)
                return cached[1]
            owner = inflight is None
            if owner:
                pending: Future[ResultType] = Future()
                self._inflight[request_id] = (fingerprint, pending)
            else:
                pending = inflight[1]
        if not owner:
            try:
                return pending.result(timeout=self._wait_timeout_seconds)
            except TimeoutError:
                # 대기자만 반환하며 원래 실행의 소유권/미완료 기록은 해제하지 않는다.
                raise RequestIdempotencyError("response_wait_timeout") from None
        # 장치 호출과 DB 대기 중 공통 잠금을 놓아 다른 요청의 진행을 막지 않는다.
        try:
            try:
                stored = self._claim(request_id, fingerprint, request) if self._database_path is not None else None
                if stored is not None:
                    result = result_type(**json.loads(stored))
                else:
                    result = execute()
                    if self._database_path is not None:
                        # 통신 실패만으로 미출력을 증명할 수 없으므로 같은 ID의 자동 재출력을 막는다.
                        if getattr(result, "status", None) == "failed":
                            result = replace(result, status="uncertain")
                        self._finish(request_id, fingerprint, result)
            except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
                if self._database_path is None:
                    raise
                raise RequestIdempotencyError("request_store_unavailable") from exc
            with self._lock:
                if getattr(result, "status", None) in {"acknowledged", "simulated"}:
                    self._entries[request_id] = (fingerprint, result)
                    while len(self._entries) > self._max_entries:
                        self._entries.popitem(last=False)
                pending.set_result(result)
            return result
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(request_id, None)
