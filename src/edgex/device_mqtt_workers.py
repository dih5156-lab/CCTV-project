"""원본 저장 뒤 ACK하고 물리 장치별 MQTT 실행을 격리한다."""

import fcntl
import json
import logging
import math
import sqlite3
import time
from contextlib import closing
from threading import Event, Lock, Thread

from .command_contract import handle_mqtt_request

logger = logging.getLogger(__name__)


class DeviceMqttWorkers:
    def __init__(self, service, device_type, publish, *, timeout_seconds=30.0):
        self.service = service
        self.device_type = device_type
        self.publish = publish
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("장치 작업 제한시간은 유한한 양수여야 합니다")
        self.cache = service._request_results
        if self.cache._database_path is None:
            raise ValueError("장치별 작업자에는 영속 실행 이력 DB가 필요합니다")
        self._stop = Event()
        self._threads = []
        self._active = {}
        self._active_lock = Lock()
        self._lease = None
        with closing(self.cache._connect()) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS device_mqtt_inbox (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT, event_id TEXT, device_id TEXT,
                    topic TEXT NOT NULL, payload BLOB NOT NULL,
                    is_status INTEGER NOT NULL, received_at REAL NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('pending','running','complete','held')),
                    error TEXT, timed_out_at REAL
                );
                CREATE INDEX IF NOT EXISTS device_mqtt_due
                    ON device_mqtt_inbox(is_status,device_id,state,sequence);
                CREATE INDEX IF NOT EXISTS device_mqtt_request ON device_mqtt_inbox(request_id);
            """)

    def start(self):
        if self._lease is not None or any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("장치 수신 작업자가 이미 실행 중입니다")
        lease = self.cache._database_path.with_suffix(".mqtt.lock").open("a+b")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with closing(self.cache._connect()) as conn, conn:
                # 재시작 전 명령은 자동 재출력하지 않는다. 읽기 전용 조회만 다시 처리한다.
                conn.execute("UPDATE device_mqtt_inbox SET state='held',error='worker_recovery_requires_review' "
                             "WHERE is_status=0 AND state IN ('pending','running')")
                conn.execute("UPDATE device_mqtt_inbox SET state='pending' WHERE is_status=1 AND state='running'")
        except BaseException:
            lease.close()
            raise
        self._lease = lease
        self._stop.clear()
        self._threads = [Thread(target=self._run, args=(device_id,), daemon=True,
                                name=f"{self.device_type}-worker-{device_id}")
                         for device_id in (*self.service.device_ids, None)]
        for thread in self._threads:
            thread.start()

    def accept(self, topic, raw):
        """유효하지 않은 입력도 원본을 저장하며 반환 뒤에만 구독기가 ACK한다."""
        request = {}
        error = None
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("invalid_request")
            if (request.get("device") != self.device_type
                    or request.get("device_id") not in self.service.device_ids
                    or any(not isinstance(request.get(key), str) or not request[key].strip()
                           for key in ("request_id", "event_id", "action"))):
                raise ValueError("invalid_request")
        except (ValueError, TypeError, UnicodeError):
            error = "invalid_request"
            request = {}
        request_id = request.get("request_id")
        is_status = int(request.get("action") == "request_status")
        with closing(self.cache._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if not error and not is_status and conn.execute(
                "SELECT 1 FROM device_mqtt_inbox WHERE request_id=? AND error='worker_recovery_requires_review' LIMIT 1",
                (request_id,),
            ).fetchone():
                error = "worker_recovery_requires_review"
            conn.execute("""INSERT INTO device_mqtt_inbox
                (request_id,event_id,device_id,topic,payload,is_status,received_at,state,error)
                VALUES (?,?,?,?,?,?,?,?,?)""", (request_id, request.get("event_id"), request.get("device_id"),
                topic, raw, is_status, time.time(), "held" if error else "pending", error))

    def _run(self, device_id):
        while not self._stop.is_set():
            try:
                self._process_one(device_id)
            except Exception:
                # 장치 호출 중에는 DB/공통 잠금을 잡지 않아 다른 물리 대상과 수신을 유지한다.
                logger.exception("장치 수신 작업 보류: device_id=%s", device_id)
            self._stop.wait(0.25)

    def _process_one(self, device_id):
        with closing(self.cache._connect()) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM device_mqtt_inbox WHERE state='pending' AND is_status=? "
                               "AND (? IS NULL OR device_id=?) ORDER BY sequence LIMIT 1",
                               (int(device_id is None), device_id, device_id)).fetchone()
            if row is None or self._stop.is_set():
                return
            conn.execute("UPDATE device_mqtt_inbox SET state='running' WHERE sequence=?", (row["sequence"],))
        request = json.loads(row["payload"])
        with self._active_lock:
            self._active[row["sequence"]] = (time.monotonic(), request)
        try:
            result = handle_mqtt_request(self.service, request, self.device_type)
            with closing(self.cache._connect()) as conn, conn:
                conn.execute("UPDATE device_mqtt_inbox SET state='complete' WHERE sequence=?", (row["sequence"],))
            # 결과 전송 유실은 기존 request_status로 대조하며 출력 명령을 재호출하지 않는다.
            self.publish(result)
        except Exception:
            with closing(self.cache._connect()) as conn, conn:
                conn.execute("UPDATE device_mqtt_inbox SET state='held',error='device_execution_unconfirmed' "
                             "WHERE sequence=? AND state='running'", (row["sequence"],))
            raise
        finally:
            with self._active_lock:
                self._active.pop(row["sequence"], None)

    def expire_active(self):
        """느린 실행의 불확실성을 알리되 스레드/점유를 버리거나 대체하지 않는다."""
        with self._active_lock:
            active = list(self._active.items())
        for sequence, (started, request) in active:
            if time.monotonic() - started < self.timeout_seconds or request.get("action") == "request_status":
                continue
            with closing(self.cache._connect()) as conn, conn:
                changed = conn.execute("UPDATE device_mqtt_inbox SET error='device_worker_timeout',timed_out_at=? "
                                       "WHERE sequence=? AND state='running' AND timed_out_at IS NULL",
                                       (time.time(), sequence)).rowcount
            if changed:
                self.publish({key: request[key] for key in ("request_id", "event_id", "device_id")} |
                             {"status": "uncertain", "error_code": "device_worker_timeout"})

    def stop(self, timeout=5.0):
        self._stop.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self._threads):
            return False
        if self._lease is not None:
            self._lease.close()
            self._lease = None
        return True
