"""물리 장치 점유와 위험/FIFO 순서를 보존하고 미확정 출력의 중첩을 막는다."""

import json
import logging
import time
from dataclasses import asdict
from threading import Event, Thread

from ..edgex.command_contract import build_command_request, build_command_topic
from .command_identifiers import build_transport_request_id
from .device_command_transport import DeviceCommand, DeviceCommandResult

_ACTIVE = "'dispatching','sent','acknowledged','uncertain'"
logger = logging.getLogger(__name__)


class ActionDeviceQueue:
    def __init__(self, *, expiry, repository, publish, jetson_id, topic_prefix):
        self.expiry = expiry
        self.store = expiry.store
        self.repository = repository
        self.publish = publish
        self.jetson_id = jetson_id
        self.topic_prefix = topic_prefix
        self._next_reconcile_at = 0.0
        self._resource_reconcile_at = {}
        self._workers = {}
        self._workers_stop = Event()
        with self.store._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS action_device_queue (
                    accepted_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE, command_id TEXT NOT NULL,
                    event_id TEXT NOT NULL, device_type TEXT NOT NULL, resource_key TEXT NOT NULL,
                    risk_rank INTEGER NOT NULL, within_risk_rank INTEGER NOT NULL,
                    command_json TEXT NOT NULL, context_json TEXT NOT NULL,
                    topic TEXT NOT NULL, wire_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('queued','dispatching','sent','acknowledged',
                        'complete','uncertain','expired','held')),
                    reason TEXT, result_json TEXT, updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS action_device_queue_owner
                    ON action_device_queue(resource_key)
                    WHERE state IN ('dispatching','sent','acknowledged','uncertain');
                CREATE INDEX IF NOT EXISTS action_device_queue_order
                    ON action_device_queue(state,risk_rank,within_risk_rank,accepted_sequence);
            """)

    def _topic(self, command, device_id):
        return build_command_topic(
            self.topic_prefix, self.jetson_id, command.device, device_id=device_id
        )

    def enqueue(self, command, context):
        """한 논리 명령의 물리 대상 전부를 같은 트랜잭션에 고정한다."""
        reason = self.expiry.check(context, command.device)
        if reason:
            return DeviceCommandResult(
                command.device,
                command.command_id,
                False,
                "expired" if reason == "expired" else "held",
                reason,
            )
        order = context["queue_order"]
        targets = context["targets"].get(command.device, [])
        if not targets:
            return DeviceCommandResult(
                command.device, command.command_id, False, "held", "no_target"
            )
        command_json = json.dumps(asdict(command), sort_keys=True, allow_nan=False)
        context_json = json.dumps(context, sort_keys=True, allow_nan=False)
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for device_id in targets:
                request_id = build_transport_request_id(command.command_id, device_id)
                old = connection.execute(
                    "SELECT command_json,context_json FROM action_device_queue WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                if old:
                    if (
                        old["command_json"] != command_json
                        or old["context_json"] != context_json
                    ):
                        raise ValueError("queue_content_conflict")
                    continue
                wire = build_command_request(
                    event_id=command.event_id,
                    request_id=request_id,
                    device=command.device,
                    device_id=device_id,
                    action=command.action,
                    payload=dict(command.payload),
                )
                connection.execute(
                    """INSERT INTO action_device_queue
                    (request_id,command_id,event_id,device_type,resource_key,risk_rank,within_risk_rank,
                     command_json,context_json,topic,wire_json,state,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,'queued',?)""",
                    (
                        request_id,
                        command.command_id,
                        command.event_id,
                        command.device,
                        device_id,
                        order["risk_rank"],
                        order["within_risk_rank"],
                        command_json,
                        context_json,
                        self._topic(command, device_id),
                        json.dumps(wire, sort_keys=True, allow_nan=False),
                        self.expiry.clock(),
                    ),
                )
        rows = self._rows(command.command_id)
        return DeviceCommandResult(
            command.device,
            command.command_id,
            True,
            "queued",
            device_ids=tuple(targets),
            request_ids=tuple(row["request_id"] for row in rows),
            target_results=tuple(self._target(row) for row in rows),
        )

    def _rows(self, command_id):
        with self.store._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM action_device_queue WHERE command_id=? ORDER BY accepted_sequence",
                    (command_id,),
                )
            ]

    @staticmethod
    def _target(row):
        status = {
            "dispatching": "pending",
            "sent": "accepted",
            "complete": "acknowledged",
        }.get(row["state"], row["state"])
        return {
            "request_id": row["request_id"],
            "event_id": row["event_id"],
            "device_id": row["resource_key"],
            "status": status,
            "error_code": row["reason"],
            "queue_state": row["state"],
            "resource_busy": row["state"]
            in {"dispatching", "sent", "acknowledged", "uncertain"},
            **(
                {"execution_completed": True, "resource_idle": True}
                if row["state"] == "complete"
                else {}
            ),
        }

    def _sync(self, command_id):
        rows = self._rows(command_id)
        command = json.loads(rows[0]["command_json"])
        return (
            self.repository.record_command(
                command_id,
                "device/" + command["device"],
                "queued",
                {
                    **command["payload"],
                    "event_id": command["event_id"],
                    "target_request_ids": [row["request_id"] for row in rows],
                    "device_results": [self._target(row) for row in rows],
                },
            )
            is not None
        )

    def recover(self):
        # 단일 소유 잠금 획득 후 호출한다. 재시작/시간 경과를 장치 유휴 증거로 쓰지 않는다.
        if any(worker.is_alive() for worker in self._workers.values()):
            raise RuntimeError("장치 작업자가 실행 중이므로 복구할 수 없습니다")
        self._workers.clear()
        self._resource_reconcile_at.clear()
        self._workers_stop.clear()
        with self.store._connect() as connection:
            commands = [row[0] for row in connection.execute(
                f"SELECT DISTINCT command_id FROM action_device_queue WHERE state IN ({_ACTIVE})")]
            connection.execute(
                f"UPDATE action_device_queue SET state='uncertain',reason='execution_unconfirmed' WHERE state IN ({_ACTIVE})"
            )
        for command_id in commands:
            self._sync(command_id)

    def _claim(self, request_id):
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return (
                connection.execute(
                    f"""UPDATE action_device_queue SET state='dispatching',updated_at=?
                WHERE request_id=? AND state='queued' AND NOT EXISTS (
                    SELECT 1 FROM action_device_queue busy WHERE busy.resource_key=action_device_queue.resource_key
                    AND busy.state IN ({_ACTIVE})) AND NOT EXISTS (
                    SELECT 1 FROM action_device_queue earlier WHERE earlier.resource_key=action_device_queue.resource_key
                    AND earlier.state='queued' AND (earlier.risk_rank,earlier.within_risk_rank,earlier.accepted_sequence)
                    < (action_device_queue.risk_rank,action_device_queue.within_risk_rank,action_device_queue.accepted_sequence))""",
                    (self.expiry.clock(), request_id),
                ).rowcount
                == 1
            )

    def _set(self, request_id, state, reason, *, previous):
        with self.store._connect() as connection:
            connection.execute(
                "UPDATE action_device_queue SET state=?,reason=?,updated_at=? WHERE request_id=? AND state=?",
                (state, reason, self.expiry.clock(), request_id, previous),
            )

    def dispatch_ready(self, resource_key=None):
        """현재 대기열을 위험/유형/FIFO로 선택하며 점유 중인 리소스는 건너뛴다."""
        with self.store._connect() as connection:
            rows = [
                dict(row)
                for row in connection.execute("""SELECT q.*, w.state AS intake_state
                FROM action_device_queue q LEFT JOIN action_execution_windows w ON q.event_id=w.event_id
                WHERE q.state='queued' AND (? IS NULL OR q.resource_key=?)
                ORDER BY risk_rank,within_risk_rank,accepted_sequence,request_id""", (resource_key, resource_key))
            ]
        for row in rows:
            if self._workers_stop.is_set():
                return
            command = DeviceCommand(**json.loads(row["command_json"]))
            context = json.loads(row["context_json"])
            reason = self.expiry.check(context, command.device, row["resource_key"])
            if not reason and row["intake_state"] != "complete":
                reason = "intake_unconfirmed"
            if not reason and row["topic"] != self._topic(command, row["resource_key"]):
                reason = "transport_changed"
            if reason:
                self._set(
                    row["request_id"],
                    "expired" if reason == "expired" else "held",
                    reason,
                    previous="queued",
                )
                self._sync(row["command_id"])
                continue
            if not self._claim(row["request_id"]):
                continue
            # 상태 저장 실패 때는 아직 발행하지 않았으므로 다시 대기할 수 있다.
            if not self._sync(row["command_id"]):
                self._set(
                    row["request_id"],
                    "queued",
                    "result_store_unavailable",
                    previous="dispatching",
                )
                continue
            reason = self.expiry.check(context, command.device, row["resource_key"])
            if self._workers_stop.is_set():
                self._set(row["request_id"], "queued", "worker_stopping", previous="dispatching")
                self._sync(row["command_id"])
                return
            if reason:
                self._set(
                    row["request_id"],
                    "expired" if reason == "expired" else "held",
                    reason,
                    previous="dispatching",
                )
            else:
                try:
                    accepted = self.publish(row["topic"], json.loads(row["wire_json"]))
                    state, reason = (
                        ("sent", None)
                        if accepted
                        else ("uncertain", "publish_unconfirmed")
                    )
                except Exception:
                    state, reason = "uncertain", "publish_unconfirmed"
                # 응답 콜백이 먼저 완료해도 늦은 발행 반환으로 점유를 되살리지 않는다.
                self._set(row["request_id"], state, reason, previous="dispatching")
            self._sync(row["command_id"])

    def observe_result(self, payload):
        """검증된 장치 결과에서 출력 종료·유휴 증거가 함께 있을 때만 점유를 푼다."""
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM action_device_queue WHERE request_id=?",
                (payload["request_id"],),
            ).fetchone()
            if (
                not row
                or row["event_id"] != payload["event_id"]
                or row["resource_key"] != payload["device_id"]
                or row["state"]
                not in {"dispatching", "sent", "acknowledged", "uncertain"}
            ):
                return
            completed = (
                payload["status"] == "acknowledged"
                and payload.get("execution_completed") is True
                and payload.get("resource_idle") is True
            )
            state = (
                "complete"
                if completed
                else "acknowledged"
                if payload["status"] == "acknowledged"
                else "uncertain"
            )
            connection.execute(
                "UPDATE action_device_queue SET state=?,reason=?,result_json=?,updated_at=? WHERE request_id=?",
                (
                    state,
                    None if completed else payload.get("error_code") or "completion_unconfirmed",
                    json.dumps(payload),
                    self.expiry.clock(),
                    row["request_id"],
                ),
            )
        self._sync(row["command_id"])

    def reconcile(self, resource_key=None):
        """미확정 점유를 저장된 결과와 대조하며 원래 출력 명령은 재전송하지 않는다."""
        now = time.monotonic()
        next_at = self._next_reconcile_at if resource_key is None else self._resource_reconcile_at.get(resource_key, 0)
        if now < next_at:
            return
        # 실패한 조회가 매 수신 반복마다 발행되지 않게 단조 시계로 주기를 제한한다.
        if resource_key is None:
            self._next_reconcile_at = now + 5.0
        else:
            self._resource_reconcile_at[resource_key] = now + 5.0
        with self.store._connect() as connection:
            rows = [dict(row) for row in connection.execute(
                f"SELECT * FROM action_device_queue WHERE state IN ({_ACTIVE}) AND (? IS NULL OR resource_key=?)",
                (resource_key, resource_key))]
        for row in rows:
            result = self.repository.get_device_result(row["command_id"], row["request_id"])
            if (result and result.get("status") == "acknowledged"
                    and result.get("execution_completed") is True and result.get("resource_idle") is True
                    and result.get("event_id") == row["event_id"]
                    and result.get("device_id") == row["resource_key"]):
                self.observe_result(result)
                continue

            # 정책이 바뀌어도 과거 실행은 조회하되 다른 통신 대상으로 질의하지 않는다.
            command = DeviceCommand(**json.loads(row["command_json"]))
            if row["topic"] != self._topic(command, row["resource_key"]):
                continue
            request = json.loads(row["wire_json"])
            request.update(action="request_status", payload={})
            try:
                self.publish(row["topic"], request)
            except Exception:
                # 조회 실패나 기록 없음은 유휴/미출력 증거가 아니므로 점유를 그대로 둔다.
                continue

    def refresh_workers(self):
        """서버가 확정한 물리 대상마다 작업자 하나만 두고 네트워크 대기를 격리한다."""
        if self._workers_stop.is_set():
            return
        with self.store._connect() as connection:
            resources = [row[0] for row in connection.execute(
                f"SELECT DISTINCT resource_key FROM action_device_queue WHERE state='queued' OR state IN ({_ACTIVE})")]
        self._workers = {key: worker for key, worker in self._workers.items()
                         if worker.is_alive() or key in resources}
        for resource in resources:
            if resource not in self._workers or not self._workers[resource].is_alive():
                worker = Thread(target=self._work_resource, args=(resource,),
                                name="action-device-" + resource, daemon=True)
                self._workers[resource] = worker
                worker.start()

    def _work_resource(self, resource):
        while not self._workers_stop.is_set():
            try:
                self.reconcile(resource)
                if not self._workers_stop.is_set():
                    self.dispatch_ready(resource)
                with self.store._connect() as connection:
                    active = connection.execute(
                        f"SELECT 1 FROM action_device_queue WHERE resource_key=? AND (state='queued' OR state IN ({_ACTIVE})) LIMIT 1",
                        (resource,),
                    ).fetchone()
                if not active:
                    return
            except Exception:
                # 저장/발행 실패를 다른 장치에 전파하지 않으며 선점 상태는 DB에 남긴다.
                logger.exception("장치 큐 작업 보류: device_id=%s", resource)
            self._workers_stop.wait(0.25)

    def request_workers_stop(self):
        """입력 처리 반환을 기다리는 동안에도 새 장치 선점을 중단한다."""
        self._workers_stop.set()

    def stop_workers(self, timeout=5.0):
        """실행 중인 발행을 취소했다고 가정하지 않고 전체 종료 대기 시간을 제한한다."""
        self.request_workers_stop()
        deadline = time.monotonic() + timeout
        for worker in self._workers.values():
            worker.join(max(0, deadline - time.monotonic()))
        return all(not worker.is_alive() for worker in self._workers.values())
