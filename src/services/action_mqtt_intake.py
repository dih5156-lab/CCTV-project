"""MQTT·REST 원본을 먼저 보존하고 Action 실행을 별도 작업자로 분리한다."""

import fcntl
import json
import logging
import os
from base64 import b64decode
from io import BufferedRandom
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

from ..ingest_store import IngestStore
from .action_device_queue import ActionDeviceQueue

logger = logging.getLogger(__name__)
_NAMESPACE = "action_mqtt"
_REST_NAMESPACE = "action_rest"


class ActionMqttIntake:
    def __init__(
        self, path: str, handler, *, capture_only: bool = False, expiry_policy=None, rest_handler=None
    ):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # 원본 이벤트가 들어가는 새 DB는 넓은 기본 umask와 무관하게 비공개로 만든다.
            descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        self.store = IngestStore(path)
        self.handler = handler
        self.rest_handler = rest_handler
        self.capture_only = capture_only
        self.expiry_policy = expiry_policy
        self.device_queue: ActionDeviceQueue | None = None
        self._admission_lock = Lock()
        self._stop = Event()
        self._wake = Event()
        self._worker: Thread | None = None
        self._lease: BufferedRandom | None = None
        self._session_id: str | None = None

    def start(self) -> None:
        if self._lease is not None:
            raise RuntimeError("Action 수신 작업자가 이미 실행 중입니다")
        lease = Path(str(self.store.path) + ".lock").open("a+b")
        try:
            # 복구 보류가 다른 프로세스의 진행 중 실행을 건드리지 않도록 단일 소유자를 고정한다.
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lease.close()
            raise RuntimeError("Action 수신 DB를 다른 작업자가 사용 중입니다") from None
        self._lease = lease
        self._session_id = uuid4().hex
        try:
            # 옵션을 끄는 것만으로 이미 점유한 장치를 새 경로가 우회하지 못하게 한다.
            if self.device_queue is None:
                with self.store._connect() as connection:
                    table = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='action_device_queue'").fetchone()
                    if table and connection.execute("SELECT 1 FROM action_device_queue WHERE state IN ('queued','dispatching','sent','acknowledged','uncertain') LIMIT 1").fetchone():
                        raise RuntimeError("미완료 물리 장치 큐가 있으므로 큐 옵션을 유지해야 합니다")
            if self.expiry_policy is not None:
                self.expiry_policy.recover()
            if self.device_queue is not None:
                self.device_queue.recover()
            with self.store._connect() as connection:
                # 정책을 저장하지 않은 구형 입력과 실행 중 종료는 자동 재출력하지 않는다.
                connection.execute(
                    "UPDATE sensor_ingest_state SET last_error=CASE stage "
                    "WHEN 'ready' THEN 'execution_unconfirmed' ELSE 'recovery_requires_review' END, "
                    "stage='quarantined' WHERE (stage='ready' OR (stage='pending' AND "
                    "(?=0 OR projection_json IS NULL))) AND ingest_id IN "
                    "(SELECT ingest_id FROM ingest_receipts WHERE source_namespace IN (?,?))",
                    (int(self.expiry_policy is not None), _NAMESPACE, _REST_NAMESPACE),
                )
            self._stop.clear()
            self._wake.clear()
            if not self.capture_only:
                self._worker = Thread(
                    target=self._run, name="action-mqtt-worker", daemon=True
                )
                self._worker.start()
        except BaseException:
            self._release()
            raise

    def accept(self, topic: str, payload: bytes) -> str:
        # 반환 후 구독기가 MQTT ACK를 보낸다. 저장 실패는 예외로 전달해 ACK를 금지한다.
        return self._accept(_NAMESPACE, topic, payload)

    def accept_rest(self, topic: str, raw: bytes, payload: bytes) -> str:
        """HTTP 원본은 그대로 보존하고 topic을 뺀 이벤트로 기존 만료 계약을 적용한다."""
        return self._accept(_REST_NAMESPACE, topic, raw, snapshot_payload=payload)

    def _accept(self, namespace, topic, payload, *, snapshot_payload=None) -> str:
        with self._admission_lock:
            if namespace == _REST_NAMESPACE and (self._lease is None or self._stop.is_set()):
                raise RuntimeError("REST 영속 접수가 시작되지 않았거나 종료 중입니다")
            # 기존 바이트 보존 형식을 재사용하고 실제 입력 경로는 namespace로 구분한다.
            ingest_id = self.store.accept_mqtt(namespace, topic, payload)
            if self.expiry_policy is not None:
                receipt = self.store.get(ingest_id)
                plan = self.expiry_policy.snapshot(
                    topic, payload if snapshot_payload is None else snapshot_payload, receipt["received_at"]
                )
                if plan.get("kind") == "control":
                    plan["session_id"] = self._session_id
                with self.store._connect() as connection:
                    connection.execute(
                        "UPDATE sensor_ingest_state SET projection_json=? WHERE ingest_id=? AND stage='pending'",
                        (json.dumps(plan, allow_nan=False), ingest_id),
                    )
        self._wake.set()
        return ingest_id

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._wake.clear()
                try:
                    for ingest_id in self.store.pending(namespace=(_NAMESPACE, _REST_NAMESPACE)):
                        if self._stop.is_set():
                            break
                        self._process(ingest_id)
                    if self.device_queue is not None and not self._stop.is_set():
                        self.device_queue.refresh_workers()
                except Exception:
                    # 저장 장애 중에는 원본과 ready 선점을 유지하며 장치 호출을 반복하지 않는다.
                    logger.exception("Action 수신 작업자 저장 실패")
                self._wake.wait(0.25)
        finally:
            # 장치 발행이 아직 반환되지 않았으면 단일 소유 잠금을 유지한다.
            if self.device_queue is None or self.device_queue.stop_workers():
                self._release()

    def _process(self, ingest_id: str) -> None:
        with self._admission_lock:
            receipt = self.store.get(ingest_id)
            if receipt["stage"] == "ready":
                self.store.quarantine(ingest_id, "execution_unconfirmed")
                return
            if receipt["stage"] != "pending":
                return
            try:
                plan = json.loads(receipt["projection_json"] or "{}")
                if not isinstance(plan, dict):
                    raise ValueError("invalid_snapshot")
            except ValueError:
                self.store.quarantine(ingest_id, "held:invalid_snapshot")
                return
            if (
                self.expiry_policy is not None
                and plan.get("kind") == "control"
                and plan.get("session_id") != self._session_id
            ):
                self.store.quarantine(
                    ingest_id, "held:control_recovery_requires_review"
                )
                return
            if self.expiry_policy is not None and plan.get("kind") == "event":
                reason = self.expiry_policy.claim(plan)
                if reason:
                    state = "expired" if reason == "expired" else "held"
                    self.expiry_policy.finish(plan, state, reason)
                    # 원본은 보존하고 처리 단계와 출력 판정 상태를 구별해 기록한다.
                    self.store.quarantine(ingest_id, state + ":" + reason)
                    return
            elif self.expiry_policy is not None and plan.get("kind") not in {
                "control",
                "result",
            }:
                self.store.quarantine(ingest_id, "held:execution_context_missing")
                return
            self.store.prepare(ingest_id, json.dumps(plan))
        try:
            raw = receipt["raw"]
            args = (raw["topic"], b64decode(raw["payload_base64"], validate=True))
            handler = self.rest_handler if receipt["source_namespace"] == _REST_NAMESPACE else self.handler
            if self.expiry_policy is not None:
                handler(
                    *args,
                    execution_context=plan if plan.get("kind") == "event" else None,
                )
            else:
                handler(*args)
        except Exception as exc:
            if self.expiry_policy is not None:
                self.expiry_policy.finish(plan, "uncertain", "execution_unconfirmed")
            self.store.quarantine(
                ingest_id, "execution_unconfirmed:" + type(exc).__name__
            )
            logger.exception("Action 수신 처리 보류: ingest_id=%s", ingest_id)
            return
        # complete는 입력 처리 반환이며 장치 ACK/실물 성공은 기존 명령 결과 저장소에서 판정한다.
        if self.expiry_policy is not None:
            self.expiry_policy.finish(plan, "complete")
        self.store.complete(ingest_id)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self.device_queue is not None:
            self.device_queue.request_workers_stop()
        if self._worker is not None:
            self._worker.join(timeout=5)
            if self._worker.is_alive():
                # 실행 소유자가 끝날 때까지 잠금을 유지해 중복 작업자를 허용하지 않는다.
                raise RuntimeError("Action 수신 작업자가 아직 종료되지 않았습니다")
            self._worker = None
        if self.device_queue is not None and not self.device_queue.stop_workers():
            raise RuntimeError("장치 작업자가 아직 종료되지 않았습니다")
        self._release()

    def _release(self) -> None:
        if self._lease is not None:
            self._lease.close()
            self._lease = None
