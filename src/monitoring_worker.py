"""Caller-owned monitoring delivery loop; injected transports perform network I/O."""

import math
import time

from .monitoring_contract import MonitoringContractError, parse_record
from .record_delivery import BrokerAck


class MonitoringDeliveryWorker:
    def __init__(
        self,
        store,
        sender,
        reader,
        *,
        namespace,
        binding,
        timeout=5,
        lease_seconds=30,
        ack_wait=10,
        clock=time.time,
        timings=None,
        commit_observer=None,
    ):
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or not math.isfinite(lease_seconds)
            or lease_seconds <= 2 * timeout + 2
            or not math.isfinite(ack_wait)
            or ack_wait <= 0
        ):
            raise ValueError(
                "lease must cover bounded status lookup, publish and cleanup"
            )
        self.store, self.sender, self.reader = store, sender, reader
        self.namespace, self.binding = namespace, binding
        self.timeout, self.lease_seconds, self.ack_wait = (
            timeout,
            lease_seconds,
            ack_wait,
        )
        self.clock = clock
        self.timings = timings
        self.commit_observer = commit_observer
        self.commit_observer_errors = 0

    def _call(self, stage, operation):
        return (
            operation()
            if self.timings is None
            else self.timings.measure(stage, operation)
        )

    def _failed(self, job, error):
        code = (
            error.code
            if isinstance(error, MonitoringContractError)
            else type(error).__name__
        )
        state = "uncertain"
        if code in {
            "invalid",
            "unsupported_version",
            "delivery_conflict",
            "record_conflict",
            "payload_hash_mismatch",
        }:
            state = "quarantined"
        elif code in {"unauthorized", "binding_mismatch"}:
            state = "blocked"
        self.store.fail_monitoring(job, code, state=state, now=self.clock())

    def drain_once(self, limit=100, *, stop_event=None):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        if stop_event is not None and stop_event.is_set():
            return 0
        self.store.enqueue_monitoring(
            self.namespace,
            self.binding,
            self.sender.endpoint,
            self.reader.endpoint,
            limit,
        )
        completed = 0
        for _ in range(limit):
            if stop_event is not None and stop_event.is_set():
                break
            job = self.store.claim_monitoring(
                self.binding,
                self.sender.endpoint,
                self.reader.endpoint,
                lease_seconds=self.lease_seconds,
                now=self.clock(),
            )
            if job is None:
                break
            try:
                record = parse_record(job["wire_payload"])
                if record.payload_sha256 != job["payload_sha256"]:
                    raise MonitoringContractError("payload_hash_mismatch")
                ack = self._call(
                    "consumer_lookup",
                    lambda: self.reader.get_delivery_status(
                        self.binding, job["job_id"], timeout=self.timeout
                    ),
                )
            except Exception as error:
                self._failed(job, error)
                continue
            if ack is not None:
                # Source DB errors must propagate, leaving the lease recoverable.
                stored = self._call(
                    "consumer_ack_commit",
                    lambda: self.store.complete_monitoring(job, ack, now=self.clock()),
                )
                completed += stored
                if stored and self.commit_observer is not None:
                    try:
                        self.commit_observer(
                            "consumer_ack_persisted_at",
                            job["first_ingest_id"],
                            received_at=job["received_at"],
                            delivery_id=job["job_id"],
                        )
                    except Exception:
                        self.commit_observer_errors += 1
                continue
            if not self.store.begin_publish(job, now=self.clock()):
                continue
            try:
                broker_ack = self._call(
                    "mqtt_publish", lambda: self.sender.send(job, timeout=self.timeout)
                )
                if (
                    not isinstance(broker_ack, BrokerAck)
                    or broker_ack.job_id != job["job_id"]
                ):
                    raise MonitoringContractError("broker_ack_mismatch")
            except Exception as error:
                self._failed(job, error)
                continue
            self.store.finish_publish(
                job, broker_ack, ack_wait=self.ack_wait, now=self.clock()
            )
        return completed
