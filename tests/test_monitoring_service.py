"""Service cancellation leaves unfinished work durable and resumable."""

import sqlite3
import threading
import time

import pytest

from src.monitoring_metrics import CommitObservations
from src.monitoring_service import MonitoringService, build_local_service
from tests.test_monitoring_delivery import seed
from tests.test_monitoring_worker import worker_setup


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.01)
    pytest.fail("condition did not become true before timeout")


def test_start_close_restart_and_ack_commit_observation(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    worker.clock, worker.ack_wait = time.time, 0.001
    observations = CommitObservations(clock_sync="confirmed")
    worker.commit_observer = observations.committed
    service = MonitoringService([worker], interval=0.01)
    service.start()
    try:
        original = service._thread
        service.start()
        assert service._thread is original
        wait_until(lambda: service.status()["completed"] == 1)
    finally:
        service.close()
    assert not service.status()["running"]
    seed(store.path, "evt-2")
    service.start()
    try:
        wait_until(lambda: service.status()["completed"] == 2)
    finally:
        service.close()
    assert observations.report()["latency"]["valid"] == 2
    for sample in observations.report()["samples"]:
        assert store.get_monitoring(sample["delivery_id"])["state"] == "stored"


def test_stop_finishes_current_send_but_does_not_claim_next(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    seed(store.path, "evt-2")
    worker.clock = time.time
    entered, release = threading.Event(), threading.Event()
    original_send = broker.send

    def blocked_send(job, *, timeout):
        entered.set()
        assert release.wait(5)
        return original_send(job, timeout=timeout)

    broker.send = blocked_send
    service = MonitoringService([worker], interval=0.01)
    service.start()
    try:
        assert entered.wait(3)
        with pytest.raises(TimeoutError):
            service.close(timeout=0.02)
        assert service.status()["state"] == "stopping"
        with pytest.raises(RuntimeError, match="not stopped"):
            service.start()
    finally:
        release.set()
        service.close()
    assert len(broker.sent) == 1
    with sqlite3.connect(store.path) as connection:
        assert sorted(
            row[0]
            for row in connection.execute("SELECT state FROM monitoring_delivery")
        ) == ["awaiting_ack", "pending"]


def test_polling_recovers_from_store_error_and_reports_failed_cycle(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    worker.clock, worker.ack_wait = time.time, 0.001
    original = store.enqueue_monitoring
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database busy")
        return original(*args, **kwargs)

    store.enqueue_monitoring = fail_once
    service = MonitoringService([worker], interval=0.01)
    service.start()
    try:
        wait_until(lambda: service.status()["completed"] == 1)
        assert service.status()["failed_cycles"] == 1
    finally:
        service.close()


def test_ack_observer_failure_cannot_undo_completed_delivery(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("observer failed")

    worker.commit_observer = broken
    worker.drain_once()
    clock[0] += 10
    assert worker.drain_once() == 1
    assert worker.commit_observer_errors == 1
    assert store.get_monitoring(broker.sent[0][0])["state"] == "stored"


def test_failed_receiver_start_does_not_leave_worker_running(tmp_path):
    *_, worker = worker_setup(tmp_path)

    class Receiver:
        closed = False

        def start(self):
            raise TimeoutError()

        def close(self):
            self.closed = True

    receiver = Receiver()
    service = MonitoringService([worker], receiver=receiver)
    with pytest.raises(TimeoutError):
        service.start()
    assert receiver.closed
    assert service.status()["state"] == "stopped"
    assert not service.status()["running"]


def test_invalid_runtime_timeout_does_not_create_databases(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "receiver.db"
    with pytest.raises(ValueError, match="timeout"):
        build_local_service(
            source,
            target,
            "consumer-test",
            "record-mqtt:cctv/records/v1/test",
            "/tmp/test.sock",
            namespaces=["video_detection"],
            timeout=0.1,
        )
    assert not source.exists() and not target.exists()


def test_cancelled_worker_does_not_prepare_or_claim(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(
        tmp_path, enqueue=False
    )
    stopped = threading.Event()
    stopped.set()
    assert worker.drain_once(stop_event=stopped) == 0
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_delivery").fetchone()[0]
            == 0
        )
