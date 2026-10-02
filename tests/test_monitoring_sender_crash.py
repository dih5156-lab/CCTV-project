"""Actual process exit with real source/consumer DBs and an in-process transport."""

import multiprocessing
import os
import sqlite3

import pytest

from src.monitoring_delivery import MonitoringDeliveryStore
from tests.test_monitoring_worker import worker_setup


@pytest.mark.parametrize(
    "point", ["enqueue", "claim", "consumer_commit", "puback_save", "ack_save"]
)
def test_restart_at_sender_boundaries_preserves_one_record(tmp_path, point):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(
        tmp_path, enqueue=point != "enqueue"
    )
    with sqlite3.connect(store.path) as connection:
        original = connection.execute(
            "SELECT job_id,wire_payload,payload_sha256 FROM monitoring_delivery"
        ).fetchone()
    if point == "ack_save":
        worker.drain_once()
        clock[0] = 110
    owner, method = {
        "enqueue": (store, "enqueue_monitoring"),
        "claim": (store, "claim_monitoring"),
        "consumer_commit": (broker, "send"),
        "puback_save": (store, "finish_publish"),
        "ack_save": (store, "complete_monitoring"),
    }[point]

    def child():
        original_method = getattr(owner, method)

        def exit_after(*args, **kwargs):
            original_method(*args, **kwargs)
            os._exit(42)

        setattr(owner, method, exit_after)
        worker.drain_once()

    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("crash child timed out")
    assert process.exitcode == 42
    if point == "enqueue":
        with sqlite3.connect(store.path) as connection:
            original = connection.execute(
                "SELECT job_id,wire_payload,payload_sha256 FROM monitoring_delivery"
            ).fetchone()
        assert original is not None
    worker.store = MonitoringDeliveryStore(store.path)
    clock[0] = 200
    worker.drain_once()
    clock[0] = 210
    worker.drain_once()
    assert worker.store.get_monitoring(original[0])["state"] == "stored"
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                "SELECT job_id,wire_payload,payload_sha256 FROM monitoring_delivery"
            ).fetchone()
            == original
        )
        assert (
            connection.execute(
                "SELECT state FROM delivery_jobs WHERE destination='mqtt:cctv/ai/events'"
            ).fetchone()[0]
            == "held"
        )
    with sqlite3.connect(receiver.path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_records").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT count(*) FROM monitoring_receipts").fetchone()[0]
            == 1
        )
