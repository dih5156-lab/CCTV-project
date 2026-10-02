"""Consumer storage confirmation continues after broker PUBACK and restart."""

import importlib
import json
import sqlite3

import pytest

from src.monitoring_receiver import MonitoringReceiverStore
from src.record_delivery import BrokerAck
from tests.test_monitoring_delivery import BROKER, READER, prepared, seed


class Reader:
    endpoint = READER

    def __init__(self, receiver, clock):
        self.receiver, self.clock = receiver, clock
        self.unavailable = False
        self.wrong = False

    def get_delivery_status(self, binding, delivery_id, *, timeout):
        if self.unavailable:
            raise TimeoutError("status unavailable")
        ack = self.receiver.get_delivery_status(binding, delivery_id)
        if self.wrong and ack:
            ack["consumer_id"] = "wrong-consumer"
        return ack


class Broker:
    endpoint = BROKER

    def __init__(self, receiver, binding):
        self.receiver, self.binding = receiver, binding
        self.deliver = True
        self.lose_puback = False
        self.sent = []

    def send(self, job, *, timeout):
        self.sent.append((job["job_id"], job["wire_payload"]))
        if self.deliver:
            self.receiver.accept(self.binding, job["wire_payload"])
        if self.lose_puback:
            raise TimeoutError("PUBACK lost")
        return BrokerAck(job["job_id"])


def worker_setup(tmp_path, *, enqueue=True):
    store, binding = prepared(tmp_path / "source.db", enqueue=enqueue)
    receiver = MonitoringReceiverStore(
        tmp_path / "consumer.db", consumer_id=binding.consumer_id
    )
    receiver.register_binding(binding)
    clock = [100.0]
    broker, reader = Broker(receiver, binding), Reader(receiver, clock)
    module = importlib.import_module("src.monitoring_worker")
    worker = module.MonitoringDeliveryWorker(
        store,
        broker,
        reader,
        namespace="video_detection",
        binding=binding,
        clock=lambda: clock[0],
    )
    return store, binding, receiver, clock, broker, reader, worker


def test_puback_without_consumer_storage_is_retried_then_ack_persisted(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    broker.deliver = False
    assert worker.drain_once() == 0
    job_id, original = broker.sent[0]
    assert store.get_monitoring(job_id)["state"] == "awaiting_ack"
    clock[0] += 10
    broker.deliver = True
    assert worker.drain_once() == 0
    assert broker.sent == [(job_id, original), (job_id, original)]
    clock[0] += 10
    assert worker.drain_once() == 1
    assert store.get_monitoring(job_id)["state"] == "stored"
    assert worker.drain_once() == 0
    assert len(broker.sent) == 2
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM record_delivery_attempts WHERE outcome='broker_acked'"
            ).fetchone()[0]
            == 2
        )
    with sqlite3.connect(receiver.path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_records").fetchone()[0]
            == 1
        )


def test_consumer_committed_but_puback_lost_recovers_by_query_without_resend(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    broker.lose_puback = True
    assert worker.drain_once() == 0
    job_id = broker.sent[0][0]
    clock[0] += 1
    assert worker.drain_once() == 1
    assert len(broker.sent) == 1
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                "SELECT acked_at FROM record_delivery_state WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            is None
        )
    assert store.get_monitoring(job_id)["acked_at"] == 101


def test_ack_save_failure_reopens_and_retries_same_committed_record(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    worker.drain_once()
    job_id = broker.sent[0][0]
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_ack BEFORE UPDATE OF ack_json ON monitoring_delivery "
            "BEGIN SELECT RAISE(ABORT, 'ACK write failed'); END"
        )
    clock[0] = 110
    with pytest.raises(sqlite3.IntegrityError, match="ACK write failed"):
        worker.drain_once()
    assert store.get_monitoring(job_id)["acked_at"] is None
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER fail_ack")
    clock[0] = 140
    reopened = type(store)(store.path)
    worker.store = reopened
    assert worker.drain_once() == 1
    assert len(broker.sent) == 1
    assert reopened.get_monitoring(job_id)["state"] == "stored"


def test_bad_ack_does_not_succeed_and_does_not_starve_other_records(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    worker.drain_once()
    reader.wrong = True
    seed(store.path, "evt-2")
    clock[0] = 110
    assert worker.drain_once() == 0
    first_id = broker.sent[0][0]
    assert store.get_monitoring(first_id)["state"] == "uncertain"
    assert store.get_monitoring(first_id)["acked_at"] is None
    assert len(broker.sent) == 2
    reader.wrong = False
    clock[0] = 120
    assert worker.drain_once() == 2
    with sqlite3.connect(store.path) as connection:
        responses = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT response_json FROM monitoring_attempts WHERE error='ack_mismatch'"
            )
        ]
    assert responses[0]["consumer_id"] == "wrong-consumer"


def test_status_outage_backs_off_without_false_success(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    reader.unavailable = True
    assert worker.drain_once() == 0
    assert broker.sent == []
    reader.unavailable = False
    assert worker.drain_once() == 0
    assert broker.sent == []
    clock[0] += 1
    worker.drain_once()
    assert len(broker.sent) == 1
    clock[0] += 10
    assert worker.drain_once() == 1


def test_changed_endpoint_is_rejected_before_transmission(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    reader.endpoint = "different-reader"
    with pytest.raises(ValueError):
        worker.drain_once()
    assert broker.sent == []
