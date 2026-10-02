"""Idle monitoring polls must leave SQLite's writer available for intake."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from queue import Queue
from unittest.mock import Mock

import pytest

from src.ingest_store import IngestStore
from src.monitoring_contract import MonitoringBinding, MonitoringContractError
from tests.test_monitoring_delivery import BROKER, READER, prepared, seed
from tests.test_mqtt_durable_subscriber import message, subscriber
from tests.test_video_event_ingest import detection, enqueue, ingestor


@pytest.mark.parametrize("operation", ["enqueue", "claim"])
def test_idle_poll_can_read_while_input_writer_holds_database(tmp_path, operation):
    store, binding = prepared(tmp_path / "source.db")
    job = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
    assert job is not None
    with closing(sqlite3.connect(store.path)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        if operation == "enqueue":
            assert (
                store.enqueue_monitoring("video_detection", binding, BROKER, READER)
                == 0
            )
        else:
            assert (
                store.claim_monitoring(
                    binding, BROKER, READER, lease_seconds=30, now=101
                )
                is None
            )
        writer.rollback()
    assert store.get_monitoring(job["job_id"])["attempts"] == 1


def test_concurrent_enqueue_rechecks_records_before_inserting(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    seed(store.path, "new-event")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: store.enqueue_monitoring(
                    "video_detection", binding, BROKER, READER
                ),
                range(2),
            )
        )
    assert sorted(results) == [0, 1]
    with closing(sqlite3.connect(store.path)) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_delivery").fetchone()[0]
            == 2
        )


def test_idle_poll_still_rejects_changed_binding(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    changed = MonitoringBinding(
        binding.origin_id, "other-consumer", binding.destination
    )
    with pytest.raises(MonitoringContractError, match="binding_mismatch"):
        store.enqueue_monitoring("video_detection", changed, BROKER, READER)


def test_idle_poll_does_not_forget_new_records_or_expired_leases(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    first = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
    assert store.enqueue_monitoring("video_detection", binding, BROKER, READER) == 0
    assert (
        store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=101)
        is None
    )
    seed(store.path, "new-event")
    assert store.enqueue_monitoring("video_detection", binding, BROKER, READER) == 1
    second = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=101)
    assert second["job_id"] != first["job_id"]
    expired = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=130)
    assert expired["job_id"] == first["job_id"]
    assert expired["wire_payload"] == first["wire_payload"]


def test_mqtt_source_lock_leaves_message_unacked_until_resubmission(tmp_path):
    store = IngestStore(tmp_path / "source.db")
    instance = subscriber(
        lambda topic, payload: store.accept_mqtt("tlv_mqtt", topic, payload)
    )
    client = Mock()
    client.ack.return_value = 0
    with closing(sqlite3.connect(store.path)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        instance._on_message(client, None, message())
        client.ack.assert_not_called()
        client.disconnect.assert_called_once()
        assert instance._restart_requested
        assert writer.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0] == 0
        writer.rollback()
    # Model broker redelivery to the same durable handler after the lock clears.
    instance._on_message(client, None, message())
    client.ack.assert_called_once_with(17, 1)
    with closing(sqlite3.connect(store.path)) as connection:
        assert (
            connection.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0]
            == 1
        )


def test_video_source_lock_is_reported_before_queueing_and_needs_resubmission(tmp_path):
    video = ingestor(tmp_path)
    queue = Queue()
    event = detection()
    with closing(sqlite3.connect(video.store.path)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            enqueue(video, queue, event)
        assert queue.empty()
        assert writer.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0] == 0
        writer.rollback()
    video.recover_pending()
    with closing(sqlite3.connect(video.store.path)) as connection:
        assert (
            connection.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0]
            == 0
        )
    assert enqueue(video, queue, event)
    assert queue.qsize() == 1
