"""Frozen manifests retain scope and do not hide late normalization."""

import importlib
import multiprocessing
import os
import sqlite3

import pytest

from src.event_ingest_store import EventIngestStore
from src.monitoring_contract import MonitoringBinding, MonitoringContractError
from tests.test_monitoring_delivery import BROKER, READER, prepared, seed


def snapshots(tmp_path, clock=None):
    cls = importlib.import_module("src.monitoring_snapshot").MonitoringSnapshots
    return cls(
        tmp_path / "snapshots.db", clock=(lambda: clock[0]) if clock else lambda: 100
    )


def test_pages_frozen_across_new_records_and_reopen(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    seed(store.path, "evt-2")
    store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    audit = snapshots(tmp_path)
    snapshot = audit.create_source(store, binding, namespace="video_detection")
    first = audit.list_snapshot(binding, snapshot["snapshot_id"], limit=1)
    seed(store.path, "evt-3")
    store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    second = snapshots(tmp_path).list_snapshot(
        binding, snapshot["snapshot_id"], cursor=first["next_cursor"], limit=1
    )
    assert snapshot["count"] == 2
    assert first["items"][0]["record_id"] == "evt-1"
    assert second["items"][0]["record_id"] == "evt-2"
    assert second["next_cursor"] is None
    assert audit.create_source(store, binding)["count"] == 3


def test_unfinished_input_is_counted_and_late_old_timestamp_is_in_next_snapshot(
    tmp_path,
):
    store, binding = prepared(tmp_path / "source.db")
    ingest = EventIngestStore(store.path)
    event = {"event_id": "late"}
    receipt = ingest.accept_event_json("video_detection", event, destination=None)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE ingest_receipts SET received_at='2000-01-01T00:00:00+00:00' WHERE ingest_id=?",
            (receipt,),
        )
    audit = snapshots(tmp_path)
    first = audit.create_source(store, binding)
    assert first["pending_normalization"] == 1
    ingest.record_event(receipt, event, converter_version="test")
    second = audit.create_source(store, binding)
    assert second["pending_delivery_creation"] == 1
    store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    third = audit.create_source(store, binding)
    assert third["count"] == 2
    assert third["pending_normalization"] == third["pending_delivery_creation"] == 0
    assert audit.list_snapshot(binding, first["snapshot_id"])["count"] == 1


def test_scope_cursor_ttl_and_limit_are_enforced(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    clock = [100]
    audit = snapshots(tmp_path, clock)
    first = audit.create_source(store, binding, ttl=10)
    wrong = MonitoringBinding(binding.origin_id, "wrong", binding.destination)
    with pytest.raises(MonitoringContractError, match="binding_mismatch"):
        audit.list_snapshot(wrong, first["snapshot_id"])
    with pytest.raises(MonitoringContractError, match="cursor_invalid"):
        audit.list_snapshot(binding, first["snapshot_id"], cursor="invented")
    with pytest.raises(ValueError):
        audit.list_snapshot(binding, first["snapshot_id"], limit=True)
    clock[0] = 110
    with pytest.raises(MonitoringContractError, match="snapshot_expired"):
        audit.list_snapshot(binding, first["snapshot_id"])


def test_capacity_and_copy_failure_leave_no_partial_snapshot(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    seed(store.path, "evt-2")
    store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    audit = snapshots(tmp_path)
    with pytest.raises(MonitoringContractError, match="snapshot_capacity"):
        audit.create_source(store, binding, max_items=1)
    with audit._connect() as connection:
        assert connection.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 0
        assert (
            connection.execute("SELECT count(*) FROM snapshot_items").fetchone()[0] == 0
        )
        connection.execute(
            "CREATE TRIGGER fail_copy BEFORE INSERT ON snapshot_items BEGIN SELECT RAISE(ABORT,'copy failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        audit.create_source(store, binding)
    with audit._connect() as connection:
        assert connection.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 0


def test_cursor_cannot_be_reused_for_another_snapshot(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    seed(store.path, "evt-2")
    store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    audit = snapshots(tmp_path)
    first, second = audit.create_source(store, binding), audit.create_source(
        store, binding
    )
    cursor = audit.list_snapshot(binding, first["snapshot_id"], limit=1)["next_cursor"]
    with pytest.raises(MonitoringContractError, match="cursor_invalid"):
        audit.list_snapshot(binding, second["snapshot_id"], cursor=cursor)


def test_process_exit_before_manifest_commit_rolls_back(tmp_path):
    from tests.test_monitoring_reconciliation import setup

    store, binding, receiver, wire, audit = setup(tmp_path)
    receiver.accept(binding, wire)

    def child():
        original = audit._consumer_items

        def stop_after_first(*args):
            for item in original(*args):
                yield item
                os._exit(42)

        audit._consumer_items = stop_after_first
        audit.create_consumer(receiver, binding)

    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("snapshot child timed out")
    assert process.exitcode == 42
    reopened = snapshots(tmp_path)
    with reopened._connect() as connection:
        assert connection.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 0
        assert (
            connection.execute("SELECT count(*) FROM snapshot_items").fetchone()[0] == 0
        )
    assert reopened.create_consumer(receiver, binding)["count"] == 1


def test_snapshot_creation_does_not_modify_source_or_consumer_files(tmp_path):
    from tests.test_monitoring_reconciliation import setup

    store, binding, receiver, wire, audit = setup(tmp_path)
    receiver.accept(binding, wire)
    source_before, consumer_before = store.path.read_bytes(), receiver.path.read_bytes()
    audit.create_source(store, binding)
    audit.create_consumer(receiver, binding)
    assert store.path.read_bytes() == source_before
    assert receiver.path.read_bytes() == consumer_before


def test_existing_source_database_is_not_adopted_as_snapshot_database(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    before = store.path.read_bytes()
    module = importlib.import_module("src.monitoring_snapshot")
    with pytest.raises(ValueError, match="Unsupported"):
        module.MonitoringSnapshots(store.path)
    assert store.path.read_bytes() == before
