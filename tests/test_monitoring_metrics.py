"""Timing quality is explicit; backlog includes quarantined/blocked records."""

import json
import sqlite3

import pytest

from src.monitoring_metrics import MonitoringTimings, source_metrics
from tests.test_monitoring_delivery import BROKER, READER, prepared, seed
from tests.test_monitoring_worker import worker_setup


def test_monotonic_samples_are_bounded_and_keep_invalid_and_errors():
    ticks = iter([1, 1.1, 2, 1, 3, 3.2, 4, 4.4])
    timing = MonitoringTimings(max_samples=3, monotonic=lambda: next(ticks))
    assert timing.measure("mqtt_publish", lambda: 7) == 7
    timing.measure("mqtt_publish", lambda: None)
    with pytest.raises(TimeoutError):
        timing.measure("mqtt_publish", lambda: (_ for _ in ()).throw(TimeoutError()))
    timing.measure("mqtt_publish", lambda: None)
    report = timing.report()["mqtt_publish"]
    assert report["valid"] == 2 and report["invalid"] == 1
    assert report["raised"] == report["evicted"] == 1
    assert report["p50_ms"] == pytest.approx(200)
    assert report["p95_ms"] == pytest.approx(400)
    assert timing.report()["consumer_lookup"]["p95_ms"] is None
    json.dumps(timing.report(), allow_nan=False)


def test_worker_measures_lookup_publish_and_successful_ack_commit(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    timing = MonitoringTimings()
    worker.timings = timing
    worker.drain_once()
    clock[0] += 10
    assert worker.drain_once() == 1
    report = timing.report()
    assert report["consumer_lookup"]["valid"] == 2
    assert report["mqtt_publish"]["valid"] == 1
    assert report["consumer_ack_commit"]["valid"] == 1
    assert report["consumer_ack_commit"]["raised"] == 0


def test_backlog_quality_rates_and_namespace_scope(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    seed(store.path, "evt-2")
    seed(store.path, "evt-3")
    store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE ingest_receipts SET received_at='1970-01-01T09:01:00+09:00'"
        )
        connection.execute(
            "UPDATE monitoring_delivery SET state='blocked' WHERE rowid=1"
        )
        connection.execute(
            "UPDATE monitoring_delivery SET state='quarantined' WHERE rowid=2"
        )
        connection.execute(
            "UPDATE ingest_receipts SET received_at='1970-01-01T00:03:00+00:00' WHERE rowid=3"
        )
    unknown = source_metrics(store, binding, now=100)
    assert unknown["unconfirmed_age"] == {
        "valid": 0,
        "invalid": 1,
        "unknown": 2,
        "oldest_seconds": None,
    }
    assert unknown["ingest_per_hour"] is None
    known = source_metrics(
        store, binding, now=100, clock_sync="confirmed", window_seconds=60
    )
    assert known["states"]["blocked"] == known["states"]["quarantined"] == 1
    assert known["unconfirmed_age"]["oldest_seconds"] == 40
    assert known["ingest_count_in_window"] == 2
    assert known["ingest_per_hour"] == 120
    assert known["database_bytes"] > 0
    assert "received_to_ack_commit" in known["not_measured"]
    empty = source_metrics(store, binding, namespace="other", now=100)
    assert sum(empty["states"].values()) == 0


def test_ack_failure_is_timed_and_still_propagates(tmp_path):
    store, binding, receiver, clock, broker, reader, worker = worker_setup(tmp_path)
    worker.timings = MonitoringTimings()
    worker.drain_once()
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER bad_ack BEFORE UPDATE OF ack_json ON monitoring_delivery BEGIN SELECT RAISE(ABORT,'disk full'); END"
        )
    clock[0] += 10
    with pytest.raises(sqlite3.IntegrityError):
        worker.drain_once()
    assert worker.timings.report()["consumer_ack_commit"]["raised"] == 1


def test_metrics_do_not_convert_database_errors_to_zero(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE monitoring_attempts")
    with pytest.raises(sqlite3.OperationalError):
        source_metrics(store, binding)
