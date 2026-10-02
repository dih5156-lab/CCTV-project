"""Commit markers must follow durability and must never invalidate a saved receipt."""

import json
import sqlite3

import pytest

from src.event_ingest_store import EventIngestStore
from src.ingest_store import IngestStore


@pytest.mark.parametrize("kind", ["sensor", "event"])
def test_input_observer_reads_committed_data_from_another_connection(tmp_path, kind):
    path = tmp_path / "source.db"
    seen = []

    def observe(phase, ingest_id, **context):
        with sqlite3.connect(path) as connection:
            assert (
                connection.execute(
                    "SELECT received_at FROM ingest_receipts WHERE ingest_id=?",
                    (ingest_id,),
                ).fetchone()[0]
                == context["received_at"]
            )
            if phase == "normalize_completed_at":
                assert (
                    connection.execute(
                        "SELECT count(*) FROM canonical_records"
                    ).fetchone()[0]
                    == 1
                )
        seen.append(phase)

    store = EventIngestStore(path, commit_observer=observe)
    if kind == "sensor":
        store.accept("tlv_mqtt", "sensor-1", {"temperature": 0})
    else:
        event = {"event_id": "evt-1"}
        receipt = store.accept_event_json("video_detection", event, destination=None)
        store.record_event(receipt, event, converter_version="test")
    assert seen == (
        ["ingest_commit_completed_at"]
        if kind == "sensor"
        else ["ingest_commit_completed_at", "normalize_completed_at"]
    )


def test_observer_failure_does_not_repeat_successful_input(tmp_path):
    def broken(*args, **kwargs):
        raise RuntimeError("instrumentation error")

    store = IngestStore(tmp_path / "source.db", commit_observer=broken)
    receipt = store.accept("tlv_mqtt", "sensor-1", {"value": 0})
    assert store.get(receipt)["raw"]["value"] == 0
    assert store.commit_observer_errors == 1


def test_failed_commit_does_not_emit_marker(tmp_path):
    seen = []
    store = IngestStore(
        tmp_path / "source.db",
        commit_observer=lambda *args, **kwargs: seen.append(args),
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_input BEFORE INSERT ON sensor_ingest_state BEGIN SELECT RAISE(ABORT,'failed'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.accept("tlv_mqtt", "sensor-1", {"value": 0})
    assert seen == []


def test_commit_latency_has_explicit_quality_and_bounded_window():
    from src.monitoring_metrics import CommitObservations

    clock = [100.0]
    observed = CommitObservations(max_samples=2, wall_clock=lambda: clock[0])
    observed.committed(
        "consumer_ack_persisted_at",
        "ing-1",
        received_at="1970-01-01T00:01:30+00:00",
        delivery_id="job-1",
    )
    assert observed.report()["latency"]["unknown"] == 1
    known = CommitObservations(
        max_samples=2, wall_clock=lambda: clock[0], clock_sync="confirmed"
    )
    for receipt, timestamp in [("old", 80), ("good", 90), ("future", 110)]:
        known.committed(
            "consumer_ack_persisted_at",
            receipt,
            received_at=f"1970-01-01T00:01:{timestamp - 60:02d}+00:00",
            delivery_id=receipt,
        )
    result = known.report()
    assert result["latency"]["valid"] == result["latency"]["invalid"] == 1
    assert result["latency"]["p95_ms"] == 10000
    assert result["evicted"] == 1
    json.dumps(result, allow_nan=False)
