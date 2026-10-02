import json
import sqlite3
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.event_ingest_store import EventIngestStore
from src.ingest_store import IngestStore


def rows(store, table):
    with sqlite3.connect(store.path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]


def accept(
    store, payload=b'{"value":0}', namespace="external_mqtt", destination="mqtt:test"
):
    return store.accept_message(
        namespace, "input/test", payload, destination=destination
    )


def record(store, ingest_id, event_id="evt_1", **kwargs):
    return store.record_event(
        ingest_id,
        {"event_id": event_id, "value": 0},
        converter_version="test-v1",
        **kwargs,
    )


def test_v1_upgrade_preserves_sensor_queue_and_frozen_projection(tmp_path):
    path = tmp_path / "common.db"
    old = IngestStore(path)
    sensor = old.accept("tlv_mqtt", "sensor-1", {"value": 0})
    old.prepare(sensor, '{"value":0}')
    store = EventIngestStore(path)
    event = accept(store, b"\xffnot-json")
    assert IngestStore(path).pending() == [sensor]
    assert IngestStore(path).get(sensor)["projection_json"] == '{"value":0}'
    receipt = EventIngestStore(path).get_event_receipt(event)
    assert b64decode(receipt["raw"]["payload_base64"]) == b"\xffnot-json"
    assert store.pending_events("external_mqtt") == [event]
    assert len(rows(store, "sensor_ingest_state")) == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)


def test_failed_upgrade_rolls_back_and_preserves_v1(tmp_path):
    path = tmp_path / "common.db"
    old = IngestStore(path)
    sensor = old.accept("tlv_mqtt", "sensor-1", {"value": 0})
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE delivery_jobs(existing TEXT)")
    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        EventIngestStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='canonical_records'"
            ).fetchone()
            is None
        )
    assert old.pending() == [sensor]


def test_receipt_failure_rolls_back_raw_and_propagates(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_state BEFORE INSERT ON event_ingest_state "
            "BEGIN SELECT RAISE(ABORT, 'disk failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="disk failure"):
        accept(store)
    assert rows(store, "ingest_receipts") == []


def test_t2_failure_preserves_raw_but_rolls_back_event_revision_and_job(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    ingest_id = accept(store)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_revision BEFORE INSERT ON ingest_revisions "
            "BEGIN SELECT RAISE(ABORT, 'commit failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="commit failure"):
        record(store, ingest_id)
    assert len(rows(store, "ingest_receipts")) == 1
    for table in ("canonical_records", "ingest_revisions", "delivery_jobs"):
        assert rows(store, table) == []
    assert store.pending_events("external_mqtt") == [ingest_id]


def test_duplicate_keeps_receipts_and_only_one_frozen_job(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    first, second = accept(store), accept(store)
    assert record(store, first)[0] == "accepted"
    assert record(store, second) == ("duplicate", None)
    assert record(store, first) == ("accepted", None)
    assert len(rows(store, "ingest_receipts")) == 2
    assert len(rows(store, "ingest_revisions")) == 2
    assert len(rows(store, "canonical_records")) == 1
    assert len(rows(store, "delivery_jobs")) == 1
    assert store.pending_events("external_mqtt") == []


def test_same_id_different_body_is_conflict_not_overwrite(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    first, second = accept(store), accept(store, b'{"value":1}')
    record(store, first)
    assert record(store, second) == ("conflict", None)
    saved = rows(store, "canonical_records")[0]
    assert saved["first_ingest_id"] == first
    assert json.loads(saved["payload_json"])["value"] == 0
    assert len(rows(store, "delivery_jobs")) == 1
    assert len(rows(store, "ingest_revisions")) == 2


def test_namespace_separates_same_business_id(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    record(store, accept(store))
    record(store, accept(store, namespace="video_detection"))
    assert len(rows(store, "canonical_records")) == 2


def test_concurrent_duplicate_normalization_creates_one_job(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    receipts = [accept(store), accept(store)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda receipt: record(store, receipt)[0], receipts))
    assert sorted(result) == ["accepted", "duplicate"]
    assert len(rows(store, "delivery_jobs")) == 1


def test_recovery_holds_outputs_and_record_only_has_no_job(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    _, job = record(store, accept(store), recovery=True)
    assert rows(store, "delivery_jobs")[0]["state"] == "held"
    assert store.begin_live_delivery(job) is False
    record(store, accept(store, destination=None), event_id="evt_2")
    assert len(rows(store, "canonical_records")) == 2
    assert len(rows(store, "delivery_jobs")) == 1


def test_quarantine_retains_raw_and_blocks_normalization(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    ingest_id = accept(store, b"invalid")
    store.reject_event(
        ingest_id, [{"code": "invalid_json"}], converter_version="test-v1"
    )
    assert record(store, ingest_id) == ("invalid", None)
    assert rows(store, "canonical_records") == []
    assert store.pending_events("external_mqtt") == []
    assert (
        b64decode(store.get_event_receipt(ingest_id)["raw"]["payload_base64"])
        == b"invalid"
    )


def test_submit_is_not_delivery_ack_and_only_one_sender_claims(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    _, job = record(store, accept(store))
    assert store.begin_live_delivery(job) is True
    assert store.begin_live_delivery(job) is False
    store.finish_live_delivery(job, submitted=True, error=None)
    saved = rows(store, "delivery_jobs")[0]
    assert saved["state"] == "submitted"
    assert saved["attempts"] == 1
