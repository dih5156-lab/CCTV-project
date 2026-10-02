import sqlite3

import pytest

from src.ingest_store import IngestStore


def test_receipt_and_raw_payload_survive_reopening(tmp_path):
    path = tmp_path / "common.db"
    store = IngestStore(path)
    raw = {"payload": "invalid-base64!", "received_at": 0}
    ingest_id = store.accept("tlv_mqtt", "sensor-1", raw)
    recovered = IngestStore(path).get(ingest_id)
    assert recovered["raw"] == raw
    assert recovered["stage"] == "pending"
    assert recovered["ingest_id"] == ingest_id
    assert recovered["received_at"].endswith("+00:00")


def test_equal_measurements_remain_separate_receipts(tmp_path):
    store = IngestStore(tmp_path / "common.db")
    first = store.accept("tlv_mqtt", "sensor-1", {"temperature": 0})
    second = store.accept("tlv_mqtt", "sensor-1", {"temperature": 0})
    assert first != second
    assert len(store.pending()) == 2


def test_sensor_worker_does_not_consume_another_source_namespace(tmp_path):
    store = IngestStore(tmp_path / "common.db")
    sensor = store.accept("tlv_mqtt", "sensor-1", {"temperature": 0})
    external = store.accept("external_mqtt", "sensor-1", {"temperature": 0})
    assert store.pending() == [sensor]
    assert store.pending(namespace="external_mqtt") == [external]


def test_invalid_numeric_raw_is_preserved_but_not_retried(tmp_path):
    store = IngestStore(tmp_path / "common.db")
    ingest_id = store.accept("tlv_mqtt", "sensor-1", {"value": float("nan")})
    receipt = store.get(ingest_id)
    assert receipt["stage"] == "quarantined"
    assert receipt["raw_format"] == "error_repr"
    assert "NaN" in receipt["raw_payload"]
    assert store.pending() == []


def test_write_failure_does_not_return_success(tmp_path):
    path = tmp_path / "common.db"
    store = IngestStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_state BEFORE INSERT ON sensor_ingest_state "
            "BEGIN SELECT RAISE(ABORT, 'injected storage failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected storage failure"):
        store.accept("tlv_mqtt", "sensor-1", {"value": 0})
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM ingest_receipts"
        ).fetchone() == (0,)


def test_unknown_schema_is_rejected_without_modification(tmp_path):
    path = tmp_path / "common.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="schema"):
        IngestStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (99,)


def test_prepared_projection_is_frozen_and_recovered(tmp_path):
    path = tmp_path / "common.db"
    store = IngestStore(path)
    ingest_id = store.accept("tlv_mqtt", "sensor-1", {"value": 0})
    store.prepare(ingest_id, '{"temperature":0}')
    store.prepare(ingest_id, '{"temperature":99}')
    recovered = IngestStore(path)
    assert recovered.get(ingest_id)["projection_json"] == '{"temperature":0}'
    assert recovered.pending() == [ingest_id]
    recovered.complete(ingest_id)
    assert recovered.pending() == []
    assert recovered.get(ingest_id)["raw"] == {"value": 0}


def test_failed_receipt_does_not_starve_later_inputs(tmp_path, monkeypatch):
    store = IngestStore(tmp_path / "common.db")
    first = store.accept("tlv_mqtt", "sensor-1", {"value": 1})
    second = store.accept("tlv_mqtt", "sensor-1", {"value": 2})
    monkeypatch.setattr("src.ingest_store.time.time", lambda: 100)
    store.retry(first, "temporary failure")
    assert store.pending() == [second]
    monkeypatch.setattr("src.ingest_store.time.time", lambda: 200)
    assert store.pending() == [first, second]
    assert store.get(first)["attempts"] == 1


def test_quarantine_preserves_original_and_cannot_be_overwritten(tmp_path):
    store = IngestStore(tmp_path / "common.db")
    ingest_id = store.accept("tlv_mqtt", "sensor-1", {"payload": "broken"})
    store.quarantine(ingest_id, "invalid_tlv")
    store.prepare(ingest_id, "{}")
    store.complete(ingest_id)
    assert store.pending() == []
    assert store.get(ingest_id)["stage"] == "quarantined"
    assert store.get(ingest_id)["raw"] == {"payload": "broken"}
