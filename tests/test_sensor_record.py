"""TLV observations reuse the receipt and preserve measurement quality."""

import importlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from src.ingest_store import IngestStore
from src.record_delivery import RecordDeliveryStore


def module():
    return importlib.import_module("src.sensor_record")


def rows(path, table):
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]


def prepared(path, *, table="t34954", data=None, stage="ready"):
    store = IngestStore(path)
    receipt = store.accept("tlv_mqtt", "sensor-1", {"packet": "original"})
    parsed = {"temperature_c": 0, "created_at": 0} if data is None else data
    projection = {
        table: [{}],
        "sensor_data": [
            {"device_id": "sensor-1", "object_id": table[1:], "payload_tlv": parsed}
        ],
    }
    store.prepare(receipt, json.dumps(projection))
    if stage == "complete":
        store.complete(receipt)
    return receipt


@pytest.mark.parametrize("stage", ["ready", "complete"])
def test_ready_and_completed_projection_becomes_observation_once(tmp_path, stage):
    path = tmp_path / "common.db"
    receipt = prepared(path, stage=stage)
    store = module().SensorRecordStore(path)
    worker = module().SensorRecordWorker(store)
    worker.drain_once()
    worker.drain_once()
    saved = rows(path, "canonical_records")
    assert len(saved) == 1
    observation = json.loads(saved[0]["payload_json"])
    assert observation["message_type"] == "sensor_observation"
    assert observation["source_namespace"] == "tlv_mqtt"
    assert observation["source_record_id"] == receipt
    assert observation["observation_id"] == saved[0]["source_record_id"]
    assert "event_id" not in observation
    assert observation["received_at"] == store.get(receipt)["received_at"]
    assert observation["observed_at"] == "1970-01-01T09:00:00+09:00"
    assert observation["measurements"]["temperature"]["value"] == 0
    assert observation["measurements"]["humidity"]["value"] is None
    assert observation["measurements"]["humidity"]["quality"] == "missing"
    assert store.get(receipt)["stage"] == stage
    assert len(rows(path, "ingest_receipts")) == 1
    assert len(rows(path, "ingest_revisions")) == 1
    assert rows(path, "delivery_jobs") == []


def test_record_delivery_uses_observation_id_not_alarm_event_id(tmp_path):
    path = tmp_path / "common.db"
    prepared(path, table="t34955", data={"angle_x_deg": 0, "angle_y_deg": "bad"})
    module().SensorRecordWorker(module().SensorRecordStore(path)).drain_once()
    delivery = RecordDeliveryStore(path)
    destination = "record-mqtt:cctv/records/v1/sensors"
    assert delivery.enqueue_pending("tlv_mqtt", destination, "test-endpoint") == 1
    job = delivery.claim(destination, "test-endpoint", lease_seconds=10)
    payload = json.loads(job["wire_payload"])
    assert "event_id" not in payload
    assert payload["observation_id"] == payload["record"]["observation_id"]
    assert payload["record"]["measurements"]["angle_y"]["quality"] == "invalid"
    assert payload["record"]["measurements"]["angle_x"]["unit"] == "deg"


def test_failed_canonical_commit_leaves_receipt_retryable(tmp_path):
    path = tmp_path / "common.db"
    receipt = prepared(path)
    store = module().SensorRecordStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_revision BEFORE INSERT ON ingest_revisions "
            "BEGIN SELECT RAISE(ABORT, 'disk failure'); END"
        )
    worker = module().SensorRecordWorker(store)
    worker.drain_once()
    assert rows(path, "canonical_records") == []
    assert store.get(receipt)["stage"] == "ready"
    assert store.get_event_receipt(receipt)["stage"] == "pending"
    assert store.get_event_receipt(receipt)["attempts"] == 1
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER fail_revision")
        connection.execute("UPDATE event_ingest_state SET next_attempt_at=0")
    worker.drain_once()
    assert len(rows(path, "canonical_records")) == 1


def test_unsupported_profile_retained_without_blocking_next_sensor(tmp_path):
    path = tmp_path / "common.db"
    unsupported = prepared(path, table="t3", data={"manufacturer": "test"})
    supported = prepared(path)
    store = module().SensorRecordStore(path)
    module().SensorRecordWorker(store).drain_once()
    assert store.get_event_receipt(unsupported)["stage"] == "invalid"
    errors = json.loads(rows(path, "ingest_revisions")[0]["errors_json"])
    assert errors == [{"field": "profile", "code": "unsupported_profile"}]
    assert store.get(unsupported)["stage"] == "ready"
    assert store.get_event_receipt(supported)["stage"] == "accepted"
    assert len(rows(path, "ingest_receipts")) == 2


def test_separate_receipts_remain_separate_observations(tmp_path):
    path = tmp_path / "common.db"
    first, second = prepared(path), prepared(path)
    module().SensorRecordWorker(module().SensorRecordStore(path)).drain_once()
    records = rows(path, "canonical_records")
    assert len({r["source_record_id"] for r in records}) == 2
    assert {json.loads(r["payload_json"])["source_record_id"] for r in records} == {
        first,
        second,
    }


def test_concurrent_projection_commits_one_observation(tmp_path):
    path = tmp_path / "common.db"
    receipt = prepared(path)
    first = module().SensorRecordStore(path)
    second = module().SensorRecordStore(path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [
            executor.submit(store.project_sensor, receipt) for store in (first, second)
        ]
        for job in jobs:
            job.result()
    assert len(rows(path, "canonical_records")) == 1
    assert len(rows(path, "ingest_revisions")) == 1


@pytest.mark.parametrize("table,stage", [("t34954", "accepted"), ("t99999", "invalid")])
def test_finished_projection_does_not_need_another_write_lock(tmp_path, table, stage):
    path = tmp_path / "common.db"
    receipt = prepared(path, table=table)
    store = module().SensorRecordStore(path)
    store.project_sensor(receipt)
    assert store.get_event_receipt(receipt)["stage"] == stage
    tables = (
        "ingest_receipts",
        "event_ingest_state",
        "canonical_records",
        "ingest_revisions",
    )
    before = {name: rows(path, name) for name in tables}

    # 다른 입력의 쓰기 예약 중에도 끝난 접수의 중복 호출은 읽기만으로 종료해야 한다.
    with closing(sqlite3.connect(path)) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        store.project_sensor(receipt)
        blocker.rollback()

    assert {name: rows(path, name) for name in tables} == before


@pytest.mark.parametrize("phase", ["state", "canonical"])
@pytest.mark.parametrize("lock", ["write", "commit"])
def test_locked_projection_preserves_receipt_and_worker_recovers_once(
    tmp_path, monkeypatch, phase, lock
):
    path = tmp_path / "common.db"
    receipt = prepared(path)
    store = module().SensorRecordStore(path)
    original_receipt = store.get(receipt)

    with closing(sqlite3.connect(path)) as blocker:

        def hold_lock():
            blocker.execute("BEGIN IMMEDIATE" if lock == "write" else "BEGIN")
            blocker.execute("SELECT * FROM ingest_receipts").fetchall()

        # 상태 준비와 정규화 각각의 쓰기/커밋을 실제 SQLite 잠금으로 실패시킨다.
        if phase == "state":
            hold_lock()
        else:
            record_observation = store.record_observation

            def blocked_record(*args, **kwargs):
                hold_lock()
                return record_observation(*args, **kwargs)

            monkeypatch.setattr(store, "record_observation", blocked_record)
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            store.project_sensor(receipt)
        blocker.rollback()

    assert store.get(receipt) == original_receipt
    assert rows(path, "canonical_records") == []
    assert rows(path, "ingest_revisions") == []
    # 새 저장소로 재개해 인메모리 상태 없이 원본 접수에서 복구되는지 확인한다.
    worker = module().SensorRecordWorker(module().SensorRecordStore(path))
    worker.drain_once()
    worker.drain_once()
    assert store.get(receipt) == original_receipt
    assert store.get_event_receipt(receipt)["stage"] == "accepted"
    assert len(rows(path, "ingest_receipts")) == 1
    assert len(rows(path, "canonical_records")) == 1
    assert len(rows(path, "ingest_revisions")) == 1
    assert rows(path, "delivery_jobs") == []


@pytest.mark.parametrize(
    "sensor_rows", [[], [1], [{"object_id": "34954", "payload_tlv": []}]]
)
def test_malformed_projection_preserves_raw_and_pg_work(tmp_path, sensor_rows):
    path = tmp_path / "common.db"
    raw = IngestStore(path)
    receipt = raw.accept("tlv_mqtt", "sensor-1", {"packet": "original"})
    raw.prepare(receipt, json.dumps({"sensor_data": sensor_rows}))
    store = module().SensorRecordStore(path)
    module().SensorRecordWorker(store).drain_once()
    assert store.get(receipt)["stage"] == "ready"
    assert store.get_event_receipt(receipt)["stage"] == "invalid"
    assert rows(path, "canonical_records") == []
    assert len(rows(path, "ingest_receipts")) == 1


def test_combined_profile_preserves_invalid_time_and_values(tmp_path):
    path = tmp_path / "common.db"
    prepared(
        path,
        table="t34957",
        data={
            "temperature_c": None,
            "angle_x_deg": -0.5,
            "angle_y_deg": False,
            "created_at": "bad",
        },
    )
    module().SensorRecordWorker(module().SensorRecordStore(path)).drain_once()
    observation = json.loads(rows(path, "canonical_records")[0]["payload_json"])
    assert observation["observed_at"] is None
    assert observation["measurements"]["temperature"]["quality"] == "missing"
    assert observation["measurements"]["angle_x"]["value"] == -0.5
    assert observation["measurements"]["angle_y"]["quality"] == "invalid"
