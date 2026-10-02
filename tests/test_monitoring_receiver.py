"""Consumer storage, identity scope and ACK loss must be durable and isolated."""

import importlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.monitoring_contract import (
    MonitoringBinding,
    MonitoringContractError,
    parse_record,
    validate_ack,
)

BINDING = MonitoringBinding(
    "origin-1", "consumer-v1", "record-mqtt:cctv/records/v1/lab"
)


def receiver(path):
    module = importlib.import_module("src.monitoring_receiver")
    store = module.MonitoringReceiverStore(path, consumer_id=BINDING.consumer_id)
    store.register_binding(BINDING)
    return store


def payload(delivery_id="job-1", record_id="evt-1", value=0):
    return json.dumps(
        {
            "schema_version": "1.0",
            "delivery_id": delivery_id,
            "source_namespace": "video_detection",
            "event_id": record_id,
            "record": {"event_id": record_id, "value": value},
        },
        sort_keys=True,
    ).encode()


def counts(path):
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in (
                "monitoring_records",
                "monitoring_receipts",
                "monitoring_rejected",
            )
        )


def test_commit_then_reopen_lookup_and_duplicate(tmp_path):
    store = receiver(tmp_path / "consumer.db")
    wire = payload()
    first = store.accept(BINDING, wire)
    assert first["status"] == "stored"
    assert validate_ack(first, BINDING, parse_record(wire)) == first
    reopened = receiver(store.path)
    recovered = reopened.get_delivery_status(BINDING, "job-1")
    assert recovered["status"] == "duplicate"
    duplicate = reopened.accept(BINDING, wire)
    assert duplicate["receiver_record_id"] == first["receiver_record_id"]
    assert duplicate["stored_at"] == first["stored_at"]
    assert counts(store.path) == (1, 1, 0)
    saved = reopened.get_record(BINDING, first["receiver_record_id"])
    assert saved["record"] == json.loads(wire)["record"]
    assert reopened.get_delivery_status(BINDING, "not-stored") is None


def test_ledger_insert_failure_rolls_back_without_ack(tmp_path):
    store = receiver(tmp_path / "consumer.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON monitoring_receipts "
            "BEGIN SELECT RAISE(ABORT, 'storage failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="storage failure"):
        store.accept(BINDING, payload())
    assert counts(store.path) == (0, 0, 0)
    assert store.get_delivery_status(BINDING, "job-1") is None
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER fail_receipt")
    assert store.accept(BINDING, payload())["status"] == "stored"


@pytest.mark.parametrize(
    "delivery_id,code", [("job-1", "delivery_conflict"), ("job-2", "record_conflict")]
)
def test_conflict_preserves_original_and_quarantines_raw(tmp_path, delivery_id, code):
    store = receiver(tmp_path / "consumer.db")
    ack = store.accept(BINDING, payload())
    changed = payload(delivery_id=delivery_id, value=1)
    with pytest.raises(MonitoringContractError, match=code):
        store.accept(BINDING, changed)
    assert counts(store.path) == (1, 1, 1)
    assert store.get_record(BINDING, ack["receiver_record_id"])["record"]["value"] == 0
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT raw_payload, error_code FROM monitoring_rejected"
        ).fetchone() == (changed, code)


def test_concurrent_retry_creates_one_record_and_receipt(tmp_path):
    store = receiver(tmp_path / "consumer.db")
    with ThreadPoolExecutor(max_workers=2) as pool:
        acks = list(pool.map(lambda _: store.accept(BINDING, payload()), range(2)))
    assert sorted(ack["status"] for ack in acks) == ["duplicate", "stored"]
    assert counts(store.path) == (1, 1, 0)


def test_business_duplicate_new_delivery_keeps_first_record(tmp_path):
    store = receiver(tmp_path / "consumer.db")
    first = store.accept(BINDING, payload())
    second = store.accept(BINDING, payload(delivery_id="job-2"))
    assert second["status"] == "duplicate"
    assert second["receiver_record_id"] == first["receiver_record_id"]
    assert counts(store.path) == (1, 2, 0)


def test_binding_is_persistent_and_does_not_leak_other_origin(tmp_path):
    store = receiver(tmp_path / "consumer.db")
    first = store.accept(BINDING, payload())
    other = MonitoringBinding(
        "origin-2", "consumer-v1", "record-mqtt:cctv/records/v1/other"
    )
    store.register_binding(other)
    assert store.get_delivery_status(other, "job-1") is None
    assert store.get_record(other, first["receiver_record_id"]) is None
    second = store.accept(other, payload())
    assert second["receiver_record_id"] != first["receiver_record_id"]
    changed = MonitoringBinding("origin-2", "consumer-v1", BINDING.destination)
    with pytest.raises(MonitoringContractError, match="binding_mismatch"):
        receiver(store.path).register_binding(changed)
    with pytest.raises(MonitoringContractError, match="binding_mismatch"):
        store.accept(changed, payload())
    with pytest.raises(MonitoringContractError, match="binding_mismatch"):
        store.get_delivery_status(changed, "job-1")
    module = importlib.import_module("src.monitoring_receiver")
    with pytest.raises(MonitoringContractError, match="binding_mismatch"):
        module.MonitoringReceiverStore(store.path, consumer_id="consumer-reset-v2")
    assert counts(store.path) == (2, 2, 1)


def test_malformed_payload_and_quarantine_storage_failure(tmp_path):
    store = receiver(tmp_path / "consumer.db")
    with pytest.raises(MonitoringContractError, match="invalid"):
        store.accept(BINDING, b"\xff")
    assert counts(store.path) == (0, 0, 1)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_rejected BEFORE INSERT ON monitoring_rejected "
            "BEGIN SELECT RAISE(ABORT, 'quarantine failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="quarantine failure"):
        store.accept(BINDING, b"bad json")


def test_foreign_database_is_not_adopted_or_modified(tmp_path):
    path = tmp_path / "other.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE existing (value TEXT)")
        connection.execute("INSERT INTO existing VALUES ('keep')")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="monitoring"):
        receiver(path)
    assert path.read_bytes() == before


def test_read_failure_is_not_reported_as_missing(tmp_path):
    store = receiver(tmp_path / "consumer.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE monitoring_receipts")
    with pytest.raises(sqlite3.OperationalError):
        store.get_delivery_status(BINDING, "job-1")


def test_commit_failure_does_not_return_ack_or_leave_partial_record(tmp_path):
    store = receiver(tmp_path / "consumer.db")
    with sqlite3.connect(store.path) as connection:
        connection.executescript(
            "CREATE TABLE commit_probe (id TEXT REFERENCES monitoring_records(receiver_record_id) "
            "DEFERRABLE INITIALLY DEFERRED);"
            "CREATE TRIGGER fail_at_commit AFTER INSERT ON monitoring_receipts "
            "BEGIN INSERT INTO commit_probe VALUES ('missing-parent'); END;"
        )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store.accept(BINDING, payload())
    assert counts(store.path) == (0, 0, 0)
    assert store.get_delivery_status(BINDING, "job-1") is None
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER fail_at_commit")
    assert store.accept(BINDING, payload())["status"] == "stored"


def test_unregistered_binding_is_not_created_by_inbound_message(tmp_path):
    store = receiver(tmp_path / "consumer.db")
    unknown = MonitoringBinding(
        "unknown", "consumer-v1", "record-mqtt:cctv/records/v1/unknown"
    )
    with pytest.raises(MonitoringContractError, match="binding_mismatch"):
        store.accept(unknown, payload())
    assert counts(store.path) == (0, 0, 1)
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_bindings").fetchone()[0]
            == 1
        )
