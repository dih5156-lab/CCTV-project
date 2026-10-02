"""Source consumer-ACK state must survive failure independently of broker ACK."""

import importlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.event_ingest_store import EventIngestStore
from src.ingest_store import IngestStore
from src.monitoring_contract import MonitoringBinding
from src.monitoring_receiver import MonitoringReceiverStore
from src.record_delivery import BrokerAck, RecordDeliveryStore

DESTINATION = "record-mqtt:cctv/records/v1/monitor"
BROKER = "test-broker"
READER = "test-consumer-status"


def seed(path, record_id="evt-1"):
    store = EventIngestStore(path)
    event = {"event_id": record_id, "value": 0}
    receipt = store.accept_event_json(
        "video_detection", event, destination="mqtt:cctv/ai/events"
    )
    _, alarm = store.record_event(
        receipt, event, converter_version="test", recovery=True
    )
    return alarm


def prepared(path, *, enqueue=True):
    seed(path)
    module = importlib.import_module("src.monitoring_delivery")
    store = module.MonitoringDeliveryStore(path)
    binding = MonitoringBinding(store.origin_id, "consumer-1", DESTINATION)
    if enqueue:
        store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    return store, binding


def test_v4_preserves_old_writers_and_reserves_monitoring_destination(tmp_path):
    path = tmp_path / "common.db"
    sensor_id = IngestStore(path).accept("tlv_mqtt", "sensor-1", {"value": 0})
    store, binding = prepared(path)
    assert IngestStore(path).get(sensor_id)["raw"]["value"] == 0
    assert (
        RecordDeliveryStore(path).claim(DESTINATION, BROKER, lease_seconds=30) is None
    )
    job = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
    assert (
        job["wire_payload"]
        == json.dumps(
            json.loads(job["wire_payload"]), ensure_ascii=False, sort_keys=True
        ).encode()
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert (
            connection.execute(
                "SELECT state FROM delivery_jobs WHERE destination='mqtt:cctv/ai/events'"
            ).fetchone()[0]
            == "held"
        )
    assert (
        importlib.import_module("src.monitoring_delivery")
        .MonitoringDeliveryStore(path)
        .origin_id
        == store.origin_id
    )


def test_frozen_wire_and_id_survive_retry_and_canonical_change(tmp_path):
    store, binding = prepared(tmp_path / "common.db")
    first = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
    assert store.fail_monitoring(first, "timeout", now=101)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE canonical_records SET payload_json=?",
            ('{"event_id":"evt-1","value":99}',),
        )
    assert store.enqueue_monitoring("video_detection", binding, BROKER, READER) == 0
    second = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=102)
    assert second["job_id"] == first["job_id"]
    assert second["wire_payload"] == first["wire_payload"]
    assert second["payload_sha256"] == first["payload_sha256"]


def test_cannot_adopt_previously_attempted_destination_or_rebind_identity(tmp_path):
    path = tmp_path / "common.db"
    seed(path)
    old = RecordDeliveryStore(path)
    old.enqueue_pending("video_detection", DESTINATION, BROKER)
    old.claim(DESTINATION, BROKER, lease_seconds=30)
    module = importlib.import_module("src.monitoring_delivery")
    store = module.MonitoringDeliveryStore(path)
    binding = MonitoringBinding(store.origin_id, "consumer-1", DESTINATION)
    with pytest.raises(ValueError, match="destination"):
        store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    fresh = MonitoringBinding(store.origin_id, "consumer-1", DESTINATION + "-new")
    assert store.enqueue_monitoring("video_detection", fresh, BROKER, READER) == 1
    with pytest.raises(ValueError):
        store.enqueue_monitoring(
            "video_detection",
            MonitoringBinding("other-origin", "consumer-1", fresh.destination),
            BROKER,
            READER,
        )
    with pytest.raises(ValueError):
        store.enqueue_monitoring("video_detection", fresh, BROKER, "changed-reader")


def test_expired_owner_cannot_complete_new_lease(tmp_path):
    store, binding = prepared(tmp_path / "common.db")
    first = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
    assert (
        store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=129)
        is None
    )
    second = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=130)
    assert first["job_id"] == second["job_id"]
    assert not store.fail_monitoring(first, "late_timeout", now=131)
    assert store.fail_monitoring(second, "timeout", now=131)


def test_puback_and_consumer_ack_are_separate_and_persisted(tmp_path):
    store, binding = prepared(tmp_path / "common.db")
    receiver = MonitoringReceiverStore(
        tmp_path / "consumer.db", consumer_id=binding.consumer_id
    )
    receiver.register_binding(binding)
    job = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
    assert store.begin_publish(job, now=101)
    assert store.finish_publish(job, BrokerAck(job["job_id"]), ack_wait=10, now=102)
    assert store.get_monitoring(job["job_id"])["state"] == "awaiting_ack"
    assert store.get_monitoring(job["job_id"])["acked_at"] is None
    assert (
        store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=111)
        is None
    )
    retried = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=112)
    ack = receiver.accept(binding, retried["wire_payload"])
    assert store.complete_monitoring(retried, ack, now=113)
    assert store.get_monitoring(job["job_id"])["state"] == "stored"
    assert (
        json.loads(store.get_monitoring(job["job_id"])["ack_json"])["ack_kind"]
        == "consumer_committed"
    )
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT ack_kind,acked_at FROM record_delivery_state WHERE job_id=?",
            (job["job_id"],),
        ).fetchone() == ("broker_puback", 102)
    assert not store.fail_monitoring(retried, "late_failure", now=114)


def test_wrong_ack_is_retained_and_does_not_complete(tmp_path):
    store, binding = prepared(tmp_path / "common.db")
    job = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
    wrong = {"ack_kind": "broker_puback", "delivery_id": job["job_id"]}
    assert not store.complete_monitoring(job, wrong, now=101)
    assert store.get_monitoring(job["job_id"])["state"] == "uncertain"
    assert store.get_monitoring(job["job_id"])["acked_at"] is None
    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT error,response_json FROM monitoring_attempts"
        ).fetchone()
        assert row[0] == "ack_mismatch"
        assert json.loads(row[1]) == wrong


def test_v4_migration_failure_rolls_back_new_tables(tmp_path):
    path = tmp_path / "source.db"
    seed(path)
    RecordDeliveryStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE monitoring_attempts(legacy TEXT)")
    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        importlib.import_module("src.monitoring_delivery").MonitoringDeliveryStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert (
            connection.execute("SELECT count(*) FROM canonical_records").fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='monitoring_origin'"
            ).fetchone()
            is None
        )


def test_freeze_failure_rolls_back_job_and_can_retry(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    seed(store.path, "evt-2")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_freeze BEFORE INSERT ON monitoring_delivery "
            "BEGIN SELECT RAISE(ABORT, 'freeze failed'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="freeze failed"):
        store.enqueue_monitoring("video_detection", binding, BROKER, READER)
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM delivery_jobs WHERE destination=?", (DESTINATION,)
            ).fetchone()[0]
            == 1
        )
        connection.execute("DROP TRIGGER fail_freeze")
    assert store.enqueue_monitoring("video_detection", binding, BROKER, READER) == 1


def test_concurrent_claim_has_one_owner(tmp_path):
    store, binding = prepared(tmp_path / "source.db")

    def claim(_):
        return store.claim_monitoring(
            binding, BROKER, READER, lease_seconds=30, now=100
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(claim, range(2)))
    assert sum(job is not None for job in jobs) == 1


def test_malformed_record_is_quarantined_without_blocking_next(tmp_path):
    store, binding = prepared(tmp_path / "source.db")
    seed(store.path, "evt-bad")
    seed(store.path, "evt-good")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE canonical_records SET payload_json='[]' WHERE source_record_id='evt-bad'"
        )
    assert store.enqueue_monitoring("video_detection", binding, BROKER, READER) == 2
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                "SELECT m.state FROM monitoring_delivery m JOIN delivery_jobs j USING(job_id) "
                "WHERE j.source_record_id='evt-bad'"
            ).fetchone()[0]
            == "quarantined"
        )
    first = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
    second = store.claim_monitoring(binding, BROKER, READER, lease_seconds=30, now=100)
    assert {json.loads(job["wire_payload"])["event_id"] for job in (first, second)} == {
        "evt-1",
        "evt-good",
    }
