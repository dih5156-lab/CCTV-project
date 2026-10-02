"""Durable record delivery must not release old physical alarm jobs."""

import importlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.event_ingest_store import EventIngestStore
from src.ingest_store import IngestStore

DESTINATION = "record-mqtt:cctv/records/v1/lab"
ENDPOINT = "unix:/tmp/test-broker"


def delivery_module():
    return importlib.import_module("src.record_delivery")


def seed(path, event_id="evt-1", *, recovery=True, namespace="video_detection"):
    store = EventIngestStore(path)
    receipt = store.accept_event_json(
        namespace, {"event_id": event_id, "value": 0}, destination="mqtt:cctv/ai/events"
    )
    _, alarm_job = store.record_event(
        receipt,
        {"event_id": event_id, "value": 0},
        converter_version="test-v1",
        recovery=recovery,
    )
    return receipt, alarm_job


def rows(path, table):
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]


def prepared(tmp_path):
    path = tmp_path / "common.db"
    seed(path)
    store = delivery_module().RecordDeliveryStore(path)
    assert store.enqueue_pending("video_detection", DESTINATION, ENDPOINT) == 1
    return store


def test_additive_upgrade_keeps_sensor_and_alarm_jobs(tmp_path):
    path = tmp_path / "common.db"
    sensor = IngestStore(path).accept("tlv_mqtt", "sensor", {"value": 0})
    receipt, alarm_job = seed(path)
    store = delivery_module().RecordDeliveryStore(path)
    assert IngestStore(path).pending() == [sensor]
    assert EventIngestStore(path).get_event_receipt(receipt)["stage"] == "accepted"
    assert store.enqueue_pending("video_detection", DESTINATION, ENDPOINT) == 1
    assert store.enqueue_pending("video_detection", DESTINATION, ENDPOINT) == 0
    jobs = rows(path, "delivery_jobs")
    assert len(jobs) == 2
    assert next(j for j in jobs if j["job_id"] == alarm_job)["state"] == "held"
    assert len(rows(path, "canonical_records")) == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (3,)


def test_atomic_claim_excludes_other_worker_and_stale_completion(tmp_path):
    store = prepared(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(
            pool.map(
                lambda _: store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=100),
                range(2),
            )
        )
    active = [claim for claim in claims if claim is not None]
    assert len(active) == 1
    first = active[0]
    assert store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=109) is None
    second = store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=110)
    assert second["job_id"] == first["job_id"]
    assert second["lease_token"] != first["lease_token"]
    assert not store.finish(first, acknowledged=True, now=111)
    assert store.finish(second, acknowledged=True, now=111)
    attempts = rows(store.path, "record_delivery_attempts")
    assert [a["outcome"] for a in attempts] == ["lease_expired", "broker_acked"]
    assert store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=1000) is None


def test_retry_is_due_later_and_payload_is_identical(tmp_path):
    store = prepared(tmp_path)
    first = store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=100)
    assert store.finish(first, acknowledged=False, error="TimeoutError", now=101)
    assert store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=101.5) is None
    second = store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=102)
    assert second["wire_payload"] == first["wire_payload"]
    envelope = json.loads(second["wire_payload"])
    assert envelope == {
        "schema_version": "1.0",
        "delivery_id": first["job_id"],
        "source_namespace": "video_detection",
        "event_id": "evt-1",
        "record": {"event_id": "evt-1", "value": 0},
    }
    assert second["attempts"] == 2
    assert rows(store.path, "record_delivery_state")[0]["acked_at"] is None


def test_other_destination_progresses_and_endpoint_cannot_drift(tmp_path):
    store = prepared(tmp_path)
    other = "record-mqtt:cctv/records/v1/other"
    assert store.enqueue_pending("video_detection", other, ENDPOINT) == 1
    first = store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=100)
    assert store.claim(other, ENDPOINT, lease_seconds=10, now=100) is not None
    with pytest.raises(ValueError, match="endpoint"):
        store.enqueue_pending("video_detection", DESTINATION, "unix:/tmp/changed")
    assert store.finish(first, acknowledged=True, now=101)
    with pytest.raises(ValueError, match="endpoint"):
        store.claim(DESTINATION, "unix:/tmp/changed", lease_seconds=10, now=102)


@pytest.mark.parametrize(
    "destination",
    [
        "mqtt:cctv/ai/events",
        "record-mqtt:cctv/ai/events",
        "record-mqtt:cctv/records/v1/#",
    ],
)
def test_alarm_and_wildcard_destinations_rejected(tmp_path, destination):
    store = delivery_module().RecordDeliveryStore(tmp_path / "common.db")
    with pytest.raises(ValueError):
        store.enqueue_pending("video_detection", destination, ENDPOINT)


def test_legacy_live_publisher_cannot_claim_record_delivery(tmp_path):
    store = prepared(tmp_path)
    record_job = rows(store.path, "record_delivery_state")[0]["job_id"]
    legacy = EventIngestStore(store.path)
    assert not legacy.begin_live_delivery(record_job)
    claim = store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=100)
    legacy.finish_live_delivery(record_job, submitted=True, error=None)
    assert (
        next(j for j in rows(store.path, "delivery_jobs") if j["job_id"] == record_job)[
            "state"
        ]
        == "sending"
    )
    assert store.finish(claim, acknowledged=False, now=101)


def test_worker_requires_matching_broker_ack_and_retains_failure(tmp_path):
    store = prepared(tmp_path)
    module = delivery_module()

    class Sender:
        endpoint = ENDPOINT

        def send(self, job, *, timeout):
            # A legacy bool is not a broker acknowledgement.
            return True

    worker = module.RecordDeliveryWorker(
        store, Sender(), namespace="video_detection", destination=DESTINATION
    )
    assert worker.drain_once() == 0
    state = rows(store.path, "record_delivery_state")[0]
    assert state["acked_at"] is None
    assert rows(store.path, "record_delivery_attempts")[0]["outcome"] == "uncertain"
    assert state["next_attempt_at"] > time.time()


def test_worker_wait_does_not_hold_database_lock(tmp_path):
    store = prepared(tmp_path)
    module = delivery_module()

    class Sender:
        endpoint = ENDPOINT

        def send(self, job, *, timeout):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(seed, store.path, "evt-2").result(timeout=2)
            return module.BrokerAck(job["job_id"])

    worker = module.RecordDeliveryWorker(
        store, Sender(), namespace="video_detection", destination=DESTINATION
    )
    assert worker.drain_once(limit=1) == 1
    assert len(rows(store.path, "canonical_records")) == 2
    state = rows(store.path, "record_delivery_state")[0]
    assert state["ack_kind"] == "broker_puback"
    assert state["acked_at"] is not None


def test_schema_upgrade_failure_rolls_back_without_losing_v2(tmp_path):
    path = tmp_path / "common.db"
    receipt, alarm_job = seed(path)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE record_delivery_state(conflict TEXT)")
    with pytest.raises(sqlite3.OperationalError):
        delivery_module().RecordDeliveryStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='record_delivery_targets'"
            ).fetchone()
            is None
        )
    assert EventIngestStore(path).get_event_receipt(receipt)["stage"] == "accepted"
    assert rows(path, "delivery_jobs")[0]["job_id"] == alarm_job


def test_finish_failure_rolls_back_ack_and_keeps_lease(tmp_path):
    store = prepared(tmp_path)
    claim = store.claim(DESTINATION, ENDPOINT, lease_seconds=10, now=100)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_finish BEFORE UPDATE ON record_delivery_attempts "
            "WHEN NEW.outcome='broker_acked' BEGIN SELECT RAISE(ABORT, 'disk failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.finish(claim, acknowledged=True, now=101)
    state = rows(store.path, "record_delivery_state")[0]
    assert state["acked_at"] is None
    assert state["lease_token"] == claim["lease_token"]
    assert rows(store.path, "record_delivery_attempts")[0]["outcome"] == "sending"


def test_enqueue_failure_preserves_canonical_for_next_scan(tmp_path):
    path = tmp_path / "common.db"
    seed(path)
    store = delivery_module().RecordDeliveryStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_enqueue BEFORE INSERT ON record_delivery_state "
            "BEGIN SELECT RAISE(ABORT, 'disk failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.enqueue_pending("video_detection", DESTINATION, ENDPOINT)
    assert len(rows(path, "delivery_jobs")) == 1
    assert len(rows(path, "canonical_records")) == 1
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER fail_enqueue")
    assert store.enqueue_pending("video_detection", DESTINATION, ENDPOINT) == 1


def test_video_recovery_flows_to_record_worker_without_alarm_replay(tmp_path):
    from src.core.video_event_ingest import VideoEventIngest

    module = delivery_module()
    store = module.RecordDeliveryStore(tmp_path / "common.db")
    video = VideoEventIngest(store, backend="deepstream", topic_prefix="cctv/ai/events")
    receipt = video.accept(
        {"type": "fall", "camera_id": "test", "timestamp": 0}, "test"
    )
    video.defer(receipt)
    video.recover_pending()
    assert len(rows(store.path, "canonical_records")) == 1

    class Sender:
        endpoint = ENDPOINT

        def send(self, job, *, timeout):
            envelope = json.loads(job["wire_payload"])
            assert envelope["record"]["type"] == "fall"
            assert envelope["record"]["occurred_at"].startswith("1970-01-01")
            return module.BrokerAck(job["job_id"])

    worker = module.RecordDeliveryWorker(
        store, Sender(), namespace="video_detection", destination=DESTINATION
    )
    assert worker.drain_once() == 1
    assert [j["state"] for j in rows(store.path, "delivery_jobs")] == [
        "held",
        "submitted",
    ]
    assert len(rows(store.path, "ingest_receipts")) == 1


@pytest.mark.parametrize(
    "timeout,lease", [(0, 10), (float("nan"), 10), (1, float("inf")), (5, 6)]
)
def test_invalid_worker_timing_cannot_claim_jobs(tmp_path, timeout, lease):
    store = prepared(tmp_path)
    module = delivery_module()
    with pytest.raises(ValueError):
        module.RecordDeliveryWorker(
            store,
            object(),
            namespace="video_detection",
            destination=DESTINATION,
            send_timeout=timeout,
            lease_seconds=lease,
        )
    assert rows(store.path, "record_delivery_attempts") == []
