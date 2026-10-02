import json
import os
import sqlite3
import subprocess
import sys
from base64 import b64decode
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.config import AppConfig
from src.event_ingest_store import EventIngestStore
from src.services.external_ingest import ExternalIngestService, normalize_external_event

PAYLOAD = b'{"camera_id":"cam-1","type":"telemetry","timestamp":"2026-09-18T00:00:00Z","message_id":"m-1","value":0}'


def rows(store, table):
    with sqlite3.connect(store.path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.delenv("AIOT_INGEST_DB", raising=False)
    monkeypatch.delenv("MQTT_EVENT_OUTBOX_DB", raising=False)
    config = AppConfig()
    config.external_ingest.db_path = str(tmp_path / "legacy.db")
    config.external_ingest.republish_enabled = True
    config.external_ingest.mqtt_client_id = "test-external-durable-v1"
    store = EventIngestStore(tmp_path / "common.db")
    instance = ExternalIngestService(
        config.external_ingest, config.mqtt, ingest_store=store
    )
    instance._publisher = Mock()
    instance._publisher.publish_event.return_value = True
    return instance


def test_event_and_job_committed_before_publisher_and_legacy_db_unused(service):
    store = service._ingest_store

    def publish(event):
        assert len(rows(store, "ingest_receipts")) == 1
        assert rows(store, "ingest_revisions")[0]["status"] == "accepted"
        saved = rows(store, "canonical_records")[0]
        assert json.loads(saved["payload_json"]) == event
        job = rows(store, "delivery_jobs")[0]
        assert job["source_record_id"] == event["event_id"]
        assert job["state"] == "sending"
        assert json.loads(job["destination"])["broker"] == service.output_config.broker
        return True

    service._publisher.publish_event.side_effect = publish
    service.handle_message("input/test", PAYLOAD)
    assert rows(store, "delivery_jobs")[0]["state"] == "submitted"
    assert service.get_stats()["republish_count"] == 1
    assert not Path(service.ingest_config.db_path).exists()


@pytest.mark.parametrize("publisher_result", [False, RuntimeError("broken connection")])
def test_failed_publish_keeps_raw_event_and_uncertain_job(service, publisher_result):
    if isinstance(publisher_result, Exception):
        service._publisher.publish_event.side_effect = publisher_result
    else:
        service._publisher.publish_event.return_value = publisher_result
    service.handle_message("input/test", PAYLOAD)
    store = EventIngestStore(service._ingest_store.path)
    receipt = rows(store, "ingest_receipts")[0]
    assert b64decode(json.loads(receipt["raw_payload"])["payload_base64"]) == PAYLOAD
    assert len(rows(store, "canonical_records")) == 1
    job = rows(store, "delivery_jobs")[0]
    assert job["state"] == "uncertain"
    assert job["attempts"] == 1
    assert service.recover_pending_once() == 0
    assert service._publisher.publish_event.call_count == 1


def test_t1_failure_prevents_normalization_and_publish(service, monkeypatch):
    service._ingest_store.accept_message = Mock(
        side_effect=sqlite3.OperationalError("full")
    )
    normalize = Mock()
    monkeypatch.setattr(
        "src.services.external_ingest.normalize_external_event", normalize
    )
    with pytest.raises(sqlite3.OperationalError, match="full"):
        service.handle_message("input/test", PAYLOAD)
    normalize.assert_not_called()
    service._publisher.publish_event.assert_not_called()
    assert rows(service._ingest_store, "ingest_receipts") == []


def test_t2_failure_preserves_raw_and_retry_recovers_without_publishing(
    service, monkeypatch
):
    store = service._ingest_store
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_job BEFORE INSERT ON delivery_jobs "
            "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="disk full"):
        service.handle_message("input/test", PAYLOAD)
    assert len(rows(store, "ingest_receipts")) == 1
    assert rows(store, "canonical_records") == []
    assert rows(store, "ingest_revisions") == []
    service._publisher.publish_event.assert_not_called()
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER fail_job")
    monkeypatch.setattr("src.event_ingest_store.time.time", lambda: 1e12)
    assert service.recover_pending_once() == 1
    assert rows(store, "delivery_jobs")[0]["state"] == "held"
    service._publisher.publish_event.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\x00",
        b"{invalid",
        b"[]",
        b'{"camera_id":null}',
        b'{"camera_id":"cam-1","value":NaN}',
        b'{"camera_id":"cam-1","confidence":"inf"}',
    ],
)
def test_invalid_input_bytes_are_preserved_without_delivery(service, payload):
    service.handle_message("input/test", payload)
    store = service._ingest_store
    receipt = rows(store, "ingest_receipts")[0]
    assert b64decode(json.loads(receipt["raw_payload"])["payload_base64"]) == payload
    assert rows(store, "ingest_revisions")[0]["status"] == "invalid"
    assert rows(store, "canonical_records") == []
    assert rows(store, "delivery_jobs") == []
    service._publisher.publish_event.assert_not_called()


def test_duplicate_and_conflict_are_preserved_without_second_publish(service):
    service.handle_message("input/test", PAYLOAD)
    service.handle_message("input/test", PAYLOAD)
    service.handle_message("input/test", PAYLOAD.replace(b'"value":0', b'"value":1'))
    store = service._ingest_store
    assert len(rows(store, "ingest_receipts")) == 3
    assert [r["status"] for r in rows(store, "ingest_revisions")] == [
        "accepted",
        "duplicate",
        "conflict",
    ]
    assert len(rows(store, "canonical_records")) == 1
    assert len(rows(store, "delivery_jobs")) == 1
    service._publisher.publish_event.assert_called_once()


def test_recovery_keeps_original_received_time_and_event_id(service):
    store = service._ingest_store
    payload = b'{"camera_id":"cam-1","type":"telemetry"}'
    ingest_id = store.accept_message(
        "external_mqtt", "input/test", payload, destination="original-route"
    )
    received_at = store.get_event_receipt(ingest_id)["received_at"]
    expected = normalize_external_event(
        json.loads(payload), "input/test", received_at=received_at
    )
    assert service.recover_pending_once() == 1
    canonical = rows(store, "canonical_records")[0]
    assert json.loads(canonical["payload_json"]) == expected
    assert canonical["source_record_id"] == expected["event_id"]
    assert rows(store, "delivery_jobs")[0]["destination"] == "original-route"
    assert service.recover_pending_once() == 0
    service._publisher.publish_event.assert_not_called()


def test_start_recovers_receipts_before_connect_without_legacy_database(service):
    store = service._ingest_store
    store.accept_message(
        "external_mqtt", "input/test", PAYLOAD, destination="original-route"
    )

    def connect():
        assert rows(store, "delivery_jobs")[0]["state"] == "held"
        return True

    service._subscriber = Mock()
    service._subscriber.connect.side_effect = connect
    service.start()
    assert service._running is True
    assert not Path(service.ingest_config.db_path).exists()
    service._publisher.publish_event.assert_not_called()
    service.stop()


def test_recovery_backoff_does_not_block_later_receipts(service, monkeypatch):
    store = service._ingest_store
    first = store.accept_message("external_mqtt", "broken", PAYLOAD, destination=None)
    second = store.accept_message("external_mqtt", "good", PAYLOAD, destination=None)

    def normalize(raw, topic, **kwargs):
        if topic == "broken":
            raise RuntimeError("temporary normalization failure")
        return normalize_external_event(raw, topic, **kwargs)

    monkeypatch.setattr(
        "src.services.external_ingest.normalize_external_event", normalize
    )
    assert service.recover_pending_once() == 1
    assert store.get_event_receipt(first)["stage"] == "pending"
    assert store.get_event_receipt(first)["attempts"] == 1
    assert store.get_event_receipt(second)["stage"] == "accepted"
    assert store.pending_events("external_mqtt") == []
    assert service.recover_pending_once() == 0
    service._publisher.publish_event.assert_not_called()


def test_env_selects_common_store_and_legacy_replay_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AIOT_INGEST_DB", str(tmp_path / "common.db"))
    monkeypatch.delenv("MQTT_EVENT_OUTBOX_DB", raising=False)
    config = AppConfig()
    config.external_ingest.mqtt_client_id = "test-external-env-v1"
    instance = ExternalIngestService.from_app_config(config)
    assert instance._ingest_store.path == tmp_path / "common.db"
    instance.handle_message("input/test", PAYLOAD)
    assert rows(instance._ingest_store, "delivery_jobs") == []
    monkeypatch.setenv("MQTT_EVENT_OUTBOX_DB", str(tmp_path / "legacy-outbox.db"))
    config.external_ingest.republish_enabled = True
    with pytest.raises(ValueError, match="cannot run together"):
        ExternalIngestService.from_app_config(config)
    assert not (tmp_path / "legacy-outbox.db").exists()


CRASH_SCRIPT = r"""
import os, sys
from src.config import AppConfig
from src.event_ingest_store import EventIngestStore
from src.services.external_ingest import ExternalIngestService
path, checkpoint, payload = sys.argv[1:]
store = EventIngestStore(path)
config = AppConfig()
config.external_ingest.republish_enabled = True
config.external_ingest.mqtt_client_id = "test-external-crash-v1"
service = ExternalIngestService(config.external_ingest, config.mqtt, ingest_store=store)
class Publisher:
    def publish_event(self, event):
        if checkpoint == "during_publish":
            os._exit(42)
        return True
service._publisher = Publisher()
method = {"receipt": "accept_message", "normalized": "record_event",
          "submitted": "finish_live_delivery"}.get(checkpoint)
if method:
    original = getattr(store, method)
    def stop_after(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(42)
    setattr(store, method, stop_after)
service.handle_message("input/test", payload.encode())
raise AssertionError("crash checkpoint not reached")
"""


@pytest.mark.parametrize(
    "checkpoint,state",
    [
        ("receipt", "held"),
        ("normalized", "pending"),
        ("during_publish", "sending"),
        ("submitted", "submitted"),
    ],
)
def test_real_process_exit_recovery_keeps_ids_and_never_replays_alarms(
    service, checkpoint, state
):
    store = service._ingest_store
    environment = os.environ.copy()
    environment.pop("AIOT_INGEST_DB", None)
    environment.pop("MQTT_EVENT_OUTBOX_DB", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            CRASH_SCRIPT,
            str(store.path),
            checkpoint,
            PAYLOAD.decode(),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        timeout=20,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 42, result.stderr
    before = rows(store, "ingest_receipts")
    expected_id = normalize_external_event(json.loads(PAYLOAD), "input/test")[
        "event_id"
    ]
    service.recover_pending_once()
    service.recover_pending_once()
    assert rows(store, "ingest_receipts") == before
    assert len(before) == 1
    canonical = rows(store, "canonical_records")
    assert len(canonical) == 1
    assert canonical[0]["first_ingest_id"] == before[0]["ingest_id"]
    assert canonical[0]["source_record_id"] == expected_id
    assert len(rows(store, "ingest_revisions")) == 1
    jobs = rows(store, "delivery_jobs")
    assert len(jobs) == 1
    assert jobs[0]["source_record_id"] == expected_id
    assert jobs[0]["state"] == state
    service._publisher.publish_event.assert_not_called()
