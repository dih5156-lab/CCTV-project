"""3종 합성 사건이 기존 저장·관제 경계를 통과할 때 근거와 식별자를 보존한다."""

import copy
import json
from pathlib import Path

import pytest

from src.core._event_publish import normalize_event_queue_item
from src.core.video_event_ingest import VideoEventIngest
from src.event_ingest_store import EventIngestStore
from src.monitoring_contract import parse_record
from src.record_delivery import RecordDeliveryStore

EXAMPLES = json.loads(
    (Path(__file__).parent / "fixtures/behavior_event_contract.json").read_text()
)["examples"]


@pytest.fixture(params=EXAMPLES, ids=lambda event: event["type"])
def example(request):
    return copy.deepcopy(request.param)


def test_three_event_payloads_survive_storage_and_record_delivery(tmp_path, example):
    original = copy.deepcopy(example)
    store = EventIngestStore(tmp_path / "events.db")
    ingest = VideoEventIngest(
        store, backend="contract-test", topic_prefix="test/events"
    )
    reference = ingest.accept(example, example["camera_id"])
    # 접수 뒤 생산자가 메타를 바꿔도 저장·전달되는 사건 근거는 바뀌면 안 된다.
    example["metadata"]["behavior"]["window_end_pts_seconds"] = 999
    sent = []
    assert ingest.publish(
        reference, mqtt_publish=lambda *args: sent.append(args), event_publisher=None
    )
    topic, payload = sent[0]
    assert topic == f'test/events/{original["camera_id"]}/{original["type"]}'
    assert payload["event_id"] == original["event_id"]
    assert payload["event"]["event_type"] == original["type"]
    assert payload["metadata"] == payload["raw"]["metadata"] == original["metadata"]
    assert payload["timestamp"] == original["timestamp"]
    assert payload["device"]["camera_id"] == original["camera_id"]

    # 실제 네트워크와 장치를 사용하지 않고 기존 관제 전용 직렬화/수신 계약을 대조한다.
    records = RecordDeliveryStore(store.path)
    destination, endpoint = (
        "record-mqtt:cctv/records/v1/contract",
        "unix:/tmp/contract-unused",
    )
    assert records.enqueue_pending("video_detection", destination, endpoint) == 1
    claimed = records.claim(destination, endpoint, lease_seconds=10, now=100)
    received = parse_record(claimed["wire_payload"].encode())
    assert received.record_id == original["event_id"]
    assert json.loads(received.record_json) == payload
    assert records.enqueue_pending("video_detection", destination, endpoint) == 0


def test_same_incident_redelivery_is_not_a_second_event(tmp_path, example):
    store = EventIngestStore(tmp_path / "events.db")
    ingest = VideoEventIngest(
        store, backend="contract-test", topic_prefix="test/events"
    )
    sent = []
    for _ in range(2):
        reference = ingest.accept(example, example["camera_id"])
        ingest.publish(
            reference,
            mqtt_publish=lambda *args: sent.append(args),
            event_publisher=None,
        )
    assert len(sent) == 1
    with store._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM ingest_receipts").fetchone()[0]
            == 2
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM canonical_records").fetchone()[0]
            == 1
        )
        assert [
            r[0]
            for r in connection.execute(
                "SELECT status FROM ingest_revisions ORDER BY rowid"
            )
        ] == ["accepted", "duplicate"]


def test_same_event_id_with_changed_evidence_is_conflict(tmp_path, example):
    store = EventIngestStore(tmp_path / "events.db")
    ingest = VideoEventIngest(
        store, backend="contract-test", topic_prefix="test/events"
    )
    sent = []
    first = ingest.accept(example, example["camera_id"])
    assert ingest.publish(
        first, mqtt_publish=lambda *args: sent.append(args), event_publisher=None
    )
    example["metadata"]["behavior"]["stream_session_id"] = "another-stream-session"
    second = ingest.accept(example, example["camera_id"])
    assert not ingest.publish(
        second, mqtt_publish=lambda *args: sent.append(args), event_publisher=None
    )
    assert len(sent) == 1
    assert store.get_event_receipt(second.ingest_id)["stage"] == "conflict"
    with store._connect() as connection:
        saved = json.loads(
            connection.execute("SELECT payload_json FROM canonical_records").fetchone()[
                0
            ]
        )
    assert saved["metadata"] == sent[0][1]["metadata"]


def test_restart_preserves_incident_without_resuming_alarm(tmp_path, example):
    store = EventIngestStore(tmp_path / "events.db")
    original = VideoEventIngest(
        store, backend="contract-test", topic_prefix="test/events"
    )
    reference = original.accept(example, example["camera_id"])
    restarted = VideoEventIngest(
        EventIngestStore(store.path),
        backend="contract-test",
        topic_prefix="test/events",
    )
    restarted.recover_pending()
    assert store.get_event_receipt(reference.ingest_id)["stage"] == "accepted"
    with store._connect() as connection:
        assert (
            connection.execute("SELECT state FROM delivery_jobs").fetchone()[0]
            == "held"
        )
        saved = json.loads(
            connection.execute("SELECT payload_json FROM canonical_records").fetchone()[
                0
            ]
        )
    assert saved["event_id"] == example["event_id"]
    assert saved["metadata"] == example["metadata"]


@pytest.mark.parametrize("object_id", [0, 18446744073709551614, None])
def test_participant_ids_keep_large_values_and_missingness(example, object_id):
    example["metadata"]["behavior"]["participants"][0]["object_id"] = object_id
    original = copy.deepcopy(example)
    payload, _, _ = normalize_event_queue_item(example, default_backend="contract-test")
    wanted = None if object_id is None else str(object_id)
    for path in (payload["metadata"], payload["raw"]["metadata"]):
        assert path["behavior"]["participants"][0]["object_id"] == wanted
    assert example == original


@pytest.mark.parametrize(
    "field,value", [("timestamp", float("nan")), ("camera_id", "unknown")]
)
def test_invalid_time_or_camera_is_saved_without_false_publication(
    tmp_path, example, field, value
):
    example[field] = value
    store = EventIngestStore(tmp_path / "events.db")
    ingest = VideoEventIngest(
        store, backend="contract-test", topic_prefix="test/events"
    )
    reference = ingest.accept(example, example["camera_id"])
    sent = []
    assert not ingest.publish(
        reference, mqtt_publish=lambda *args: sent.append(args), event_publisher=None
    )
    assert store.get_event_receipt(reference.ingest_id)["stage"] == "invalid"
    assert not sent
