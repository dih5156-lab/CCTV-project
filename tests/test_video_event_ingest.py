import importlib
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest

from src.core._deepstream_event_queue import enqueue_queue_item
from src.core._event_publish import run_publish_loop
from src.core.events import DetectionEvent, EventType
from src.event_ingest_store import EventIngestStore


def video_module():
    return importlib.import_module("src.core.video_event_ingest")


def detection(timestamp=1700000000.0):
    return DetectionEvent(
        EventType.PERSON,
        1,
        2,
        30,
        40,
        0.8,
        timestamp,
        object_id=0,
        metadata={"camera_id": "cam1", "frame_num": 7},
    )


def rows(store, table):
    with store._connect() as connection:
        return [
            dict(row)
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
        ]


def ingestor(tmp_path):
    store = EventIngestStore(tmp_path / "common.db")
    return video_module().VideoEventIngest(
        store, backend="deepstream", topic_prefix="cctv/events"
    )


def enqueue(ingest, event_queue, item):
    return enqueue_queue_item(
        event_queue=event_queue,
        queue_item=item,
        camera_name="cam1",
        increment_stat=lambda key: None,
        video_ingest=ingest,
    )


def test_raw_commits_before_queue_and_freezes_producer_mutations(tmp_path):
    ingest = ingestor(tmp_path)
    queue = Queue()
    item = detection()
    assert enqueue(ingest, queue, item)
    receipt = rows(ingest.store, "ingest_receipts")[0]
    assert rows(ingest.store, "canonical_records") == []
    item.metadata["frame_num"] = 99
    delivered = []
    assert ingest.publish(
        queue.get_nowait(),
        mqtt_publish=lambda topic, data: delivered.append((topic, data)),
        event_publisher=None,
    )
    assert delivered[0][0] == "cctv/events/cam1/person"
    assert delivered[0][1]["raw"]["metadata"]["frame_num"] == 7
    assert delivered[0][1]["observation"]["received_at"] == receipt["received_at"]
    assert rows(ingest.store, "delivery_jobs")[0]["state"] == "submitted"


def test_full_queue_retains_new_receipt_for_record_only_recovery(tmp_path):
    ingest = ingestor(tmp_path)
    queue = Queue(maxsize=1)
    assert enqueue(ingest, queue, detection())
    assert not enqueue(ingest, queue, detection(1700000001))
    ingest.recover_pending()
    assert len(rows(ingest.store, "ingest_receipts")) == 2
    assert len(rows(ingest.store, "canonical_records")) == 1
    assert rows(ingest.store, "delivery_jobs")[0]["state"] == "held"
    assert rows(ingest.store, "event_ingest_state")[0]["stage"] == "pending"


def test_failed_t1_propagates_and_never_queues(tmp_path):
    ingest = ingestor(tmp_path)
    with ingest.store._connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_state BEFORE INSERT ON event_ingest_state BEGIN SELECT RAISE(ABORT, 'test disk failure'); END"
        )
    queue = Queue()
    with pytest.raises(sqlite3.IntegrityError):
        enqueue(ingest, queue, detection())
    assert queue.empty() and rows(ingest.store, "ingest_receipts") == []


@pytest.mark.parametrize(
    "item",
    [
        detection(float("nan")),
        {"type": "person", "camera_id": "cam1"},
        {"timestamp": 1700000000, "metadata": []},
        [["type", "person"], ["camera_id", "cam1"], ["timestamp", 1700000000]],
    ],
)
def test_invalid_input_is_persisted_then_quarantined_without_publication(
    tmp_path, item
):
    ingest = ingestor(tmp_path)
    reference = ingest.accept(item, "cam1")
    sent = []
    assert not ingest.publish(
        reference, mqtt_publish=lambda *args: sent.append(args), event_publisher=None
    )
    assert len(rows(ingest.store, "ingest_receipts")) == 1
    assert rows(ingest.store, "event_ingest_state")[0]["stage"] == "invalid"
    assert rows(ingest.store, "ingest_revisions")[0]["errors_json"]
    assert not sent and rows(ingest.store, "canonical_records") == []


def test_publish_failure_keeps_canonical_event_and_uncertain_job(tmp_path):
    ingest = ingestor(tmp_path)
    queue = Queue()
    enqueue(ingest, queue, detection())
    stop = Event()
    stats = []

    def fail(topic, payload):
        stop.set()
        raise OSError("test connection lost")

    run_publish_loop(
        is_running=lambda: True,
        stop_event=stop,
        event_queue=queue,
        topic_prefix="cctv/events",
        mqtt_publish=fail,
        event_publisher=None,
        increment_stat=stats.append,
        video_ingest=ingest,
    )
    assert queue.empty() and stats == ["events_failed"]
    assert len(rows(ingest.store, "canonical_records")) == 1
    assert rows(ingest.store, "delivery_jobs")[0]["state"] == "uncertain"


def test_restarted_ingestor_recovers_without_using_new_destination(tmp_path):
    ingest = ingestor(tmp_path)
    reference = ingest.accept(detection(), "cam1")
    resumed = video_module().VideoEventIngest(
        EventIngestStore(ingest.store.path), backend="opencv", topic_prefix="changed"
    )
    resumed.recover_pending()
    resumed.recover_pending()
    record = rows(ingest.store, "canonical_records")[0]
    assert record["first_ingest_id"] == reference.ingest_id
    assert json.loads(record["payload_json"])["backend"] == "deepstream"
    assert rows(ingest.store, "delivery_jobs")[0]["destination"] == "mqtt:cctv/events"
    assert rows(ingest.store, "delivery_jobs")[0]["state"] == "held"


def test_epoch_zero_is_stable_across_repeated_normalization(tmp_path):
    ingest = ingestor(tmp_path)
    for _ in range(2):
        reference = ingest.accept(detection(0), "cam1")
        ingest.defer(reference)
        ingest.recover_pending()
    records = rows(ingest.store, "canonical_records")
    assert len(records) == 1
    assert (
        json.loads(records[0]["payload_json"])["occurred_at"]
        == "1970-01-01T09:00:00+09:00"
    )
    assert [r["stage"] for r in rows(ingest.store, "event_ingest_state")] == [
        "accepted",
        "duplicate",
    ]


def test_common_store_rejects_legacy_outbox_replay(monkeypatch, tmp_path):
    monkeypatch.setenv("AIOT_INGEST_DB", str(tmp_path / "common.db"))
    monkeypatch.setenv("MQTT_EVENT_OUTBOX_DB", str(tmp_path / "outbox.db"))
    with pytest.raises(ValueError, match="outbox"):
        video_module().VideoEventIngest.from_env(
            backend="deepstream", topic_prefix="cctv/events"
        )


@pytest.mark.parametrize("entry", ["_enqueue_queue_item", "_put_event_dict"])
def test_deepstream_entry_points_use_durable_intake(tmp_path, entry):
    from src.core.deepstream_processor import DeepStreamProcessor

    processor = object.__new__(DeepStreamProcessor)
    processor.event_queue = Queue()
    processor._increment_stat = lambda key: None
    processor._video_ingest = ingestor(tmp_path)
    assert getattr(processor, entry)(
        {"type": "person", "timestamp": 1700000000}, "cam1"
    )
    assert len(rows(processor._video_ingest.store, "ingest_receipts")) == 1
    assert isinstance(processor.event_queue.get_nowait(), video_module().VideoReceipt)


def test_opencv_dispatch_and_worker_use_same_receipt(tmp_path):
    from src.core.event_dispatcher import EventDispatcher
    from src.core.processor import VideoProcessor

    ingest = ingestor(tmp_path)
    processor = object.__new__(VideoProcessor)
    processor._video_ingest = ingest
    processor.event_queue = Queue()
    processor.stop_event = Event()
    processor.running = True
    stats, delivered = [], []
    processor._increment_stat = stats.append
    processor.config = SimpleNamespace(
        processing=SimpleNamespace(consecutive_failure_threshold=3)
    )

    class Publisher:
        def publish_event(self, payload):
            delivered.append(payload)
            processor.stop_event.set()
            return True

    processor.event_publisher = Publisher()
    dispatcher = EventDispatcher(
        processor.event_queue, None, stats.append, video_ingest=ingest
    )
    assert dispatcher.enqueue("cam1", detection())
    processor._send_events_worker()
    assert stats == ["events_detected", "events_sent"]
    assert delivered[0]["camera_id"] == "cam1"
    assert rows(ingest.store, "delivery_jobs")[0]["state"] == "submitted"


def test_t2_failure_keeps_receipt_and_recovers_once(tmp_path):
    ingest = ingestor(tmp_path)
    reference = ingest.accept(detection(), "cam1")
    with ingest.store._connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_revision BEFORE INSERT ON ingest_revisions BEGIN SELECT RAISE(ABORT, 'test T2 failure'); END"
        )
    sent = []
    with pytest.raises(sqlite3.IntegrityError):
        ingest.publish(
            reference,
            mqtt_publish=lambda *args: sent.append(args),
            event_publisher=None,
        )
    assert not sent and rows(ingest.store, "canonical_records") == []
    with ingest.store._connect() as connection:
        connection.execute("DROP TRIGGER fail_revision")
        connection.execute("UPDATE event_ingest_state SET next_attempt_at=0")
    ingest.recover_pending()
    assert len(rows(ingest.store, "canonical_records")) == 1
    assert rows(ingest.store, "delivery_jobs")[0]["state"] == "held"


def test_callback_false_is_not_recorded_as_success(tmp_path):
    ingest = ingestor(tmp_path)
    reference = ingest.accept(detection(), "cam1")
    assert not ingest.publish(
        reference, mqtt_publish=lambda *args: False, event_publisher=None
    )
    assert rows(ingest.store, "delivery_jobs")[0]["state"] == "uncertain"


def test_bad_converted_values_are_quarantined_not_retried_forever(tmp_path):
    ingest = ingestor(tmp_path)
    item = detection()
    item.confidence = "NaN"
    reference = ingest.accept(item, "cam1")
    assert not ingest.publish(
        reference,
        mqtt_publish=lambda *args: pytest.fail("invalid publish"),
        event_publisher=None,
    )
    assert rows(ingest.store, "event_ingest_state")[0]["stage"] == "invalid"


def test_destination_configuration_does_not_change_source_hash(tmp_path):
    ingest = ingestor(tmp_path)
    first = ingest.accept(detection(), "cam1")
    ingest.defer(first)
    ingest.recover_pending()
    changed = video_module().VideoEventIngest(
        ingest.store, backend="deepstream", topic_prefix="changed"
    )
    second = changed.accept(detection(), "cam1")
    changed.defer(second)
    changed.recover_pending()
    assert [r["stage"] for r in rows(ingest.store, "event_ingest_state")] == [
        "accepted",
        "duplicate",
    ]
    assert len(rows(ingest.store, "delivery_jobs")) == 1


def test_blocked_publisher_does_not_hold_intake_lock(tmp_path):
    ingest = ingestor(tmp_path)
    reference = ingest.accept(detection(), "cam1")
    entered, release = Event(), Event()

    def slow_output(topic, event):
        entered.set()
        assert release.wait(20)

    with ThreadPoolExecutor(max_workers=2) as executor:
        publishing = executor.submit(
            ingest.publish, reference, mqtt_publish=slow_output, event_publisher=None
        )
        try:
            assert entered.wait(10)
            accepting = executor.submit(ingest.accept, detection(1700000001), "cam1")
            second = accepting.result(timeout=10)
            assert (
                ingest.store.get_event_receipt(second.ingest_id)["stage"] == "pending"
            )
            assert not publishing.done()
        finally:
            release.set()
        assert publishing.result(timeout=10)


def test_opencv_pipeline_retains_invalid_metadata_before_to_dict(tmp_path):
    from src.core._inference_pipeline import _InferencePipeline
    from src.core.event_dispatcher import EventDispatcher

    ingest = ingestor(tmp_path)
    queue = Queue()
    pipeline = object.__new__(_InferencePipeline)
    pipeline._track_manager = SimpleNamespace(
        get_frame_count=lambda *args: 1, min_track_frames=1
    )
    pipeline._debouncer = SimpleNamespace(should_send=lambda *args, **kwargs: True)
    pipeline._event_dispatcher = EventDispatcher(
        queue, None, lambda key: None, video_ingest=ingest
    )
    item = detection()
    item.metadata = [["invalid"]]
    pipeline._enqueue("cam1", [item], [])
    assert len(rows(ingest.store, "ingest_receipts")) == 1
    assert not ingest.publish(
        queue.get_nowait(),
        mqtt_publish=lambda *args: pytest.fail("invalid publish"),
        event_publisher=None,
    )
    assert rows(ingest.store, "event_ingest_state")[0]["stage"] == "invalid"


def test_explicit_id_collision_preserves_original_and_blocks_second_output(tmp_path):
    ingest = ingestor(tmp_path)
    sent = []
    for value in (0.8, 0.1):
        reference = ingest.accept(
            {
                "event_id": "producer-event",
                "timestamp": 1700000000,
                "type": "person",
                "camera_id": "cam1",
                "confidence": value,
            },
            "cam1",
        )
        ingest.publish(
            reference,
            mqtt_publish=lambda *args: sent.append(args),
            event_publisher=None,
        )
    assert len(sent) == 1
    assert [r["stage"] for r in rows(ingest.store, "event_ingest_state")] == [
        "accepted",
        "conflict",
    ]
    assert (
        json.loads(rows(ingest.store, "canonical_records")[0]["payload_json"])[
            "confidence"
        ]
        == 0.8
    )


CRASH_SCRIPT = r"""
import os, sys
from pathlib import Path
from queue import Queue
from src.core.events import DetectionEvent, EventType
from src.core.video_event_ingest import VideoEventIngest
from src.event_ingest_store import EventIngestStore
path, checkpoint, output = sys.argv[1:]
store = EventIngestStore(path)
ingest = VideoEventIngest(store, backend="deepstream", topic_prefix="cctv/events")
if checkpoint == "after_T1":
    original = store.accept_event_json
    def accept(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(42)
    store.accept_event_json = accept
elif checkpoint == "after_T2":
    original = store.record_event
    def record(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(42)
    store.record_event = record
elif checkpoint == "after_claim":
    original = store.begin_live_delivery
    def begin(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(42)
    store.begin_live_delivery = begin
reference = ingest.accept(DetectionEvent(EventType.FALL_DETECTED, 1, 2, 30, 40, 0.9, 1700000000, metadata={"camera_id":"cam1"}), "cam1")
queue = Queue()
queue.put_nowait(reference)
if checkpoint == "after_queue":
    os._exit(42)
def publish(topic, event):
    Path(output).write_text(event["event_id"])
    os._exit(42)
ingest.publish(queue.get_nowait(), mqtt_publish=publish, event_publisher=None)
raise AssertionError("crash not reached")
"""


@pytest.mark.parametrize(
    "checkpoint,state,was_sent",
    [
        ("after_T1", "held", False),
        ("after_queue", "held", False),
        ("after_T2", "pending", False),
        ("after_claim", "sending", False),
        ("after_publish", "sending", True),
    ],
)
def test_real_process_exit_preserves_event_without_replaying_outputs(
    tmp_path, checkpoint, state, was_sent
):
    path, output = tmp_path / "common.db", tmp_path / "submitted"
    result = subprocess.run(
        [sys.executable, "-c", CRASH_SCRIPT, str(path), checkpoint, str(output)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 42, result.stderr
    store = EventIngestStore(path)
    ingest = video_module().VideoEventIngest(
        store, backend="deepstream", topic_prefix="changed"
    )
    before = output.read_text() if output.exists() else None
    ingest.recover_pending()
    ingest.recover_pending()
    assert (
        len(rows(store, "ingest_receipts"))
        == len(rows(store, "canonical_records"))
        == len(rows(store, "ingest_revisions"))
        == 1
    )
    job = rows(store, "delivery_jobs")[0]
    assert job["state"] == state and job["destination"] == "mqtt:cctv/events"
    assert output.exists() is was_sent
    assert (output.read_text() if output.exists() else None) == before
    if was_sent:
        assert before == rows(store, "canonical_records")[0]["source_record_id"]
