"""실제 영상 접수 경로의 담당 정보 고정·장애 재처리·직접 출력 차단 시험."""

import copy
import sqlite3
import threading
import uuid
from queue import Queue
from types import SimpleNamespace

import pytest

from src.core.events import DetectionEvent, EventType
from src.core.video_event_ingest import VideoEventIngest
from src.event_ingest_store import EventIngestStore
from src.services.edge_orchestrator import RegistryError
from src.services.edge_result_context import EdgeResultContext


@pytest.fixture
def edge_ingest(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    context = EdgeResultContext(private / "context.sqlite3", create=True)
    identity = {
        "node_id": "jetson-b",
        "camera_id": "cam1",
        "epoch": 1,
        "assignment_id": str(uuid.uuid4()),
        "supervisor_boot_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
    }
    context.begin(identity)
    ingest = VideoEventIngest(
        EventIngestStore(private / "intake.sqlite3"),
        backend="deepstream",
        topic_prefix="poc/unused",
        edge_context=context,
        session_id=identity["session_id"],
    )
    try:
        yield ingest, context, identity
    finally:
        context.close()


def detection():
    return DetectionEvent(
        EventType.PERSON,
        1,
        2,
        30,
        40,
        0.8,
        1700000000,
        metadata={"camera_id": "cam1", "frame_num": 7},
    )


def activate(context, identity):
    return context.activate(
        identity["session_id"],
        {
            "command_id": "activate",
            "activation_id": str(uuid.uuid4()),
            "expected_activation_id": None,
        },
    )


def test_actual_detection_intake_freezes_identity_before_activation(edge_ingest):
    ingest, context, identity = edge_ingest
    item = detection()
    reference = ingest.accept(item, "cam1")
    original = copy.deepcopy(item.metadata["edge_context"])
    barrier = activate(context, identity)
    assert barrier["sequence_floor"] == 1
    item.metadata["frame_num"] = 999
    assert ingest.publish(
        reference,
        mqtt_publish=None,
        event_publisher=SimpleNamespace(
            publish_event=lambda event: pytest.fail("legacy publisher used")
        ),
    )
    queued = context.pending()[0]["payload"]
    assert queued["event"]["event_id"] == original["event_id"]
    assert queued["edge"] == original["edge"]
    assert queued["edge"]["activation_id"] is None
    assert queued["event"]["raw"]["metadata"]["frame_num"] == 7
    assert queued["event"]["observation"]["source_record_id"] == original["event_id"]


def test_commit_gap_recovery_reuses_identical_canonical_body(edge_ingest, monkeypatch):
    ingest, context, _ = edge_ingest
    reference = ingest.accept(detection(), "cam1")
    original_record = ingest.store.record_event

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("simulated disk failure after result commit")

    monkeypatch.setattr(ingest.store, "record_event", fail)
    with pytest.raises(sqlite3.OperationalError):
        ingest.publish(reference, mqtt_publish=None, event_publisher=None)
    first = context.pending()
    assert len(first) == 1
    assert ingest.store.get_event_receipt(reference.ingest_id)["stage"] == "pending"
    monkeypatch.setattr(ingest.store, "record_event", original_record)
    context.end()
    with ingest.store._connect() as connection:
        connection.execute("UPDATE event_ingest_state SET next_attempt_at=0")
    ingest.recover_pending()
    assert context.pending() == first
    assert ingest.store.get_event_receipt(reference.ingest_id)["stage"] == "accepted"


def test_queue_storage_failure_keeps_original_pending(edge_ingest):
    ingest, context, _ = edge_ingest
    reference = ingest.accept(detection(), "cam1")
    context.db.execute("PRAGMA query_only=ON")
    with pytest.raises(sqlite3.OperationalError):
        ingest.publish(reference, mqtt_publish=None, event_publisher=None)
    assert ingest.store.get_event_receipt(reference.ingest_id)["stage"] == "pending"
    context.db.execute("PRAGMA query_only=OFF")
    with ingest.store._connect() as connection:
        connection.execute("UPDATE event_ingest_state SET next_attempt_at=0")
    ingest.recover_pending()
    assert len(context.pending()) == 1


def test_publish_worker_recovers_original_after_context_database_lock(edge_ingest):
    from src.core._event_publish import run_publish_loop

    ingest, context, identity = edge_ingest
    item = detection()
    reference = ingest.accept(item, "cam1")
    original = copy.deepcopy(item.metadata["edge_context"])
    queue = Queue()
    queue.put(reference)
    stop, failed = threading.Event(), threading.Event()
    worker = threading.Thread(
        target=run_publish_loop,
        kwargs={
            "is_running": lambda: True,
            "stop_event": stop,
            "event_queue": queue,
            "topic_prefix": "poc/unused",
            "mqtt_publish": None,
            "event_publisher": None,
            "increment_stat": lambda name: failed.set() if name == "events_failed" else None,
            "queue_timeout_sec": 0.02,
            "video_ingest": ingest,
        },
    )
    writer = sqlite3.connect(context.path)
    context.db.execute("PRAGMA busy_timeout=20")
    try:
        writer.execute("BEGIN EXCLUSIVE")
        worker.start()
        assert failed.wait(3)
        assert worker.is_alive()
        assert ingest.store.get_event_receipt(reference.ingest_id)["stage"] == "pending"
        writer.rollback()
        activate(context, identity)
        with ingest.store._connect() as connection:
            connection.execute("UPDATE event_ingest_state SET next_attempt_at=0")
        for _ in range(150):
            if context.pending():
                break
            stop.wait(0.02)
        queued = context.pending()
        assert len(queued) == 1
        assert queued[0]["event_id"] == original["event_id"]
        assert queued[0]["payload"]["edge"] == original["edge"]
        # 송신 큐 저장과 접수 완료 사이를 포함해 종료 시점까지 실제 복구 작업자를 기다린다.
        stop.set()
        worker.join(5)
        assert not worker.is_alive()
        assert ingest.store.get_event_receipt(reference.ingest_id)["stage"] == "accepted"
    finally:
        writer.rollback()
        writer.close()
        stop.set()
        if worker.ident is not None:
            worker.join(5)


def test_forged_or_wrong_camera_stamp_is_rejected(edge_ingest):
    ingest, context, _ = edge_ingest
    item = detection()
    ingest.stamp(item, "cam1")
    item.metadata["edge_context"]["edge"]["epoch"] = 99
    with pytest.raises(RegistryError, match="capture_identity_conflict"):
        ingest.accept(item, "cam1")
    item = detection()
    item.metadata["camera_id"] = "other-camera"
    with pytest.raises(ValueError, match="camera session"):
        ingest.accept(item, "cam1")
    assert context.pending() == []


def test_deferred_aux_keeps_pre_activation_stamp(edge_ingest):
    from src.core.deepstream_processor import DeepStreamProcessor

    ingest, context, identity = edge_ingest
    processor = object.__new__(DeepStreamProcessor)
    processor._video_ingest = ingest
    processor._should_confirm_fall_with_aux_before_publish = lambda event: True
    processor._should_enqueue_event = lambda event, camera: True
    processor._submit_falldata_aux_work = lambda camera, events: True
    processor.event_queue = Queue()
    processor._increment_stat = lambda key: None
    item = detection()
    assert not processor._enqueue_event_or_defer_fall_aux(item, "cam1")
    original = copy.deepcopy(item.metadata["edge_context"])
    activate(context, identity)
    fallback = item.to_dict()
    assert processor._put_event_dict(fallback, "cam1")
    assert ingest.publish(
        processor.event_queue.get_nowait(), mqtt_publish=None, event_publisher=None
    )
    assert context.pending()[0]["payload"]["edge"] == original["edge"]


def test_edge_profile_cannot_use_legacy_direct_publish(edge_ingest, monkeypatch):
    from src.core._event_publish import publish_queue_item
    from src.core.deepstream_processor import DeepStreamProcessor
    from src.core.processor import VideoProcessor
    from src.protocols.mqtt_publisher import MqttEventPublisher

    ingest, context, _ = edge_ingest
    reference = ingest.accept(detection(), "cam1")
    with pytest.raises(ValueError, match="bypass"):
        ingest.publish(
            reference,
            mqtt_publish=lambda *args: pytest.fail("bypass"),
            event_publisher=None,
        )
    monkeypatch.setenv("EDGE_RESULT_CONTEXT_DB", str(context.path))
    with pytest.raises(ValueError, match="durable"):
        publish_queue_item(
            {},
            topic_prefix="operating",
            mqtt_publish=lambda *args: None,
            event_publisher=None,
        )
    with pytest.raises(ValueError, match="bypass"):
        object.__new__(MqttEventPublisher).publish_event({})
    with pytest.raises(ValueError, match="bypass"):
        object.__new__(DeepStreamProcessor).set_mqtt_publish_callback(
            lambda *args: None
        )
    with pytest.raises(ValueError, match="DeepStream"):
        VideoProcessor(None)
    monkeypatch.delenv("AIOT_INGEST_DB", raising=False)
    with pytest.raises(ValueError, match="AIOT_INGEST_DB"):
        VideoEventIngest.from_env(backend="deepstream", topic_prefix="operating")
