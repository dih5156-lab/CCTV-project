"""Bounded first-intake retries preserve bytes and never imply durable success."""

import sqlite3
from contextlib import closing, contextmanager
from queue import Queue
from unittest.mock import Mock

import pytest

from src.core._deepstream_event_queue import enqueue_queue_item
from src.core.event_dispatcher import EventDispatcher
from tests.test_monitoring_capacity import FULL_ERROR, LARGE_VALUE, capped_store
from tests.test_video_event_ingest import detection, enqueue, ingestor, rows


@pytest.mark.parametrize("failure_phase", ["insert", "commit"])
def test_one_busy_retry_freezes_original_and_commits_one_receipt(
    tmp_path, monkeypatch, failure_phase
):
    video = ingestor(tmp_path)
    observer = Mock()
    video.store.commit_observer = observer
    event, queue = detection(), Queue()
    original = video.store._connect
    attempts, inserts = [], []
    with closing(sqlite3.connect(video.store.path)) as blocker:
        if failure_phase == "insert":
            blocker.execute("BEGIN IMMEDIATE")
        else:
            blocker.execute("BEGIN")
            blocker.execute("SELECT * FROM ingest_receipts").fetchall()

        @contextmanager
        def connected():
            attempts.append(1)
            try:
                with original() as connection:
                    connection.set_trace_callback(
                        lambda sql: (
                            inserts.append(sql)
                            if sql.startswith("INSERT INTO ingest_receipts")
                            else None
                        )
                    )
                    yield connection
            except sqlite3.OperationalError as error:
                assert str(error) == "database is locked"
                blocker.rollback()
                event.metadata["frame_num"] = 999
                raise

        with monkeypatch.context() as patch:
            patch.setattr(video.store, "_connect", connected)
            assert enqueue(video, queue, event)
    assert len(attempts) == 2
    assert len(inserts) == 2 and inserts[0] == inserts[1]
    assert queue.qsize() == 1
    receipt = video.store.get_event_receipt(queue.get_nowait().ingest_id)
    assert receipt["raw"]["payload"]["metadata"]["frame_num"] == 7
    assert len(rows(video.store, "ingest_receipts")) == 1
    assert len(rows(video.store, "event_ingest_state")) == 1
    observer.assert_called_once_with(
        "ingest_commit_completed_at",
        receipt["ingest_id"],
        received_at=receipt["received_at"],
    )


@pytest.mark.parametrize("backend", ["deepstream", "opencv"])
@pytest.mark.parametrize("failure_phase", ["insert", "commit"])
def test_exhausted_retry_reports_failure_once_without_receipt_or_queue(
    tmp_path, monkeypatch, backend, failure_phase
):
    video, queue, stats = ingestor(tmp_path), Queue(), []
    backup, observer = Mock(), Mock()
    video.store.commit_observer = observer
    if backend == "opencv":
        dispatcher = EventDispatcher(queue, backup, stats.append, video_ingest=video)

        def submit():
            return dispatcher.enqueue("cam1", detection())

    else:

        def submit():
            return enqueue_queue_item(
                event_queue=queue,
                queue_item=detection(),
                camera_name="cam1",
                increment_stat=stats.append,
                video_ingest=video,
            )

    original, attempts = video.store._connect, []

    @contextmanager
    def connected():
        attempts.append(1)
        with original() as connection:
            yield connection

    with closing(sqlite3.connect(video.store.path)) as blocker:
        blocker.execute("BEGIN IMMEDIATE" if failure_phase == "insert" else "BEGIN")
        blocker.execute("SELECT * FROM ingest_receipts").fetchall()
        with monkeypatch.context() as patch:
            patch.setattr(video.store, "_connect", connected)
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                submit()
        blocker.rollback()
    assert len(attempts) == 2
    assert queue.empty() and stats == ["events_failed"]
    observer.assert_not_called()
    backup.save_locally.assert_not_called()
    assert not video._live_receipts
    video.recover_pending()
    assert rows(video.store, "ingest_receipts") == []
    assert rows(video.store, "event_ingest_state") == []
    assert submit()
    assert stats == ["events_failed", "events_detected"]
    assert len(rows(video.store, "ingest_receipts")) == queue.qsize() == 1


@pytest.mark.parametrize(
    "error_code,message,expected_attempts",
    [
        (None, "database is locked", 2),  # Python 3.10 sqlite exceptions.
        (517, "extended busy", 2),  # SQLITE_BUSY_SNAPSHOT on newer Python.
        (None, "database table is locked", 1),
        (None, "disk I/O error", 1),
        (13, "database is locked", 1),  # Code takes priority over text.
    ],
)
def test_only_busy_is_retried(
    tmp_path, monkeypatch, error_code, message, expected_attempts
):
    video = ingestor(tmp_path)
    error = sqlite3.OperationalError(message)
    if error_code is not None:
        error.sqlite_errorcode = error_code
    connect = Mock(side_effect=error)
    monkeypatch.setattr(video.store, "_connect", connect)
    with pytest.raises(sqlite3.OperationalError) as raised:
        video.accept(detection(), "cam1")
    assert raised.value is error
    assert connect.call_count == expected_attempts


def test_generic_event_intake_keeps_single_attempt_default(tmp_path, monkeypatch):
    store = ingestor(tmp_path).store
    connect = Mock(side_effect=sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(store, "_connect", connect)
    with pytest.raises(sqlite3.OperationalError):
        store.accept_event_json("external_mqtt", {"value": 1}, destination=None)
    assert connect.call_count == 1


def test_video_full_is_not_retried_or_reported_as_committed(
    tmp_path, monkeypatch, record_property
):
    video = ingestor(tmp_path)
    event = detection()
    event.metadata["large_value"] = LARGE_VALUE
    observer = Mock()
    video.store.commit_observer = observer
    with capped_store(video.store, monkeypatch, record_property):
        connect = Mock(wraps=video.store._connect)
        with monkeypatch.context() as patch:
            patch.setattr(video.store, "_connect", connect)
            with pytest.raises(sqlite3.OperationalError, match=FULL_ERROR):
                video.accept(event, "cam1")
        assert connect.call_count == 1
    observer.assert_not_called()
    assert rows(video.store, "ingest_receipts") == []
    assert rows(video.store, "event_ingest_state") == []
