"""같은 DB의 로컬 연결 경합과 제한 시간·프로세스 경계를 확인한다."""

import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from src import ingest_store as ingest_module
from src.core.video_event_ingest import VideoEventIngest
from src.event_ingest_store import EventIngestStore
from src.ingest_store import IngestStore


@pytest.mark.parametrize("alias", [False, True])
def test_video_receipt_waits_for_local_writer_without_using_sqlite_retry(
    tmp_path, monkeypatch, alias
):
    path = tmp_path / "source.db"
    writer = EventIngestStore(path)
    if alias:
        link = tmp_path / "alias.db"
        link.symlink_to(path)
        path = link
    video = VideoEventIngest(
        EventIngestStore(path), backend="test", topic_prefix="cctv/ai/events"
    )
    original_connect = sqlite3.connect
    started, finished = Event(), Event()

    def no_native_wait(*args, **kwargs):
        # SQLite 재시도 시간으로 우연히 통과하지 않도록 실제 연결 대기만 제거한다.
        kwargs["timeout"] = 0
        return original_connect(*args, **kwargs)

    def accept():
        started.set()
        try:
            return video.accept({"event_id": "one", "type": "fall"}, "cam1")
        finally:
            finished.set()

    monkeypatch.setattr(sqlite3, "connect", no_native_wait)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with writer._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            future = pool.submit(accept)
            assert started.wait(2)
            finished.wait(0.05)
        reference = future.result(timeout=2)
    assert video.store.get_event_receipt(reference.ingest_id)["raw"]["payload"]["event_id"] == "one"
    with writer._connect() as connection:
        assert connection.execute("SELECT count(*) FROM ingest_receipts").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM event_ingest_state").fetchone()[0] == 1


def test_idle_local_connection_does_not_reject_writes_to_available_database(tmp_path):
    first = IngestStore(tmp_path / "first.db")
    same = IngestStore(first.path)
    other = IngestStore(tmp_path / "other.db")
    with ThreadPoolExecutor(max_workers=2) as pool:
        # 로컬 연결이 오래 열려 있어도 실제 SQLite가 허용하는 저장을 거절하면 안 된다.
        with first._connect():
            pending = pool.submit(same.accept, "sensor", "one", {"value": 1})
            independent = pool.submit(other.accept, "sensor", "two", {"value": 2})
            saved = independent.result(timeout=2)
            same_saved = pending.result(timeout=2)
    assert same.pending(namespace="sensor") == [same_saved]
    assert other.get(saved)["raw"] == {"value": 2}
    assert same.get(same_saved)["raw"] == {"value": 1}


def _read_in_child(store, receipt, result):
    try:
        result.send(store.get(receipt)["raw"])
    except Exception as error:
        result.send(type(error).__name__)
    finally:
        result.close()


@pytest.mark.parametrize("registry_held", [False, True])
def test_fork_child_does_not_inherit_another_threads_admission_lock(tmp_path, registry_held):
    store = IngestStore(tmp_path / "source.db")
    receipt = store.accept("sensor", "one", {"value": 3})
    entered, release = Event(), Event()

    def hold():
        # fork 순간 registry를 갱신하던 부모 스레드도 자식의 새 연결을 막지 않아야 한다.
        context = ingest_module._connection_locks_guard if registry_held else store._connect()
        with context:
            entered.set()
            assert release.wait(5)

    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_read_in_child, args=(store, receipt, send))
    with ThreadPoolExecutor(max_workers=1) as pool:
        holder = pool.submit(hold)
        try:
            assert entered.wait(2)
            child.start()
            send.close()
            assert receive.poll(3)
            assert receive.recv() == {"value": 3}
            child.join(timeout=2)
            assert child.exitcode == 0
        finally:
            release.set()
            holder.result(timeout=2)
            if child.pid is not None and child.is_alive():
                child.terminate()
                child.join(timeout=2)
            send.close()
            receive.close()
