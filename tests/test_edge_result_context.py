"""최초 접수 표본과 활성화 경계가 서로 다른 SQLite 연결에서도 직렬화되는지 검사한다."""

import sqlite3
import threading
import uuid

import pytest

from src.services.edge_orchestrator import RegistryError
from src.services.edge_result_context import EdgeResultContext


@pytest.fixture
def context(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    store = EdgeResultContext(root / "context.sqlite3", create=True)
    identity = {
        "node_id": "jetson-b",
        "camera_id": "camera_1",
        "assignment_id": str(uuid.uuid4()),
        "epoch": 1,
        "supervisor_boot_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
    }
    store.begin(identity)
    try:
        yield store, identity
    finally:
        store.close()


def test_queued_snapshot_is_frozen_across_activation_and_reconciliation(context):
    control, identity = context
    producer = EdgeResultContext(control.path)
    session = identity["session_id"]
    try:
        queued = producer.capture(session)
        command = {
            "command_id": "activate-1",
            "activation_id": str(uuid.uuid4()),
            "expected_activation_id": None,
        }
        barrier = control.activate(session, command)
        assert barrier["sequence_floor"] == queued["edge"]["event_sequence"] == 1
        assert producer.snapshot(session) == barrier
        fresh = producer.capture(session)
        assert fresh["edge"]["event_sequence"] == 2
        assert fresh["edge"]["activation_id"] == command["activation_id"]
        assert queued["edge"]["activation_id"] is None
        assert control.activate(session, command) == barrier
        next_command = {
            "command_id": "reconcile-2",
            "activation_id": str(uuid.uuid4()),
            "expected_activation_id": command["activation_id"],
        }
        next_barrier = control.activate(session, next_command)
        assert next_barrier["sequence_floor"] == 2
        # 과거 명령 재시도의 ACK는 유지하지만 현재 토큰을 과거 값으로 되돌리지 않는다.
        assert control.activate(session, command) == barrier
        assert producer.snapshot(session) == next_barrier
        with pytest.raises(RegistryError, match="activation_changed"):
            control.activate(session, dict(command, command_id="late-old-start"))
    finally:
        producer.close()


def test_retired_session_cannot_capture_or_activate_and_sequence_survives_reconnect(
    context,
):
    store, identity = context
    session = identity["session_id"]
    first = store.capture(session)
    second = EdgeResultContext(store.path)
    try:
        assert second.capture(session)["edge"]["event_sequence"] == 2
        with pytest.raises(RegistryError, match="result_session_already_running"):
            store.begin(dict(identity, session_id=str(uuid.uuid4())))
        store.end(session)
        with pytest.raises(RegistryError, match="retired_result_session"):
            second.capture(session)
        store.begin(
            dict(
                identity,
                session_id=str(uuid.uuid4()),
                assignment_id=str(uuid.uuid4()),
                epoch=2,
            )
        )
        assert first["edge"]["session_id"] == session
        with pytest.raises(RegistryError, match="retired_result_session"):
            second.activate(
                session,
                {
                    "command_id": "late",
                    "activation_id": str(uuid.uuid4()),
                    "expected_activation_id": None,
                },
            )
    finally:
        second.close()


def test_parallel_producer_connections_have_unique_monotonic_sequences(context):
    store, identity = context
    producers = [EdgeResultContext(store.path), EdgeResultContext(store.path)]
    snapshots = []
    failures = []
    barrier = threading.Barrier(3)

    def capture(producer):
        try:
            barrier.wait()
            for _ in range(30):
                snapshots.append(producer.capture(identity["session_id"]))
        except Exception as error:
            failures.append(error)

    threads = [
        threading.Thread(target=capture, args=(producer,)) for producer in producers
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    for producer in producers:
        producer.close()
    assert not failures
    assert sorted(item["edge"]["event_sequence"] for item in snapshots) == list(
        range(1, 61)
    )
    assert len({item["event_id"] for item in snapshots}) == 60


def test_no_snapshot_returned_before_commit_when_storage_fails(context):
    store, identity = context
    store.db.execute("PRAGMA query_only=ON")
    with pytest.raises(sqlite3.OperationalError):
        store.capture(identity["session_id"])
    store.db.execute("PRAGMA query_only=OFF")
    assert store.capture(identity["session_id"])["edge"]["event_sequence"] == 1


@pytest.mark.parametrize("lifetime", [0, -1, float("nan"), float("inf"), 301, True])
def test_bad_lifetime_never_allocates_sequence(context, lifetime):
    store, identity = context
    with pytest.raises(ValueError):
        store.capture(identity["session_id"], lifetime=lifetime)
    assert store.capture(identity["session_id"])["edge"]["event_sequence"] == 1


def result_payload(store, identity, event_id=None):
    stamp = store.capture(identity["session_id"], event_id=event_id)
    return {
        "kind": "event",
        "edge": stamp["edge"],
        "event": {
            "event_id": stamp["event_id"],
            "camera_id": identity["camera_id"],
            "type": "person",
        },
    }


def test_node_queue_needs_matching_central_storage_receipt(context):
    store, identity = context
    payload = result_payload(store, identity)
    queued = store.enqueue(payload)
    assert store.enqueue(payload) == queued
    assert len(store.pending()) == 1
    with pytest.raises(RegistryError, match="result_identity_conflict"):
        store.enqueue(dict(payload, event=dict(payload["event"], type="fall")))
    with pytest.raises(RegistryError, match="central_storage_receipt_required"):
        store.confirm(dict(queued, stored=False))
    with pytest.raises(RegistryError, match="central_receipt_mismatch"):
        store.confirm(dict(queued, stored=True, sha256="0" * 64))
    assert len(store.pending()) == 1
    store.confirm(
        dict(queued, stored=True, decision="rejected", reason="retired_assignment")
    )
    assert store.pending() == []
    assert (
        store.db.execute("SELECT state,receipt FROM node_outbox").fetchone()[0]
        == "committed"
    )


def test_queue_recovery_keeps_original_retired_capture_and_body(context):
    store, identity = context
    payload = result_payload(store, identity, "saved-original")
    assert (
        store.capture(identity["session_id"], event_id="saved-original")["edge"]
        == payload["edge"]
    )
    store.end()
    queued = store.enqueue(payload)
    reloaded = EdgeResultContext(store.path)
    try:
        assert reloaded.pending() == [{**queued, "payload": payload}]
        assert reloaded.enqueue(payload) == queued
        forged = dict(
            payload, edge=dict(payload["edge"], activation_id=str(uuid.uuid4()))
        )
        with pytest.raises(RegistryError, match="capture_identity_conflict"):
            reloaded.enqueue(forged)
        reloaded.db.execute("PRAGMA query_only=ON")
        with pytest.raises(sqlite3.OperationalError):
            reloaded.confirm(dict(queued, stored=True))
        assert len(reloaded.pending()) == 1
    finally:
        reloaded.close()
