"""CPU 영상마다 무대상·만료·퇴거·미확인 누락을 독립적으로 검증한다."""

from collections import Counter
from queue import Queue
from types import SimpleNamespace

import pytest

from src.core import _deepstream_context_worker as worker_module
from src.core._deepstream_context_worker import DeepStreamContextWorker
from src.core._preview_frame_store import read_frame_identity
from src.core.events import DetectionEvent, EventType


@pytest.fixture
def matching_context(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    # 스레드 대기나 실시간 경과 없이 만료 경계만 제어한다.
    monkeypatch.setattr(worker_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    counters = Counter()
    worker = DeepStreamContextWorker(
        queue=Queue(maxsize=8),
        feature_flags_for_camera=lambda camera: {"use_face": True},
        remember_context_events=lambda camera, events: None,
        collect_context_events=lambda camera, events: list(events),
        run_face_recognition=lambda *args, **kwargs: [],
        log_appearance_capability_hints=lambda *args: None,
        refresh_appearance_conditions=lambda: None,
        appearance_pipeline=SimpleNamespace(clear_camera=lambda camera: None),
        enqueue_event=lambda *args: True,
        record_stat=lambda name: counters.update([name]),
    )
    return worker, counters, clock


def _frame(worker, camera="cam1", number=1):
    native = SimpleNamespace(
        misc_frame_info=[0, 0, 0, 0], source_id=0,
        frame_num=number, buf_pts=number * 33_333_333,
    )
    worker.stamp_frame(native, camera)
    return native, read_frame_identity(native, camera)


def _event(identity, event_type=EventType.PERSON):
    return DetectionEvent(
        event_type, 1, 2, 30, 40, 0.9, 1.0, object_id=7,
        metadata={"frame_identity": identity},
    )


def _assert_misses(counters, **expected):
    actual = {
        reason: counters[f"context_match_missed_{reason}"]
        for reason in ("no_target", "expired", "evicted", "unknown")
    }
    assert actual == {reason: expected.get(reason, 0) for reason in actual}
    assert counters["context_match_missed"] == sum(actual.values())


@pytest.mark.parametrize("non_person", [False, True])
def test_finished_frame_without_eligible_person_is_no_target(matching_context, non_person):
    worker, counters, _ = matching_context
    native, identity = _frame(worker)
    worker.submit("cam1", [_event(identity, EventType.HEAD)] if non_person else [])
    worker.finish_frame(native, "cam1")

    assert not worker.accept_frame("cam1", object(), identity)
    assert worker.queue.empty()
    _assert_misses(counters, no_target=1)


def test_preview_before_metadata_completion_remains_unknown(matching_context):
    worker, counters, _ = matching_context
    native, identity = _frame(worker)

    assert not worker.accept_frame("cam1", object(), identity)
    _assert_misses(counters, unknown=1)

    # 늦은 메타 완료로 앞서 관찰한 미확인 원인을 소급 변경하지 않는다.
    worker.finish_frame(native, "cam1")
    _assert_misses(counters, unknown=1)


@pytest.mark.parametrize("person_phase", ["tensor", "object", "both"])
def test_completed_person_frame_matches_after_both_metadata_phases(
    matching_context, person_phase,
):
    worker, counters, _ = matching_context
    native, identity = _frame(worker)
    person = _event(identity)
    for phase in ("tensor", "object"):
        worker.submit("cam1", [person] if person_phase in (phase, "both") else [])
    worker.finish_frame(native, "cam1")

    cpu_frame = object()
    assert worker.accept_frame("cam1", cpu_frame, identity)
    task = worker.queue.get_nowait()
    assert task.frame is cpu_frame
    assert task.person_events == [person]
    assert task.frame_identity == identity
    assert worker.queue.empty()
    _assert_misses(counters)


def test_late_cpu_for_person_frame_is_expired_instead_of_no_target(matching_context):
    worker, counters, clock = matching_context
    native, identity = _frame(worker)
    worker.submit("cam1", [_event(identity)])
    worker.finish_frame(native, "cam1")
    clock.now += worker.max_pending_age_sec + 0.01

    assert not worker.accept_frame("cam1", object(), identity)
    assert worker.queue.empty()
    _assert_misses(counters, expired=1)
    assert counters["context_match_expired"] == 1


def test_expiry_triggered_by_another_metadata_frame_preserves_cpu_reason(matching_context):
    worker, counters, clock = matching_context
    native, identity = _frame(worker)
    worker.submit("cam1", [_event(identity)])
    clock.now += worker.max_pending_age_sec + 0.01
    _, newer_identity = _frame(worker, number=2)
    worker.submit("cam1", [_event(newer_identity)])
    # 대상이 만료된 뒤 완료 통지가 와도 무대상으로 바뀌어서는 안 된다.
    worker.finish_frame(native, "cam1")

    assert not worker.accept_frame("cam1", object(), identity)
    _assert_misses(counters, expired=1)


def test_pending_capacity_eviction_is_reported_once_for_each_late_cpu(matching_context):
    worker, counters, _ = matching_context
    worker.max_pending_frames = 1
    native, first_identity = _frame(worker)
    worker.submit("cam1", [_event(first_identity)])
    _, second_identity = _frame(worker, number=2)
    worker.submit("cam1", [_event(second_identity)])
    worker.finish_frame(native, "cam1")

    assert not worker.accept_frame("cam1", object(), first_identity)
    _assert_misses(counters, evicted=1)
    assert counters["context_match_evicted"] == 1
    assert worker.accept_frame("cam1", object(), second_identity)
    _assert_misses(counters, evicted=1)


def test_duplicate_of_matched_cpu_is_unknown_and_never_enqueued_twice(matching_context):
    worker, counters, _ = matching_context
    native, identity = _frame(worker)
    worker.submit("cam1", [_event(identity)])
    worker.finish_frame(native, "cam1")

    assert worker.accept_frame("cam1", object(), identity)
    assert not worker.accept_frame("cam1", object(), identity)
    assert worker.queue.qsize() == 1
    _assert_misses(counters, unknown=1)


def test_empty_frame_history_is_bounded_and_forgotten_evidence_becomes_unknown(matching_context):
    worker, counters, _ = matching_context
    assert worker.max_match_history_frames == 256
    worker.max_match_history_frames = 2
    identities = []
    for number in range(3):
        native, identity = _frame(worker, number=number)
        worker.finish_frame(native, "cam1")
        identities.append(identity)

    assert not worker.accept_frame("cam1", object(), identities[0])
    _assert_misses(counters, unknown=1)
    assert not worker.accept_frame("cam1", object(), identities[-1])
    _assert_misses(counters, unknown=1, no_target=1)


def test_forgotten_eviction_reason_is_unknown_not_inferred_no_target(matching_context):
    worker, counters, _ = matching_context
    worker.max_pending_frames = 1
    worker.max_match_history_frames = 2
    _, old_identity = _frame(worker)
    worker.submit("cam1", [_event(old_identity)])
    for number in range(2, 6):
        native, identity = _frame(worker, number=number)
        worker.submit("cam1", [_event(identity)])
        worker.finish_frame(native, "cam1")

    assert not worker.accept_frame("cam1", object(), old_identity)
    _assert_misses(counters, unknown=1)


def test_camera_reset_rejects_old_generation_without_reusing_its_no_target_reason(matching_context):
    worker, counters, _ = matching_context
    native, old_identity = _frame(worker)
    worker.finish_frame(native, "cam1")
    with worker.reset_camera("cam1"):
        pass
    _, fresh_identity = _frame(worker)

    assert not worker.accept_frame("cam1", object(), old_identity)
    assert counters["context_identity_rejected"] == 1
    _assert_misses(counters)
    assert not worker.accept_frame("cam1", object(), fresh_identity)
    assert not worker.accept_frame(
        "cam1", object(), {**old_identity, "generation": fresh_identity["generation"]},
    )
    _assert_misses(counters, unknown=2)


def test_camera_evidence_and_reset_do_not_change_other_camera_reason(matching_context):
    worker, counters, _ = matching_context
    native, identity = _frame(worker, camera="cam1")
    worker.finish_frame(native, "cam1")
    other_native, other_identity = _frame(worker, camera="cam2")
    worker.finish_frame(other_native, "cam2")
    with worker.reset_camera("cam1"):
        pass

    assert not worker.accept_frame("cam2", object(), {**identity, "camera_id": "cam2"})
    _assert_misses(counters, unknown=1)
    assert not worker.accept_frame("cam2", object(), other_identity)
    _assert_misses(counters, unknown=1, no_target=1)


def test_sampling_rejection_is_not_counted_as_an_unmatched_cpu(matching_context):
    worker, counters, _ = matching_context
    worker.min_interval_sec = 0.5
    for number in range(2):
        native, identity = _frame(worker, number=number)
        worker.submit("cam1", [_event(identity)])
        worker.finish_frame(native, "cam1")
        assert worker.accept_frame("cam1", object(), identity) is (number == 0)

    assert worker.queue.qsize() == 1
    assert counters["context_frames_sampled"] == 1
    assert counters["context_people_busy"] == 1
    _assert_misses(counters)
