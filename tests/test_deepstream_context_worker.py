from queue import Queue
from unittest.mock import MagicMock

import pytest

from src.core._deepstream_context_worker import DeepStreamContextWorker, FaceContextTask
from src.core.events import DetectionEvent, EventType


def _person_event() -> DetectionEvent:
    return DetectionEvent(
        EventType.PERSON,
        1,
        2,
        30,
        40,
        0.9,
        1.0,
        object_id=7,
    )


def _submit_matched(worker, camera, events):
    from types import SimpleNamespace

    import numpy as np

    from src.core._preview_frame_store import read_frame_identity

    native = SimpleNamespace(misc_frame_info=[0, 0, 0, 0], source_id=0, frame_num=7, buf_pts=23)
    worker.stamp_frame(native, camera)
    identity = read_frame_identity(native, camera)
    for event in events:
        event.metadata = {**(event.metadata or {}), "frame_identity": identity}
    worker.submit(camera, events)
    assert worker.accept_frame(camera, np.zeros((80, 80, 3), dtype=np.uint8), identity)


def test_context_worker_submit_queues_person_task() -> None:
    queue = Queue(maxsize=2)
    person = _person_event()
    worker = DeepStreamContextWorker(
        queue=queue,
        feature_flags_for_camera=lambda camera_name: {
            "use_face": True,
            "use_appearance": False,
        },
        remember_context_events=MagicMock(),
        collect_context_events=lambda camera_name, events: list(events),
        run_face_recognition=MagicMock(return_value=[]),
        log_appearance_capability_hints=MagicMock(),
        refresh_appearance_conditions=MagicMock(),
        appearance_pipeline=MagicMock(),
        enqueue_event=MagicMock(return_value=True),
    )

    _submit_matched(worker, "cam1", [person])

    task = queue.get_nowait()
    assert task.camera_name == "cam1"
    assert task.person_events == [person]
    assert task.flags["use_face"] is True


def test_context_worker_process_adds_appearance_metadata() -> None:
    person = _person_event()
    appearance_event = DetectionEvent(
        EventType.PERSON,
        3,
        4,
        20,
        10,
        0.8,
        1.0,
        object_id=7,
    )
    appearance_pipeline = MagicMock()
    appearance_pipeline.run.return_value = [appearance_event]
    enqueue_event = MagicMock(return_value=True)
    worker = DeepStreamContextWorker(
        queue=Queue(),
        feature_flags_for_camera=MagicMock(),
        remember_context_events=MagicMock(),
        collect_context_events=MagicMock(),
        run_face_recognition=MagicMock(return_value=[]),
        log_appearance_capability_hints=MagicMock(),
        refresh_appearance_conditions=MagicMock(),
        appearance_pipeline=appearance_pipeline,
        enqueue_event=enqueue_event,
    )

    worker.process(
        FaceContextTask(
            camera_name="cam1",
            person_events=[person],
            frame=object(),
            flags={"use_face": False, "use_appearance": True},
            context_events=[person],
        )
    )

    assert appearance_event.metadata["backend"] == "deepstream_context"
    assert appearance_event.metadata["camera_id"] == "cam1"
    enqueue_event.assert_called_once_with(appearance_event, "cam1")


@pytest.mark.parametrize("phase", ["queued", "face", "appearance"])
@pytest.mark.parametrize("reset_camera", ["cam1", "cam2"])
def test_reset_blocks_old_context_side_effects_without_waiting_for_inference(
    tmp_path, phase, reset_camera,
):
    from threading import Event, Thread
    from types import SimpleNamespace

    from src.core._deepstream_face_context import run_deepstream_face_recognition
    from src.core.ai._appearance_pipeline import AppearancePipeline
    from src.core.deepstream_processor import DeepStreamProcessor

    entered, release, reset_done = Event(), Event(), Event()
    errors = []
    attrs = {"upper_color": "black", "lower_color": "blue"}

    def wait_for_reset(stage):
        if phase == stage:
            entered.set()
            assert release.wait(5), "inference release timeout"

    def recognize(frame, roi):
        wait_for_reset("face")
        return [SimpleNamespace(matched=True, label="test", confidence=.9,
            age=None, gender=None, bbox=dict(x=1, y=2, width=10, height=10))]

    def extract(*args, **kwargs):
        wait_for_reset("appearance")
        return dict(attrs)

    analyzer = SimpleNamespace(conditions=[], extract_attributes=extract)
    appearance = AppearancePipeline(analyzer, tmp_path / "crops")
    insert, snapshot, enqueue = MagicMock(), MagicMock(return_value=None), MagicMock()
    appearance._appearance_log = SimpleNamespace(insert=insert)
    cache = {}
    face = MagicMock(side_effect=lambda frame, persons, camera, **kwargs:
        run_deepstream_face_recognition(frame=frame, person_events=persons,
            camera_name=camera, recognizer=SimpleNamespace(enabled=True,
                detect_and_recognize=recognize), cache=cache,
            timestamp_factory=lambda: 1000., snapshot_saver=snapshot, **kwargs))
    worker = DeepStreamContextWorker(queue=Queue(),
        feature_flags_for_camera=lambda camera: {"use_face": True, "use_appearance": True},
        remember_context_events=MagicMock(), collect_context_events=lambda c, es: es,
        run_face_recognition=face, log_appearance_capability_hints=MagicMock(),
        refresh_appearance_conditions=MagicMock(), appearance_pipeline=appearance,
        enqueue_event=enqueue)
    proc = object.__new__(DeepStreamProcessor)
    proc._context_worker = worker
    proc.track_manager = proc.violation_filter = proc._debouncer = MagicMock()
    proc._synthetic_id_assigner = proc._context_event_store = MagicMock()
    proc._face_identity_cache = cache
    proc._falldata_aux = None
    _submit_matched(worker, "cam1", [_person_event()])
    task = worker.queue.get_nowait()

    def reset():
        try:
            proc._reset_camera_tracking_state(reset_camera)
        except Exception as exc:
            errors.append(exc)
        finally:
            reset_done.set()

    def process():
        try:
            worker.process(task)
        except Exception as exc:
            errors.append(exc)

    if phase == "queued":
        reset()
        process()
    else:
        thread = Thread(target=process)
        thread.start()
        reset_thread = None
        try:
            assert entered.wait(5), "inference did not start"
            reset_thread = Thread(target=reset)
            reset_thread.start()
            assert reset_done.wait(5), "reset waited for stalled inference"
        finally:
            release.set()
            thread.join(5)
            if reset_thread:
                reset_thread.join(5)
        assert not thread.is_alive()
    assert not errors
    if reset_camera == "cam1":
        enqueue.assert_not_called()
        insert.assert_not_called()
        assert not cache and not appearance._attribute_history
        assert not appearance._appearance_cooldown
        if phase != "appearance":
            snapshot.assert_not_called()
        # A face snapshot committed BEFORE appearance inference/reset is historical evidence.
        _submit_matched(worker, "cam1", [_person_event()])
        fresh = worker.queue.get_nowait()
        assert fresh.generation > task.generation
        release.set()
        worker.process(fresh)
        assert insert.call_count == 1 and enqueue.call_count == 1
        assert ("cam1", 7) in cache


    else:
        assert insert.call_count == 1 and enqueue.call_count == 1
        assert ("cam1", 7) in cache


@pytest.fixture
def matching_worker():
    return DeepStreamContextWorker(queue=Queue(maxsize=2),
        feature_flags_for_camera=lambda c: {"use_face": True},
        remember_context_events=MagicMock(), collect_context_events=lambda c, es: es,
        run_face_recognition=MagicMock(return_value=[]),
        log_appearance_capability_hints=MagicMock(), refresh_appearance_conditions=MagicMock(),
        appearance_pipeline=MagicMock(), enqueue_event=MagicMock(), record_stat=MagicMock())


def _pending_frame(worker, camera="cam1", frame_num=7, pts=23):
    from types import SimpleNamespace

    from src.core._preview_frame_store import read_frame_identity

    native = SimpleNamespace(misc_frame_info=[0, 0, 0, 0], source_id=0,
                             frame_num=frame_num, buf_pts=pts)
    worker.stamp_frame(native, camera)
    identity = read_frame_identity(native, camera)
    person = _person_event()
    person.metadata = {"frame_identity": identity}
    worker.submit(camera, [person])
    return identity, person


@pytest.mark.parametrize("field,value", [("camera_id", "cam2"), ("source_id", 1),
    ("frame_num", 8), ("source_pts_ns", 24), ("generation", 1), ("frame_token", 99)])
def test_mismatched_preview_never_uses_latest_frame(matching_worker, field, value):
    worker = matching_worker
    identity, person = _pending_frame(worker)
    assert worker.queue.empty()
    assert not worker.accept_frame("cam1", object(), {**identity, field: value})
    assert worker.queue.empty()
    frame = object()
    assert worker.accept_frame("cam1", frame, {**identity, "buffer_pts_ns": 999})
    task = worker.queue.get_nowait()
    assert task.frame is frame and task.person_events == [person]
    assert task.frame_identity == identity
    assert not worker.accept_frame("cam1", frame, identity)


def test_late_preview_cannot_match_reused_frame_and_pts_after_reset(matching_worker):
    worker = matching_worker
    old, _ = _pending_frame(worker)
    other, _ = _pending_frame(worker, "cam2")
    with worker.reset_camera("cam1"):
        pass
    fresh, _ = _pending_frame(worker)
    assert old["frame_num"] == fresh["frame_num"] and old["source_pts_ns"] == fresh["source_pts_ns"]
    assert old["frame_token"] != fresh["frame_token"]
    assert not worker.accept_frame("cam1", object(), old)
    # Even mislabelled generation cannot let an old native token match the new frame.
    assert not worker.accept_frame("cam1", object(), {**old, "generation": fresh["generation"]})
    assert worker.accept_frame("cam2", object(), other)
    assert worker.accept_frame("cam1", object(), fresh)


def test_pending_is_bounded_and_expires_without_waiting(matching_worker, monkeypatch):
    worker = matching_worker
    monkeypatch.setattr("src.core._deepstream_context_worker.time.monotonic", lambda: 10.)
    identities = [_pending_frame(worker, frame_num=i)[0] for i in range(40)]
    assert len(worker._pending["cam1"]) == 32
    assert not worker.accept_frame("cam1", object(), identities[0])
    monkeypatch.setattr("src.core._deepstream_context_worker.time.monotonic", lambda: 12.1)
    assert not worker.accept_frame("cam1", object(), identities[-1])
    assert not worker._pending["cam1"] and worker.queue.empty()


def test_full_worker_queue_drops_match_without_blocking(matching_worker):
    worker = matching_worker
    for i in range(3):
        identity, _ = _pending_frame(worker, frame_num=i)
        assert worker.accept_frame("cam1", object(), identity) is (i < 2)
    assert worker.queue.qsize() == 2 and not worker._pending["cam1"]
    worker.record_stat.assert_any_call("context_queue_dropped")


def _sample_people(worker, camera, object_ids):
    from dataclasses import replace

    identity, person = _pending_frame(worker, camera)
    # 같은 프레임의 인물 목록만 바꾸어 매칭/샘플링을 함께 통과시킨다.
    pending = worker._pending[camera][worker._identity_key(camera, identity)][1]
    pending.person_events[:] = [replace(person, object_id=value) for value in object_ids]
    return worker.accept_frame(camera, object(), identity)


def test_sampling_keeps_new_people_and_retries_existing_people_at_boundary(matching_worker, monkeypatch):
    worker = matching_worker
    worker.min_interval_sec = 0.5
    now = [10.0]
    monkeypatch.setattr("src.core._deepstream_context_worker.time.monotonic", lambda: now[0])
    assert _sample_people(worker, "cam1", [7])
    worker.process(worker.queue.get_nowait())
    now[0] = 10.1
    assert _sample_people(worker, "cam1", [7, 8])
    task = worker.queue.get_nowait()
    assert [p.object_id for p in task.person_events] == [8]
    worker.process(task)
    now[0] = 10.49
    assert not _sample_people(worker, "cam1", [7, 8])
    now[0] = 10.5
    assert _sample_people(worker, "cam1", [7, 8])
    assert [p.object_id for p in worker.queue.get_nowait().person_events] == [7]
    worker.record_stat.assert_any_call("context_frames_sampled")
    assert not any(call.args == ("context_queue_dropped",) for call in worker.record_stat.call_args_list)


def test_sampling_isolates_cameras_and_resets_with_input_generation(matching_worker, monkeypatch):
    worker = matching_worker
    worker.min_interval_sec = 0.5
    monkeypatch.setattr("src.core._deepstream_context_worker.time.monotonic", lambda: 10.0)
    assert _sample_people(worker, "cam1", [7])
    assert _sample_people(worker, "cam2", [7])
    worker.queue.get_nowait()
    worker.queue.get_nowait()
    with worker.reset_camera("cam1"):
        pass
    assert _sample_people(worker, "cam1", [7])
    assert not _sample_people(worker, "cam2", [7])


def test_full_queue_does_not_delay_retry_of_new_person(matching_worker, monkeypatch):
    worker = matching_worker
    worker.min_interval_sec = 0.5
    monkeypatch.setattr("src.core._deepstream_context_worker.time.monotonic", lambda: 10.0)
    assert _sample_people(worker, "cam1", [7])
    assert _sample_people(worker, "cam1", [8])
    assert not _sample_people(worker, "cam1", [9])
    worker.queue.get_nowait()
    assert _sample_people(worker, "cam1", [9])
    worker.record_stat.assert_any_call("context_queue_dropped")


@pytest.mark.parametrize("object_id", [None, -1, 2**64 - 1])
def test_untracked_people_are_not_mistaken_for_one_cached_person(matching_worker, object_id):
    matching_worker.min_interval_sec = 0.5
    assert _sample_people(matching_worker, "cam1", [object_id])
    assert _sample_people(matching_worker, "cam1", [object_id])


def test_sampling_metrics_are_distinct_from_queue_loss(matching_worker, monkeypatch):
    from collections import Counter
    from types import SimpleNamespace

    from prometheus_client import CollectorRegistry

    from src.services.processor_metrics import ProcessorStatsCollector

    worker = matching_worker
    worker.min_interval_sec = 0.5
    stats = Counter(context_queue_dropped=0)
    worker.record_stat = lambda name: stats.update([name])
    monkeypatch.setattr("src.core._deepstream_context_worker.time.monotonic", lambda: 10.0)
    assert _sample_people(worker, "cam1", [7])
    worker.process(worker.queue.get_nowait())
    assert not _sample_people(worker, "cam1", [7])
    registry = CollectorRegistry()
    registry.register(ProcessorStatsCollector(SimpleNamespace(get_stats=lambda: dict(stats))))
    assert registry.get_sample_value("cctv_processor_context_frames_sampled_total") == 1
    assert registry.get_sample_value("cctv_processor_context_people_sampled_total") == 1
    assert registry.get_sample_value("cctv_processor_context_queue_dropped_total") == 0


def test_busy_person_does_not_accumulate_old_frames_but_new_person_is_admitted(matching_worker, monkeypatch):
    worker = matching_worker
    worker.min_interval_sec = 0.5
    now = [10.0]
    monkeypatch.setattr("src.core._deepstream_context_worker.time.monotonic", lambda: now[0])
    assert _sample_people(worker, "cam1", [7])
    now[0] = 11.0
    assert not _sample_people(worker, "cam1", [7])
    assert _sample_people(worker, "cam1", [7, 8])
    first = worker.queue.get_nowait()
    assert [p.object_id for p in worker.queue.get_nowait().person_events] == [8]
    worker.process(first)
    assert _sample_people(worker, "cam1", [7])
    worker.record_stat.assert_any_call("context_people_busy")


def test_failed_analysis_releases_person_reservation(matching_worker, monkeypatch):
    worker = matching_worker
    worker.min_interval_sec = 0.5
    now = [10.0]
    monkeypatch.setattr("src.core._deepstream_context_worker.time.monotonic", lambda: now[0])
    assert _sample_people(worker, "cam1", [7])
    worker.run_face_recognition.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        worker.process(worker.queue.get_nowait())
    now[0] = 11.0
    assert _sample_people(worker, "cam1", [7])


def test_old_generation_completion_preserves_new_person_reservation(matching_worker, monkeypatch):
    worker = matching_worker
    worker.min_interval_sec = 0.5
    now = [10.0]
    monkeypatch.setattr("src.core._deepstream_context_worker.time.monotonic", lambda: now[0])
    assert _sample_people(worker, "cam1", [7])
    old = worker.queue.get_nowait()
    with worker.reset_camera("cam1"):
        pass
    assert _sample_people(worker, "cam1", [7])
    worker.process(old)
    now[0] = 11.0
    assert not _sample_people(worker, "cam1", [7])
    worker.record_stat.assert_any_call("context_people_busy")


def test_context_queue_overflow_is_visible_when_primary_drops_are_zero(matching_worker):
    from collections import Counter
    from types import SimpleNamespace

    from prometheus_client import CollectorRegistry, generate_latest

    from src.services.processor_metrics import ProcessorStatsCollector

    # 실제 매칭 큐를 포화시켜 전체 프레임 드롭과 후처리 누락이 별개로 노출되는지 확인한다.
    stats = Counter(frames_dropped=0)
    matching_worker.record_stat = lambda name: stats.update([name])
    for index in range(3):
        identity, _ = _pending_frame(matching_worker, frame_num=index)
        assert matching_worker.accept_frame("cam1", object(), identity) is (index < 2)

    registry = CollectorRegistry()
    registry.register(ProcessorStatsCollector(SimpleNamespace(get_stats=lambda: dict(stats))))
    payload = generate_latest(registry).decode()
    assert registry.get_sample_value("cctv_processor_frames_dropped_total") == 0
    assert registry.get_sample_value("cctv_processor_context_frames_matched_total") == 2
    assert registry.get_sample_value("cctv_processor_context_queue_dropped_total") == 1
    assert "# TYPE cctv_processor_context_queue_dropped_total counter" in payload
    assert "cam1" not in payload


def test_unavailable_context_metrics_are_not_reported_as_zero():
    from types import SimpleNamespace

    from prometheus_client import CollectorRegistry, generate_latest

    from src.services.processor_metrics import ProcessorStatsCollector

    # 계측이 없는 백엔드를 누락 0건으로 보이게 하지 않는다.
    registry = CollectorRegistry()
    registry.register(ProcessorStatsCollector(SimpleNamespace(get_stats=lambda: {})))
    assert "cctv_processor_context_" not in generate_latest(registry).decode()


def test_missing_pts_and_mixed_event_identities_are_rejected(matching_worker):
    worker = matching_worker
    identity, person = _pending_frame(worker, pts=2**64-1)
    assert identity is None and not worker._pending
    worker.remember_context_events.assert_not_called()
    first, person = _pending_frame(worker)
    second, other = _pending_frame(worker, frame_num=8)
    worker._pending.clear()
    worker.submit("cam1", [person, other])
    assert not worker._pending and worker.queue.empty()


def test_frame_tag_preserves_other_users_fields(matching_worker):
    from types import SimpleNamespace

    frame = SimpleNamespace(misc_frame_info=[123, 5, 8, 9])
    matching_worker.stamp_frame(frame, "cam1")
    assert frame.misc_frame_info == [123, 5, 8, 9]
    matching_worker.record_stat.assert_called_once_with("context_identity_rejected")


def test_duplicate_metadata_and_stale_context_do_not_enqueue_twice(matching_worker):
    worker = matching_worker
    identity, person = _pending_frame(worker)
    worker.submit("cam1", [person])
    assert len(worker._pending["cam1"]) == 1
    assert worker.accept_frame("cam1", object(), identity)
    with worker.reset_camera("cam1"):
        pass
    worker.remember_context_events.reset_mock()
    person.event_type = EventType.HELMET
    worker.submit("cam1", [person])
    worker.remember_context_events.assert_not_called()
    assert not worker._pending
