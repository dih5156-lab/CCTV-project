"""RF 비교 기록이 원 경보를 지연하거나 중복 발행하지 않는지 검증한다."""

import json
import threading
from dataclasses import replace
from queue import Queue
from types import SimpleNamespace

import pytest

from src.core import event_debouncer
from src.core.ai._falldata_aux import FallDataAuxConfig, FallDataAuxVerifier
from src.core.deepstream_processor import DeepStreamProcessor
from src.core.event_debouncer import EventDebouncer
from src.core.events import DetectionEvent, EventType


@pytest.fixture
def shadow_processor(tmp_path, monkeypatch):
    # 실제 디바운서를 사용하되 실행 부하로 시간 경계가 변하지 않게 고정한다.
    monkeypatch.setattr(event_debouncer, "time", SimpleNamespace(time=lambda: 1000.0))
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._falldata_aux_queue = Queue(maxsize=2)
    proc.stop_event = threading.Event()
    proc._events_detected = proc._events_dropped = 0
    proc._fall_aux_confirm_borderline = True
    proc._fall_aux_confirm_max_fall_score = 4.5
    proc._fall_detector = SimpleNamespace(score_threshold=3.0)
    proc._falldata_aux = FallDataAuxVerifier(FallDataAuxConfig(
        enabled=True, mode="shadow", inline_pose_rf=True,
        fail_open_on_unavailable=True,
    ))
    proc._fall_shadow_review_log_path = tmp_path / "review.jsonl"
    proc._fall_shadow_clip_dir = tmp_path / "clips"
    proc._fall_shadow_save_clips = False
    config = SimpleNamespace(events=SimpleNamespace(
        debounce_enabled=True, fall_sustained_seconds=0.0,
        fall_gap_reset_seconds=10.0, fall_resend_cooldown=60.0,
        debounce_seconds=3.0,
    ))
    proc._debouncer = EventDebouncer(config, proc._increment_stat)
    yield proc
    proc._falldata_aux.close()


def scoped_fall(proc):
    person = DetectionEvent(
        EventType.PERSON, 10, 20, 80, 160, 0.9, 1.0, object_id=7,
        keypoints=[[float(10 + i), float(20 + i), 0.9] for i in range(17)],
        metadata={"frame_num": 12, "frame_width": 640, "frame_height": 480,
                  "fall_score": 4.0, "camera_id": "cam1"},
    )
    fall = replace(person, event_type=EventType.FALL_DETECTED,
                   metadata=dict(person.metadata))
    proc._falldata_aux.add_pose_events("cam1", [person, fall])
    return fall


def test_shadow_rf_queues_scoped_comparison_after_original_alert(shadow_processor):
    proc = shadow_processor
    fall = scoped_fall(proc)
    assert proc._enqueue_event_or_defer_fall_aux(fall, "cam1") is True
    # 워커를 시작하기 전 원 경보가 전달되고, 별도 비교 작업에만 표식을 붙인다.
    assert proc.event_queue.get_nowait() is fall
    assert proc._falldata_aux_queue.qsize() == 1
    camera, payload = proc._falldata_aux_queue.get_nowait()
    assert camera == "cam1"
    assert payload["metadata"]["falldata_aux_shadow_only"] is True
    assert payload["metadata"]["falldata_pose_scope"] == {
        "camera_id": "cam1", "object_id": 7, "generation": 1, "frame_index": 12,
    }
    assert "falldata_aux_shadow_only" not in fall.metadata
    assert "falldata_aux_publish_pending" not in fall.metadata


def test_shadow_rf_debounced_fall_does_not_repeat_comparison(shadow_processor):
    proc = shadow_processor
    fall = scoped_fall(proc)
    assert proc._enqueue_event_or_defer_fall_aux(fall, "cam1") is True
    assert proc._enqueue_event_or_defer_fall_aux(fall, "cam1") is False
    assert proc.event_queue.qsize() == 1
    assert proc._falldata_aux_queue.qsize() == 1


def test_shadow_rf_full_aux_queue_keeps_original_alert(shadow_processor):
    proc = shadow_processor
    for _ in range(2):
        proc._falldata_aux_queue.put_nowait(("occupied", {}))
    fall = scoped_fall(proc)
    assert proc._enqueue_event_or_defer_fall_aux(fall, "cam1") is True
    assert proc.event_queue.get_nowait() is fall
    assert proc._falldata_aux_queue.qsize() == 2


def test_shadow_rf_full_event_queue_does_not_submit_comparison(shadow_processor):
    proc = shadow_processor
    for _ in range(2):
        proc.event_queue.put_nowait({"occupied": True})
    assert proc._enqueue_event_or_defer_fall_aux(scoped_fall(proc), "cam1") is False
    assert proc._falldata_aux_queue.empty()


@pytest.mark.parametrize("event_queue_full", [False, True])
def test_confirm_aux_full_queue_attempts_fail_open_once(shadow_processor, event_queue_full):
    proc = shadow_processor
    proc._falldata_aux.config = replace(proc._falldata_aux.config, mode="confirm")
    for _ in range(2):
        proc._falldata_aux_queue.put_nowait(("occupied", {}))
        if event_queue_full:
            proc.event_queue.put_nowait({"occupied": True})
    fall = scoped_fall(proc)

    # 보조 큐 제출 전 이미 허용된 경보는 중복 억제에 다시 막히면 안 된다.
    assert proc._enqueue_event_or_defer_fall_aux(fall, "cam1") is (not event_queue_full)
    assert proc._events_detected == (0 if event_queue_full else 1)
    assert proc._events_dropped == (1 if event_queue_full else 0)
    assert proc._falldata_aux_queue.qsize() == 2
    assert fall.metadata["falldata_aux_confirm_fallback"] == "submit_failed"
    assert "falldata_aux_publish_pending" not in fall.metadata
    if event_queue_full:
        assert not proc._fall_shadow_review_log_path.exists()
    else:
        assert proc.event_queue.get_nowait() is fall
        records = [json.loads(row) for row in proc._fall_shadow_review_log_path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["falldata_aux_publish_pending"] is False

    # 최초 경보만 전달하며, 같은 낙상 재입력의 cooldown은 유지한다.
    assert proc._enqueue_event_or_defer_fall_aux(fall, "cam1") is False
    assert proc._events_detected == (0 if event_queue_full else 1)
    assert proc._events_dropped == (1 if event_queue_full else 0)
    assert proc.event_queue.qsize() == (2 if event_queue_full else 0)


@pytest.mark.parametrize("enabled,kind", [
    (False, EventType.FALL_DETECTED), (True, EventType.PERSON),
])
def test_shadow_rf_ignores_disabled_or_nonfall_events(shadow_processor, enabled, kind):
    proc = shadow_processor
    fall = scoped_fall(proc)
    proc._falldata_aux.config = replace(proc._falldata_aux.config, enabled=enabled)
    fall.event_type = kind
    assert proc._enqueue_event_or_defer_fall_aux(fall, "cam1") is True
    assert proc._falldata_aux_queue.empty()


@pytest.mark.parametrize("status,confirmed", [
    ("ok", True), ("ok", False), ("incompatible_feature_contract", False),
    ("error", False), ("stale_pose_scope", False),
])
def test_shadow_rf_worker_records_without_republishing(
    shadow_processor, monkeypatch, status, confirmed,
):
    proc = shadow_processor
    fall = scoped_fall(proc)
    payload = fall.to_dict()
    payload["metadata"] = {**payload["metadata"], "falldata_aux_shadow_only": True}
    proc._falldata_aux_queue.put_nowait(("cam1", payload))
    # 큐에 들어온 뒤 설정이 바뀌어도 기존 비교 작업이 발행용으로 바뀌면 안 된다.
    proc._falldata_aux.config = replace(proc._falldata_aux.config, mode="confirm")
    scope = fall.metadata["falldata_pose_scope"]

    def verify(camera_name, *, pose_scope):
        assert camera_name == "cam1" and pose_scope == scope
        proc.stop_event.set()
        return {"enabled": True, "mode": "confirm", "status": status,
                "confirmed": confirmed, "runtime": "deepstream_pose_inline",
                "pose_scope": dict(scope), "fall_probability": 0.9 if confirmed else None}

    # 모델 실행만 통제하고 실제 워커·리뷰 파일·최종 발행 큐는 그대로 검증한다.
    monkeypatch.setattr(proc._falldata_aux, "verify", verify)
    proc._falldata_aux_worker_loop()
    rows = [json.loads(line) for line in proc._fall_shadow_review_log_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["falldata_aux"]["status"] == status
    assert proc.event_queue.empty()
    assert proc._events_detected == 0
