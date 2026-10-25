"""낙상 영상시계와 실시간 재알림·보존 시계의 경계를 검증한다."""

from types import SimpleNamespace

import pytest

from src.core.event_debouncer import EventDebouncer
from tests.conftest import make_event


def _debouncer():
    return EventDebouncer(SimpleNamespace(events=SimpleNamespace(
        debounce_enabled=True, fall_sustained_seconds=2.0,
        fall_gap_reset_seconds=5.0, fall_resend_cooldown=60.0,
        event_retention_hours=1,
    )), lambda *_: None)


def _event(seconds, *, camera="cam1", object_id=1, generation=0):
    event = make_event("fall_detected", object_id=object_id, timestamp=1_000_000)
    event.metadata = {"frame_identity": {
        "camera_id": camera, "source_id": 0, "frame_num": int(seconds * 30),
        "source_pts_ns": round(seconds * 1_000_000_000),
        "generation": generation, "frame_token": int(seconds * 30) + 1,
    }}
    return event


def _send(debouncer, event, camera="cam1"):
    return debouncer.should_send(camera, "fall_detected", event.object_id, event=event)


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0, 4.0])
def test_processing_speed_does_not_change_fall_alarm_frame(speed, debounce_clock):
    debouncer = _debouncer()
    decisions = []
    # 0초 PTS도 유효하며, 추적 ID 변경과 3초 영상 공백은 기존 5초 기준 안이다.
    for seconds, object_id in [(0, 1), (1, 1), (2, 2), (5, 3), (6, 3)]:
        debounce_clock.now = 1_000_000 + seconds * speed
        event = _event(seconds, object_id=object_id)
        decisions.append(_send(debouncer, event))
        assert event.timestamp == 1_000_000
    assert decisions == [False, False, True, False, False]


def test_resend_uses_real_elapsed_time_and_rejects_frozen_frames(debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(0))
    debounce_clock.now += 1
    assert _send(debouncer, _event(2))
    # 영상 62초를 빨리 처리해도 실제 60초가 지나기 전에는 재알림하지 않는다.
    for seconds in range(4, 65, 2):
        assert not _send(debouncer, _event(seconds))
    debounce_clock.now += 60
    assert not _send(debouncer, _event(64))
    assert _send(debouncer, _event(66))


@pytest.mark.parametrize("changed_id", [False, True])
def test_cleanup_uses_receipt_time_instead_of_small_video_pts(changed_id, debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(0))
    debounce_clock.now += 1
    assert _send(debouncer, _event(2, object_id=2 if changed_id else 1))
    debouncer.cleanup()
    assert not _send(debouncer, _event(3, object_id=3))
    debounce_clock.now += 3601
    debouncer.cleanup()
    assert not _send(debouncer, _event(4, object_id=3))
    assert _send(debouncer, _event(6, object_id=3))


def test_regression_and_camera_reset_do_not_inherit_sustained_duration(debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(10))
    assert not _send(debouncer, _event(11))
    assert not _send(debouncer, _event(0))
    assert not _send(debouncer, _event(1))
    assert _send(debouncer, _event(2))
    assert not _send(debouncer, _event(0))
    assert not _send(debouncer, _event(2))  # PTS 역행으로 재알림 제한을 우회하지 않는다.
    assert not _send(debouncer, _event(0, camera="cam2"), "cam2")
    debouncer.clear_camera("cam1")
    assert not _send(debouncer, _event(3, generation=1))
    assert _send(debouncer, _event(2, camera="cam2"), "cam2")
    assert _send(debouncer, _event(5, generation=1))


def test_new_generation_resets_and_old_generation_cannot_reenter(debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(0))
    assert _send(debouncer, _event(2))
    assert not _send(debouncer, _event(100, generation=1))
    assert not _send(debouncer, _event(101, generation=1))
    assert not _send(debouncer, _event(102, generation=0))
    assert _send(debouncer, _event(102, generation=1))


@pytest.mark.parametrize("bad_pts", [None, -1, 2**64 - 1, float("nan"), float("inf"), "2000000000", True])
def test_invalid_pts_cannot_bridge_sustained_duration(bad_pts, debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(0))
    event = _event(1)
    event.metadata["frame_identity"]["source_pts_ns"] = bad_pts
    debounce_clock.now += 3
    assert not _send(debouncer, event)
    assert not _send(debouncer, _event(2))
    assert _send(debouncer, _event(4))


def test_missing_identity_falls_back_without_mixing_clock_origins(debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(0))
    event = _event(1)
    event.metadata = None
    debounce_clock.now += 3
    assert not _send(debouncer, event)
    debounce_clock.now += 2
    assert _send(debouncer, event)
    # 영상시계로 돌아와도 첫 관측부터 새로 누적하고 실시간 재알림은 유지한다.
    assert not _send(debouncer, _event(2))
    assert not _send(debouncer, _event(4))


def test_camera_identity_mismatch_cannot_advance_fall_timer(debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(0))
    debounce_clock.now += 3
    assert not _send(debouncer, _event(2, camera="cam2"))
    assert not _send(debouncer, _event(3))
    assert _send(debouncer, _event(5))


@pytest.mark.parametrize("identity", [None, {}, [], {"camera_id": "cam1", "generation": True, "source_pts_ns": 0}])
def test_invalid_identity_is_not_treated_as_legacy_missing_metadata(identity, debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(0))
    event = _event(2)
    event.metadata["frame_identity"] = identity
    debounce_clock.now += 3
    assert not _send(debouncer, event)
    assert not _send(debouncer, _event(3))
    assert _send(debouncer, _event(5))


def test_duplicate_with_changed_id_keeps_reassociation_and_cooldown(debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(0))
    assert _send(debouncer, _event(2))
    assert not _send(debouncer, _event(2, object_id=2))
    assert not _send(debouncer, _event(3, object_id=3))
    assert not _send(debouncer, _event(5, object_id=3))


def test_video_gap_resets_duration_and_same_pts_different_people_stay_independent(debounce_clock):
    debouncer = _debouncer()
    for seconds, expected in [(0, False), (6, False), (7, False), (8, True)]:
        for object_id, x in [(1, 10), (2, 1000)]:
            event = _event(seconds, object_id=object_id)
            event.x = x
            assert _send(debouncer, event) is expected


@pytest.mark.parametrize("metadata", ["bad", 1, []])
def test_malformed_metadata_cannot_enable_processing_clock_fallback(metadata, debounce_clock):
    debouncer = _debouncer()
    assert not _send(debouncer, _event(0))
    event = _event(2)
    event.metadata = metadata
    debounce_clock.now += 3
    assert not _send(debouncer, event)
    debounce_clock.now += 2
    assert not _send(debouncer, event)
    assert not _send(debouncer, _event(3))
    assert _send(debouncer, _event(5))
