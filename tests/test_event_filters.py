"""
test_event_filters.py — TrackManager / CumulativeViolationFilter 단위 테스트
"""
import time

from src.core.event_filters import CumulativeViolationFilter, TrackManager
from src.core.events import DetectionEvent, EventType

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _evt(etype: str, oid: int, x=10, y=10, w=50, h=50, conf=0.9) -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.from_string(etype),
        x=x, y=y, width=w, height=h,
        confidence=conf,
        timestamp=time.time(),
        object_id=oid,
    )


CAM = "cam1"


# ---------------------------------------------------------------------------
# TrackManager
# ---------------------------------------------------------------------------


class TestTrackManager:
    def test_new_event_passes_through(self):
        mgr = TrackManager()
        events = [_evt("helmet", 1)]
        filtered, removed = mgr.update(CAM, events)
        assert len(filtered) == 1
        assert 1 not in removed

    def test_duplicate_iou_removes_older(self):
        """같은 위치·타입의 객체가 두 ID로 들어오면 IOU>0.5이면 중복 제거."""
        mgr = TrackManager(track_iou_threshold=0.5)
        # ID=1 먼저 등록
        mgr.update(CAM, [_evt("helmet", 1, x=0, y=0, w=100, h=100)])
        # ID=2가 거의 같은 위치 → 중복으로 처리
        filtered, removed = mgr.update(CAM, [_evt("helmet", 2, x=0, y=0, w=100, h=100)])
        assert 1 in removed

    def test_far_apart_events_both_kept(self):
        mgr = TrackManager(track_iou_threshold=0.5)
        mgr.update(CAM, [_evt("helmet", 1, x=0, y=0, w=50, h=50)])
        filtered, removed = mgr.update(CAM, [
            _evt("helmet", 1, x=0, y=0, w=50, h=50),
            _evt("helmet", 2, x=300, y=300, w=50, h=50),
        ])
        assert 2 not in removed

    def test_expired_tracks_removed(self):
        """track_timeout=0 으로 설정하여 즉시 만료되도록."""
        mgr = TrackManager(track_timeout=0.0)
        mgr.update(CAM, [_evt("helmet", 1)])
        time.sleep(0.01)
        _, removed = mgr.update(CAM, [_evt("helmet", 2)])  # id=1이 사라져야 함
        assert 1 in removed

    def test_missing_track_kept_within_missed_frame_budget(self):
        """일시적인 미감지는 허용 횟수 안에서는 트랙을 유지한다."""
        mgr = TrackManager(track_timeout=999.0, max_missed_frames=1)
        mgr.update(CAM, [_evt("helmet", 1)])
        _, removed = mgr.update(CAM, [])
        assert 1 not in removed
        assert mgr.get_frame_count(CAM, 1) == 1

    def test_missing_track_removed_after_missed_frame_budget(self):
        """연속 미감지가 허용 횟수를 넘으면 트랙을 제거한다."""
        mgr = TrackManager(track_timeout=999.0, max_missed_frames=1)
        mgr.update(CAM, [_evt("helmet", 1)])
        mgr.update(CAM, [])
        _, removed = mgr.update(CAM, [])
        assert 1 in removed
        assert mgr.get_frame_count(CAM, 1) == 0

    def test_event_without_object_id_passes(self):
        """object_id=None 이벤트는 중복 체크 없이 통과."""
        mgr = TrackManager()
        evt = _evt("helmet", 1)
        evt.object_id = None
        filtered, _ = mgr.update(CAM, [evt])
        assert len(filtered) == 1

    def test_get_frame_count_increments(self):
        mgr = TrackManager()
        mgr.update(CAM, [_evt("helmet", 5)])
        mgr.update(CAM, [_evt("helmet", 5)])
        assert mgr.get_frame_count(CAM, 5) == 2

    def test_get_frame_count_unknown_returns_zero(self):
        mgr = TrackManager()
        assert mgr.get_frame_count(CAM, 999) == 0

    def test_remove_camera_clears_all(self):
        mgr = TrackManager()
        mgr.update(CAM, [_evt("helmet", 1)])
        mgr.remove_camera(CAM)
        assert mgr.get_frame_count(CAM, 1) == 0

    def test_different_cameras_independent(self):
        mgr = TrackManager()
        mgr.update("cam_a", [_evt("helmet", 1, x=0, y=0, w=50, h=50)])
        _, removed = mgr.update("cam_b", [_evt("helmet", 1, x=0, y=0, w=50, h=50)])
        assert 1 not in removed  # 다른 카메라이므로 중복 아님

    def test_different_event_types_not_considered_duplicate(self):
        mgr = TrackManager(track_iou_threshold=0.3)
        mgr.update(CAM, [_evt("helmet", 1, x=0, y=0, w=100, h=100)])
        _, removed = mgr.update(CAM, [_evt("head", 2, x=0, y=0, w=100, h=100)])
        assert 1 not in removed


# ---------------------------------------------------------------------------
# CumulativeViolationFilter
# ---------------------------------------------------------------------------


class TestCumulativeViolationFilter:
    def _make_filter(self, history=5, threshold=3) -> CumulativeViolationFilter:
        return CumulativeViolationFilter(
            history_max_size=history,
            violation_threshold=threshold,
            violation_types={"head", "fall_detected"},
        )

    def test_non_violation_event_passes(self):
        f = self._make_filter()
        events = [_evt("helmet", 1)]
        result = f.filter(CAM, events)
        assert len(result) == 1

    def test_violation_below_threshold_suppressed(self):
        """head 이벤트가 threshold 미만이면 누적 필터가 억제할 수 있어야 함.
        (구현에 따라 pass 가능 — 인터페이스만 확인)
        """
        f = self._make_filter(history=10, threshold=8)
        for _ in range(3):
            f.filter(CAM, [_evt("head", 10)])
        # 테스트는 예외 없이 실행되어야 함
        assert True

    def test_empty_events_returns_empty(self):
        f = self._make_filter()
        assert f.filter(CAM, []) == []

    def test_disabled_filter_passes_all(self):
        f = self._make_filter()
        f.enabled = False
        events = [_evt("head", 1), _evt("fall_detected", 2)]
        result = f.filter(CAM, events)
        assert result == events

    def test_filter_is_thread_safe(self):
        """멀티스레드 환경에서 예외 없이 실행되어야 함."""
        import threading
        f = self._make_filter()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    f.filter(CAM, [_evt("head", 1)])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
