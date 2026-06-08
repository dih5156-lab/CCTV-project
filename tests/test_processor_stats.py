"""
test_processor_stats.py — ProcessorStats / _EventDebouncer 단위 테스트
"""
import time
from unittest.mock import MagicMock

import pytest

from src.core.processor import ProcessorStats, _EventDebouncer

# ---------------------------------------------------------------------------
# ProcessorStats
# ---------------------------------------------------------------------------


class TestProcessorStats:
    def _make(self, **kwargs) -> ProcessorStats:
        s = ProcessorStats()
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    def test_get_fps_returns_zero_at_start(self):
        """방금 생성하면 경과 시간이 매우 짧아 fps >= 0."""
        s = ProcessorStats()
        assert s.get_fps() >= 0

    def test_get_fps_nonzero(self):
        s = ProcessorStats()
        s.frames_processed = 300
        s.start_time = time.time() - 10  # 10초 경과
        fps = s.get_fps()
        assert fps == pytest.approx(30.0, abs=1.0)

    def test_get_avg_inference_time_zero_count(self):
        assert ProcessorStats().get_avg_inference_time() == 0.0

    def test_get_avg_inference_time(self):
        s = ProcessorStats()
        s.inference_count = 100
        s.total_inference_time = 2.0   # 2초 → 평균 20ms
        assert s.get_avg_inference_time() == pytest.approx(20.0)

    def test_snapshot_is_dict(self):
        s = ProcessorStats()
        snap = s.snapshot()
        assert isinstance(snap, dict)
        assert "frames_processed" in snap

    def test_with_derived_stats_adds_fps(self):
        raw = {
            "frames_processed": 600,
            "start_time": time.time() - 60,  # 60초 전 시작
            "inference_count": 0,
            "total_inference_time": 0.0,
        }
        result = ProcessorStats.with_derived_stats(raw)
        assert "fps" in result
        assert result["fps"] == pytest.approx(10.0, abs=1.0)
        assert "uptime_seconds" in result
        assert result["uptime_seconds"] == pytest.approx(60.0, abs=2.0)

    def test_with_derived_stats_avg_inference(self):
        raw = {
            "frames_processed": 0,
            "start_time": time.time(),
            "inference_count": 10,
            "total_inference_time": 0.5,  # 500ms → 50ms 평균
        }
        result = ProcessorStats.with_derived_stats(raw)
        assert result["avg_inference_ms"] == pytest.approx(50.0)

    def test_to_dict_includes_derived(self):
        s = ProcessorStats()
        s.frames_processed = 100
        s.start_time = time.time() - 10
        d = s.to_dict()
        assert "fps" in d
        assert "uptime_seconds" in d
        assert "avg_inference_ms" in d


# ---------------------------------------------------------------------------
# _EventDebouncer
# ---------------------------------------------------------------------------


class TestEventDebouncer:
    def _make_config(self, debounce_enabled=True, debounce_seconds=5.0, retention_hours=24,
                      fall_sustained_seconds=10.0, fall_resend_cooldown=60.0, fall_gap_reset_seconds=2.0,
                      head_resend_cooldown=30.0, head_gap_reset_seconds=5.0):
        cfg = MagicMock()
        cfg.events.debounce_enabled = debounce_enabled
        cfg.events.debounce_seconds = debounce_seconds
        cfg.events.event_retention_hours = retention_hours
        cfg.events.fall_sustained_seconds = fall_sustained_seconds
        cfg.events.fall_resend_cooldown = fall_resend_cooldown
        cfg.events.fall_gap_reset_seconds = fall_gap_reset_seconds
        cfg.events.head_resend_cooldown = head_resend_cooldown
        cfg.events.head_gap_reset_seconds = head_gap_reset_seconds
        return cfg

    def test_should_send_when_debounce_disabled(self):
        cfg = self._make_config(debounce_enabled=False)
        stat = MagicMock()
        d = _EventDebouncer(cfg, stat)
        assert d.should_send("cam1", "helmet", 1) is True
        # 두 번 연속도 True
        assert d.should_send("cam1", "helmet", 1) is True

    def test_should_send_first_call(self):
        cfg = self._make_config()
        d = _EventDebouncer(cfg, MagicMock())
        assert d.should_send("cam1", "helmet", 1) is True

    def test_should_not_send_within_debounce_window(self):
        cfg = self._make_config(debounce_seconds=10.0)
        d = _EventDebouncer(cfg, MagicMock())
        d.should_send("cam1", "helmet", 1)   # 첫 전송 → True
        assert d.should_send("cam1", "helmet", 1) is False  # 10초 이내 → False

    def test_should_send_after_debounce_window(self):
        cfg = self._make_config(debounce_seconds=0.01)
        d = _EventDebouncer(cfg, MagicMock())
        d.should_send("cam1", "helmet", 1)
        time.sleep(0.05)
        assert d.should_send("cam1", "helmet", 1) is True

    def test_head_event_state_change_sends_immediately(self):
        """head 이벤트: 재등장(상태 변화) 시 즉시 전송."""
        cfg = self._make_config(head_resend_cooldown=9999.0, head_gap_reset_seconds=0.02)
        d = _EventDebouncer(cfg, MagicMock())
        # 첫 감지 → gap이 길었으므로 상태 변화 → 즉시 전송
        assert d.should_send("cam1", "head", 1) is True
        # 연속 감지 → cooldown(9999s) 이내 → 억제
        assert d.should_send("cam1", "head", 1) is False
        time.sleep(0.03)  # gap_reset 이상 경과
        # 재등장 → 상태 변화 → 즉시 전송
        assert d.should_send("cam1", "head", 1) is True

    def test_head_event_resend_cooldown(self):
        """head 이벤트: cooldown 경과 후 재전송 허용."""
        cfg = self._make_config(head_resend_cooldown=0.02, head_gap_reset_seconds=9999.0)
        d = _EventDebouncer(cfg, MagicMock())
        assert d.should_send("cam1", "head", 1) is True   # 첫 전송
        assert d.should_send("cam1", "head", 1) is False  # cooldown 이내
        time.sleep(0.03)
        assert d.should_send("cam1", "head", 1) is True   # cooldown 경과 후 재전송

    def test_fall_detected_requires_sustained_duration(self):
        """낙상은 sustained_seconds 이상 지속되어야 전송된다 (연속 감지 시뮬레이션)."""
        cfg = self._make_config(fall_sustained_seconds=0.05, fall_resend_cooldown=9999.0, fall_gap_reset_seconds=0.1)
        d = _EventDebouncer(cfg, MagicMock())
        # 첫 감지 (t≈0): first_seen 설정, duration=0 → False
        assert d.should_send("cam1", "fall_detected", 1) is False
        time.sleep(0.01)
        # gap(0.01) < gap_reset(0.1) → 리셋 안 됨, duration≈0.01 < sustained(0.05) → False
        assert d.should_send("cam1", "fall_detected", 1) is False
        time.sleep(0.06)
        # gap(0.06) < gap_reset(0.1) → 리셋 안 됨, duration≈0.07 > sustained(0.05) → True
        assert d.should_send("cam1", "fall_detected", 1) is True

    def test_fall_detected_resend_cooldown(self):
        """낙상 전송 후 cooldown 동안 재전송 억제"""
        cfg = self._make_config(fall_sustained_seconds=0.01, fall_resend_cooldown=9999.0, fall_gap_reset_seconds=0.1)
        d = _EventDebouncer(cfg, MagicMock())
        # 첫 감지 (t≈0): first_seen 설정
        d.should_send("cam1", "fall_detected", 1)
        time.sleep(0.02)
        # gap(0.02) < gap_reset(0.1) → 리셋 안 됨, duration≈0.02 > sustained(0.01) → True
        assert d.should_send("cam1", "fall_detected", 1) is True
        # 즉시 재호출 → cooldown(9999s) 이내 → False
        assert d.should_send("cam1", "fall_detected", 1) is False

    def test_fall_detected_timer_resets_on_gap(self):
        """낙상이 gap_reset_seconds 이상 끊기면 타이머 리셋"""
        cfg = self._make_config(fall_sustained_seconds=0.05, fall_resend_cooldown=9999.0, fall_gap_reset_seconds=0.02)
        d = _EventDebouncer(cfg, MagicMock())
        d.should_send("cam1", "fall_detected", 1)   # 첫 감지 (t≈0)
        time.sleep(0.03)  # gap(0.03) > gap_reset(0.02) → 타이머 리셋
        d.should_send("cam1", "fall_detected", 1)   # 리셋 후 first_seen 재설정
        time.sleep(0.01)  # duration≈0.01 < sustained(0.05)
        assert d.should_send("cam1", "fall_detected", 1) is False  # 리셋 때문에 미전송

    def test_different_cameras_independent(self):
        cfg = self._make_config(debounce_seconds=9999.0)
        d = _EventDebouncer(cfg, MagicMock())
        d.should_send("cam1", "helmet", 1)
        assert d.should_send("cam2", "helmet", 1) is True  # 다른 카메라 → 허용

    def test_different_object_ids_independent(self):
        cfg = self._make_config(debounce_seconds=9999.0)
        d = _EventDebouncer(cfg, MagicMock())
        d.should_send("cam1", "helmet", 1)
        assert d.should_send("cam1", "helmet", 2) is True  # 다른 ID → 허용

    def test_cleanup_removes_expired_keys(self):
        cfg = self._make_config(retention_hours=0)  # 0시간 → 즉시 만료
        d = _EventDebouncer(cfg, MagicMock())
        d.should_send("cam1", "helmet", 1)
        removed = d.cleanup(max_age_hours=0)
        assert removed >= 1

    def test_cleanup_keeps_recent_keys(self):
        cfg = self._make_config(retention_hours=24)
        d = _EventDebouncer(cfg, MagicMock())
        d.should_send("cam1", "helmet", 1)
        removed = d.cleanup(max_age_hours=24)
        assert removed == 0

    def test_stat_callback_called_on_filter(self):
        cfg = self._make_config(debounce_seconds=9999.0)
        stat = MagicMock()
        d = _EventDebouncer(cfg, stat)
        d.should_send("cam1", "helmet", 1)   # 첫 전송
        d.should_send("cam1", "helmet", 1)   # 두 번째 → events_filtered 콜백
        stat.assert_called_with("events_filtered")

    def test_save_locally_creates_file(self, tmp_path, monkeypatch):
        """save_locally 가 파일을 생성하는지 확인."""
        monkeypatch.chdir(tmp_path)
        cfg = self._make_config()
        d = _EventDebouncer(cfg, MagicMock())
        d.save_locally({"camera_id": "test_cam", "type": "helmet"})
        backup_dir = tmp_path / "event_backup"
        files = list(backup_dir.glob("*.json"))
        assert len(files) == 1
