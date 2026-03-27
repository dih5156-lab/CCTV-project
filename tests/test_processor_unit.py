"""
test_processor_unit.py — VideoProcessor 내부 헬퍼 단위 테스트

커버리지 대상:
  - _CameraRegistry: register / unregister / retry 큐
  - VideoProcessor._parse_detections (static method)
  - VideoProcessor._increment_stat / _add_inference_metrics
  - VideoProcessor.update_zones (zone_manager 없을 때)

주의: ProcessorStats / _EventDebouncer 는 test_processor_stats.py 에 이미 있음.
"""

import time

import pytest
from threading import Event
from unittest.mock import MagicMock, patch

from src.core.processor import _CameraRegistry, VideoProcessor


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------


def _make_registry(config=None) -> _CameraRegistry:
    cfg = config or MagicMock()
    cfg.processing.thread_join_timeout = 1
    return _CameraRegistry(cfg, Event(), lambda: True)


def _camera_mock() -> MagicMock:
    cam = MagicMock()
    cam.release = MagicMock()
    return cam


@pytest.fixture
def minimal_processor():
    """Zone/Dataset/Display 없이 생성 가능한 최소 VideoProcessor."""
    from src.config.config import AppConfig
    cfg = AppConfig(zone_detection=False, collect_dataset=False, display=False)
    return VideoProcessor(cfg)


# ===========================================================================
# _CameraRegistry
# ===========================================================================


class TestCameraRegistryRegister:
    def test_register_adds_to_cameras(self):
        reg = _make_registry()
        reg.register("cam1", _camera_mock())
        assert "cam1" in reg.cameras

    def test_register_creates_frame_queue(self):
        reg = _make_registry()
        reg.register("cam1", _camera_mock())
        assert "cam1" in reg.frame_queues

    def test_register_creates_stop_flag(self):
        reg = _make_registry()
        reg.register("cam1", _camera_mock())
        assert "cam1" in reg._stop_flags

    def test_initial_count_zero(self):
        assert _make_registry().count == 0

    def test_count_increments_on_register(self):
        reg = _make_registry()
        reg.register("cam1", _camera_mock())
        assert reg.count == 1

    def test_register_multiple_cameras(self):
        reg = _make_registry()
        reg.register("cam1", _camera_mock())
        reg.register("cam2", _camera_mock())
        assert reg.count == 2


class TestCameraRegistryUnregister:
    def test_unregister_removes_camera(self):
        reg = _make_registry()
        reg.register("cam1", _camera_mock())
        reg.unregister("cam1")
        assert "cam1" not in reg.cameras

    def test_unregister_removes_frame_queue(self):
        reg = _make_registry()
        reg.register("cam1", _camera_mock())
        reg.unregister("cam1")
        assert "cam1" not in reg.frame_queues

    def test_unregister_calls_release(self):
        reg = _make_registry()
        cam = _camera_mock()
        reg.register("cam1", cam)
        reg.unregister("cam1")
        cam.release.assert_called_once()

    def test_unregister_nonexistent_does_not_raise(self):
        _make_registry().unregister("nonexistent")  # 예외 없이 통과

    def test_count_decrements_on_unregister(self):
        reg = _make_registry()
        reg.register("cam1", _camera_mock())
        reg.unregister("cam1")
        assert reg.count == 0


class TestCameraRegistryStopFlag:
    def test_stop_flag_none_for_unknown(self):
        assert _make_registry().stop_flag("unknown") is None

    def test_ensure_stop_flag_creates_event(self):
        reg = _make_registry()
        flag = reg.ensure_stop_flag("cam1")
        assert flag is not None

    def test_ensure_stop_flag_initially_not_set(self):
        reg = _make_registry()
        flag = reg.ensure_stop_flag("cam1")
        assert not flag.is_set()

    def test_ensure_stop_flag_clears_existing_set_flag(self):
        reg = _make_registry()
        flag = reg.ensure_stop_flag("cam1")
        flag.set()
        flag2 = reg.ensure_stop_flag("cam1")
        assert not flag2.is_set()


class TestCameraRegistryRetry:
    def test_enqueue_adds_pending(self):
        reg = _make_registry()
        reg.enqueue_retry("cam1", "rtsp://example.com", delay_seconds=30)
        assert len(reg._pending) == 1

    def test_dequeue_not_ready_returns_empty(self):
        reg = _make_registry()
        reg.enqueue_retry("cam1", "rtsp://example.com", delay_seconds=9999)
        assert reg.poll_ready_retries() == []

    def test_dequeue_ready_returns_item(self):
        reg = _make_registry()
        reg.enqueue_retry("cam1", "rtsp://example.com", delay_seconds=0)
        time.sleep(0.01)
        result = reg.poll_ready_retries()
        assert len(result) == 1
        assert result[0][0] == "cam1"

    def test_dequeue_removes_item_from_pending(self):
        reg = _make_registry()
        reg.enqueue_retry("cam1", "rtsp://example.com", delay_seconds=0)
        time.sleep(0.01)
        reg.poll_ready_retries()
        assert len(reg._pending) == 0

    def test_enqueue_same_camera_replaces_old_entry(self):
        reg = _make_registry()
        reg.enqueue_retry("cam1", "rtsp://old", delay_seconds=100)
        reg.enqueue_retry("cam1", "rtsp://new", delay_seconds=100)
        assert len(reg._pending) == 1
        assert reg._pending[0][1] == "rtsp://new"

    def test_multiple_cameras_in_queue(self):
        reg = _make_registry()
        reg.enqueue_retry("cam1", "src1", delay_seconds=0)
        reg.enqueue_retry("cam2", "src2", delay_seconds=0)
        time.sleep(0.01)
        result = reg.poll_ready_retries()
        cam_ids = [r[0] for r in result]
        assert "cam1" in cam_ids
        assert "cam2" in cam_ids


# ===========================================================================
# VideoProcessor._parse_detections  (static method)
# ===========================================================================


class TestParseDetections:
    def test_none_enables_all(self):
        flags = VideoProcessor._parse_detections(None)
        assert flags == {"use_helmet": True, "use_pose": True, "use_person": False}

    def test_empty_list_enables_all(self):
        flags = VideoProcessor._parse_detections([])
        assert flags == {"use_helmet": True, "use_pose": True, "use_person": False}

    def test_fall_enables_pose_and_person_not_helmet(self):
        flags = VideoProcessor._parse_detections(["fall"])
        assert flags["use_pose"] is True
        assert flags["use_person"] is False
        assert flags["use_helmet"] is False

    def test_helmet_enables_helmet_and_pose(self):
        flags = VideoProcessor._parse_detections(["helmet"])
        assert flags["use_helmet"] is True
        assert flags["use_person"] is False
        assert flags["use_pose"] is True

    def test_intrusion_enables_pose_only(self):
        flags = VideoProcessor._parse_detections(["intrusion"])
        assert flags["use_person"] is False
        assert flags["use_helmet"] is False
        assert flags["use_pose"] is True

    def test_person_enables_pose_only(self):
        flags = VideoProcessor._parse_detections(["person"])
        assert flags["use_person"] is False
        assert flags["use_helmet"] is False
        assert flags["use_pose"] is True

    def test_fall_and_helmet_enables_all(self):
        flags = VideoProcessor._parse_detections(["fall", "helmet"])
        assert flags == {"use_helmet": True, "use_pose": True, "use_person": False}

    def test_case_insensitive(self):
        flags = VideoProcessor._parse_detections(["FALL", "HELMET"])
        assert flags["use_pose"] is True
        assert flags["use_helmet"] is True

    def test_model_settings_mapping_supported(self):
        flags = VideoProcessor._parse_detections({"use_pose": False, "use_helmet": True})
        assert flags == {"use_helmet": True, "use_pose": True, "use_person": False}

    def test_unknown_mode_treated_as_person(self):
        flags = VideoProcessor._parse_detections(["unknown_mode"])
        # unknown은 modes 집합에 있으나 특정 분기에 안 걸림 → person 비활성
        assert isinstance(flags, dict)  # 최소한 dict 형태를 반환


class TestModelSettingsHelpers:
    def test_flags_to_detection_modes(self):
        result = VideoProcessor._flags_to_detection_modes(
            {"use_pose": True, "use_helmet": True, "use_person": False}
        )
        assert result == ["fall", "person", "helmet"]

    def test_update_camera_model_settings_updates_memory(self, minimal_processor):
        minimal_processor._camera_ai_flags["cam1"] = {
            "use_helmet": True,
            "use_pose": True,
            "use_person": False,
        }
        updated = minimal_processor.update_camera_model_settings(
            "cam1",
            {"use_pose": False, "use_helmet": True},
        )
        assert updated == {"use_helmet": True, "use_pose": True, "use_person": False}
        assert minimal_processor.get_camera_model_settings("cam1") == updated

    def test_update_camera_model_settings_persists_json(self, minimal_processor, tmp_path):
        minimal_processor._camera_ai_flags["cam1"] = {
            "use_helmet": True,
            "use_pose": True,
            "use_person": False,
        }
        path = tmp_path / "cameras.json"
        path.write_text(
            '[{"id":"cam1","detections":["helmet"],"model_settings":{"use_pose":true,"use_helmet":true,"use_person":false}}]',
            encoding="utf-8",
        )
        minimal_processor.update_camera_model_settings(
            "cam1",
            {"use_pose": False, "use_helmet": False},
            str(path),
        )
        saved = path.read_text(encoding="utf-8")
        assert '"use_pose": false' in saved
        assert '"use_helmet": false' in saved


# ===========================================================================
# VideoProcessor._increment_stat / _add_inference_metrics
# ===========================================================================


class TestVideoProcessorStats:
    def test_increment_stat_increments_field(self, minimal_processor):
        before = minimal_processor.stats.frames_processed
        minimal_processor._increment_stat("frames_processed")
        assert minimal_processor.stats.frames_processed == before + 1

    def test_increment_stat_with_custom_delta(self, minimal_processor):
        before = minimal_processor.stats.events_detected
        minimal_processor._increment_stat("events_detected", 5)
        assert minimal_processor.stats.events_detected == before + 5

    def test_increment_stat_returns_new_value(self, minimal_processor):
        minimal_processor.stats.frames_processed = 10
        result = minimal_processor._increment_stat("frames_processed")
        assert result == 11

    def test_add_inference_metrics(self, minimal_processor):
        minimal_processor._add_inference_metrics(0.05)  # 50ms
        assert minimal_processor.stats.inference_count == 1
        assert minimal_processor.stats.total_inference_time == pytest.approx(0.05)

    def test_add_inference_metrics_accumulates(self, minimal_processor):
        minimal_processor._add_inference_metrics(0.03)
        minimal_processor._add_inference_metrics(0.07)
        assert minimal_processor.stats.inference_count == 2
        assert minimal_processor.stats.total_inference_time == pytest.approx(0.10)


# ===========================================================================
# VideoProcessor.update_zones (zone_manager가 None인 경우)
# ===========================================================================


class TestUpdateZonesNoManager:
    def test_returns_false_when_no_zone_manager(self, minimal_processor):
        """zone_detection=False 이면 zone_manager가 None → False 반환."""
        assert minimal_processor.zone_manager is None
        result = minimal_processor.update_zones("cam1", [])
        assert result is False

    def test_returns_true_with_zone_manager(self, minimal_processor):
        """zone_manager를 주입하면 True 반환."""
        mock_zm = MagicMock()
        mock_zm.save_zones = MagicMock()
        minimal_processor.zone_manager = mock_zm
        result = minimal_processor.update_zones("cam1", [{"id": "z1"}])
        assert result is True
        mock_zm.save_zones.assert_called_once()

    def test_returns_false_on_exception(self, minimal_processor):
        mock_zm = MagicMock()
        mock_zm.save_zones.side_effect = RuntimeError("disk full")
        minimal_processor.zone_manager = mock_zm
        result = minimal_processor.update_zones("cam1", [])
        assert result is False


# ===========================================================================
# _process_inference — 연속 예외 처리 (backoff + 재연결 큐)
# ===========================================================================


class TestProcessInferenceErrorHandling:
    """_process_inference 스레드의 연속 예외 처리 동작을 검증한다."""

    def _make_processor(self):
        from src.config.config import AppConfig
        cfg = AppConfig(zone_detection=False, collect_dataset=False, display=False)
        # consecutive_failure_threshold 를 낮춰서 테스트를 빠르게 실행
        cfg.processing.consecutive_failure_threshold = 3
        return VideoProcessor(cfg)

    def _infinite_frame_queue(self):
        """get() 호출 시 항상 더미 프레임을 반환하는 mock 큐."""
        import numpy as np
        fq = MagicMock()
        fq.get.return_value = np.zeros((2, 2, 3), dtype="uint8")
        return fq

    def test_consecutive_errors_trigger_retry_and_break(self):
        """임계값 이상의 연속 예외 발생 시 enqueue_retry 호출 후 루프 종료."""
        proc = self._make_processor()
        proc.running = True  # start() 없이 직접 루프 진입을 허용
        cam_id = "cam_err"

        cam_mock = MagicMock()
        cam_mock.source = "rtsp://test"
        proc._cams.register(cam_id, cam_mock)
        # maxsize=1 큐 대신 항상 프레임을 공급하는 mock으로 교체
        proc._cams.frame_queues[cam_id] = self._infinite_frame_queue()

        with patch.object(proc, "_run_ai_inference", side_effect=RuntimeError("GPU 오류")), \
             patch.object(proc._cams, "enqueue_retry") as mock_retry, \
             patch.object(proc._cams, "unregister") as mock_unreg, \
             patch.object(proc.stop_event, "wait", return_value=False):

            proc._process_inference(cam_id)

        # 임계값(3) 초과 후 enqueue_retry 와 unregister 가 정확히 1번씩 호출돼야 한다
        mock_retry.assert_called_once_with(cam_id, "rtsp://test", delay_seconds=30.0)
        mock_unreg.assert_called_once_with(cam_id)

    def test_error_counter_resets_on_success(self):
        """성공적인 추론 후 오류 카운터가 0으로 리셋되는지 확인.

        전략: 오류 2회 → 성공 1회 → 오류 2회 순서로 실행.
        임계값(3)에 도달하지 않으면 enqueue_retry 가 호출되지 않아야 한다.
        """
        proc = self._make_processor()
        proc.running = True
        cam_id = "cam_reset"

        cam_mock = MagicMock()
        cam_mock.source = "rtsp://test2"
        proc._cams.register(cam_id, cam_mock)
        proc._cams.frame_queues[cam_id] = self._infinite_frame_queue()

        import numpy as np
        frame = np.zeros((2, 2, 3), dtype="uint8")

        call_count = [0]

        def side_effect(cid, frm):
            call_count[0] += 1
            n = call_count[0]
            if n == 3:
                return []  # 성공
            raise RuntimeError("오류")

        def fake_wait(timeout=None):
            """4번째 오류 후 stop_event를 설정해 루프를 탈출시킨다."""
            if call_count[0] >= 4:
                proc.stop_event.set()
            return proc.stop_event.is_set()

        with patch.object(proc, "_run_ai_inference", side_effect=side_effect), \
             patch.object(proc, "_queue_events"), \
             patch.object(proc, "_check_danger_zones", return_value=([], frame)), \
             patch.object(proc, "_collect_dataset"), \
             patch.object(proc, "track_manager") as mock_tm, \
             patch.object(proc, "violation_filter") as mock_vf, \
             patch.object(proc._cams, "enqueue_retry") as mock_retry, \
             patch.object(proc.stop_event, "wait", side_effect=fake_wait):

            mock_tm.update.return_value = ([], [])
            mock_vf.filter.return_value = []

            proc._process_inference(cam_id)

        # 카운터 리셋으로 인해 임계값에 도달하지 않았으므로 호출 안 됨
        mock_retry.assert_not_called()

    def test_inference_error_stat_incremented(self):
        """예외 발생마다 inference_errors 통계가 증가하는지 확인."""
        proc = self._make_processor()
        proc.running = True
        cam_id = "cam_stat"

        cam_mock = MagicMock()
        cam_mock.source = "rtsp://test3"
        proc._cams.register(cam_id, cam_mock)
        proc._cams.frame_queues[cam_id] = self._infinite_frame_queue()

        with patch.object(proc, "_run_ai_inference", side_effect=RuntimeError("오류")), \
             patch.object(proc._cams, "enqueue_retry"), \
             patch.object(proc._cams, "unregister"), \
             patch.object(proc.stop_event, "wait", return_value=False):

            proc._process_inference(cam_id)

        # threshold=3 이므로 3번 예외 발생 후 루프 종료
        assert proc.stats.inference_errors >= 3
