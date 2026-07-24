"""test_deepstream_processor.py — DeepStreamProcessor 단위 테스트.

[테스트 전략]
  - Windows / CPU 환경에서도 실행 가능한 테스트만 포함한다.
  - DeepStream 의존 테스트는 DEEPSTREAM_AVAILABLE = True 인 환경에서만 실행된다.
  - pyds 객체는 MagicMock 으로 대체하여 인터페이스 정합성만 검증한다.

[테스트 범주]
  1. 가용성 감지 — DEEPSTREAM_AVAILABLE 플래그
  2. CPU 환경 RuntimeError — pyds 없을 때 생성 시도
  3. 인터페이스 정합성 — BaseProcessor 서브클래스 여부
  4. create_processor 팩토리 — USE_DEEPSTREAM 환경 변수 분기
"""

from __future__ import annotations

import json
import os
import threading
import types
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.core.deepstream_processor as deepstream_processor
from src.core._context_event_store import ContextEventStore
from src.core._preview_frame_store import PreviewFrameStore
from src.core.ai._fall_detector import FallDetector
from src.core.base_processor import BaseProcessor
from src.core.deepstream_processor import DEEPSTREAM_AVAILABLE, DeepStreamProcessor
from src.core.event_debouncer import EventDebouncer
from src.core.events import DetectionEvent, EventType
from src.core.processor import VideoProcessor

# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_config():
    """최소 AppConfig mock."""
    cfg = MagicMock()
    cfg.events.queue_max_size = 10
    cfg.events.debounce_enabled = True
    cfg.events.debounce_seconds = 3.0
    cfg.detection.fall_height_ratio = 0.55
    cfg.detection.device = "cpu"
    cfg.processing.min_track_frames = 1
    cfg.processing.detection_history_size = 10
    cfg.processing.violation_threshold = 2
    cfg.processing.cumulative_detection_enabled = True
    cfg.appearance.enabled = False
    cfg.appearance.backend = "hsv"
    cfg.appearance.model_path = None
    cfg.appearance.label_map_path = None
    cfg.appearance.runtime = "auto"
    cfg.appearance.input_size = 224
    cfg.appearance.score_threshold = 0.5
    cfg.appearance.bbox_expand_ratio = 0.1
    cfg.zone_detection = False
    cfg.zones_config = "zones_config.json"
    cfg.mqtt.broker = "localhost"
    cfg.mqtt.port = 1883
    cfg.mqtt.topic_prefix = "cctv/ai/events"
    cfg.mqtt.client_id_prefix = "test"
    cfg.mqtt.qos = 0
    cfg.mqtt.retain = False
    return cfg


@pytest.fixture(autouse=True)
def mock_heavy_deepstream_dependencies(monkeypatch):
    """DeepStream 인터페이스 테스트에서 모델/네트워크 의존성은 로딩하지 않는다."""
    monkeypatch.setattr(
        "src.core.deepstream_processor.FaceRecognitionEngine",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "src.core.deepstream_processor.AppearanceAnalyzer",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "src.core.deepstream_processor.AppearancePipeline",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "src.core.deepstream_processor.MqttEventPublisher",
        lambda *args, **kwargs: MagicMock(),
    )


# ---------------------------------------------------------------------------
# 1. 가용성 플래그
# ---------------------------------------------------------------------------


def test_deepstream_available_is_bool():
    """DEEPSTREAM_AVAILABLE 은 반드시 bool 타입이어야 한다."""
    assert isinstance(DEEPSTREAM_AVAILABLE, bool)


def test_deepstream_module_probe_does_not_import_native_bindings(monkeypatch):
    """가용성 확인은 pyds import 없이 find_spec만 사용해야 한다."""
    find_spec_calls = []

    def fake_find_spec(name):
        find_spec_calls.append(name)
        return object()

    import_module_mock = MagicMock(side_effect=AssertionError("import_module should not be called"))

    monkeypatch.setattr(deepstream_processor.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(deepstream_processor.importlib, "import_module", import_module_mock)

    assert deepstream_processor._has_deepstream_modules() is True
    assert find_spec_calls == ["gi", "pyds"]
    import_module_mock.assert_not_called()


def test_deepstream_native_bindings_load_only_when_ensured(monkeypatch):
    """실제 pyds/Gst/GLib 로드는 명시적으로 보장할 때만 수행한다."""
    fake_gi = MagicMock()
    fake_pyds = object()
    fake_glib = object()
    fake_gst = object()

    def fake_import_module(name):
        return {
            "gi": fake_gi,
            "pyds": fake_pyds,
            "gi.repository.GLib": fake_glib,
            "gi.repository.Gst": fake_gst,
        }[name]

    monkeypatch.setattr(deepstream_processor, "DEEPSTREAM_AVAILABLE", True)
    monkeypatch.setattr(deepstream_processor, "Gst", None)
    monkeypatch.setattr(deepstream_processor, "GLib", None)
    monkeypatch.setattr(deepstream_processor, "pyds", None)
    monkeypatch.setattr(deepstream_processor.importlib, "import_module", fake_import_module)

    assert deepstream_processor._ensure_deepstream_loaded() is True
    fake_gi.require_version.assert_called_once_with("Gst", "1.0")
    assert deepstream_processor.pyds is fake_pyds
    assert deepstream_processor.GLib is fake_glib
    assert deepstream_processor.Gst is fake_gst


def test_deepstream_not_available_on_windows_ci():
    """CI/Windows 환경에서는 DEEPSTREAM_AVAILABLE == False 여야 한다.

    Jetson 실기기에서는 이 테스트를 건너뜁니다.
    """
    if DEEPSTREAM_AVAILABLE:
        pytest.skip("Jetson 환경 — DeepStream 설치됨")
    assert DEEPSTREAM_AVAILABLE is False


# ---------------------------------------------------------------------------
# 2. CPU 환경 RuntimeError 검증
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    DEEPSTREAM_AVAILABLE,
    reason="DeepStream 설치됨 — RuntimeError 테스트 불필요",
)
def test_deepstream_processor_raises_on_cpu(mock_config):
    """pyds 가 없는 환경에서 DeepStreamProcessor 생성 시 RuntimeError 발생."""
    with pytest.raises(RuntimeError, match="DeepStreamProcessor"):
        DeepStreamProcessor(mock_config)


# ---------------------------------------------------------------------------
# 3. 인터페이스 정합성
# ---------------------------------------------------------------------------


def test_deepstream_processor_is_subclass_of_base():
    """DeepStreamProcessor 는 BaseProcessor 의 서브클래스여야 한다."""
    assert issubclass(DeepStreamProcessor, BaseProcessor)


def test_video_processor_is_subclass_of_base():
    """VideoProcessor 도 BaseProcessor 의 서브클래스여야 한다."""
    assert issubclass(VideoProcessor, BaseProcessor)


def test_get_camera_frame_can_skip_copy_for_internal_postprocessing():
    """내부 후처리는 preview 프레임 추가 복사를 피할 수 있어야 한다."""
    proc = object.__new__(DeepStreamProcessor)
    frame = [[1, 2, 3]]
    proc._preview_store = PreviewFrameStore(max_fps=5.0)
    proc._preview_store.put_frame("cam1", frame, now_monotonic=1.0, wall_time=1.0)
    proc._preview_camera_id = None

    copied = proc.get_camera_frame("cam1")
    shared = proc.get_camera_frame("cam1", copy_frame=False)

    assert copied == frame
    assert copied is not frame
    assert shared is frame


def test_deepstream_get_camera_status_uses_common_fields_without_runtime():
    proc = object.__new__(DeepStreamProcessor)
    proc.running = False
    proc._preview_store = PreviewFrameStore(max_fps=5.0)
    proc._source_backoff_until = {}
    proc._source_last_error = {}
    proc._cameras = {
        "cam1": {
            "source": "rtsp://192.168.1.1/stream",
            "pad_id": 0,
            "reconnect_attempts": 1,
        }
    }

    status = proc.get_camera_status()

    assert "cam1" in status
    entry = status["cam1"]
    assert {
        "status",
        "connected",
        "source",
        "reconnect_attempts",
        "last_frame_time",
        "last_frame_age_sec",
        "pad_id",
    } <= set(entry.keys())
    assert entry["status"] == "reconnecting"
    assert entry["connected"] is False
    assert entry["source"] == "rtsp://192.168.1.1/stream"
    assert entry["pad_id"] == 0


def test_deepstream_marks_failed_source_with_backoff(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._source_failure_backoff_sec = 30.0
    proc._source_backoff_until = {}
    proc._source_last_error = {}
    proc._cameras = {
        "cam1": {
            "source": "rtsp://192.168.1.1/stream",
            "reconnect_attempts": 0,
        }
    }
    monkeypatch.setattr("src.core.deepstream_processor.time.monotonic", lambda: 100.0)

    proc._mark_source_failed("cam1", "rtsp timeout")

    assert proc._source_backoff_until["cam1"] == 130.0
    assert proc._source_last_error["cam1"] == "rtsp timeout"
    assert proc._cameras["cam1"]["reconnect_attempts"] == 1


def test_deepstream_build_source_entries_skips_backoff_camera(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._source_backoff_until = {"cam1": 130.0}
    proc._cameras = {
        "cam1": {"source": "rtsp://192.168.1.1/stream"},
        "cam2": {"source": "rtsp://192.168.1.2/stream"},
    }
    monkeypatch.setattr("src.core.deepstream_processor.time.monotonic", lambda: 100.0)

    entries = proc._build_source_entries()

    assert [(pad_id, camera_id, source_uri) for pad_id, camera_id, _info, source_uri in entries] == [
        (0, "cam2", "rtsp://192.168.1.2/stream")
    ]


def test_deepstream_next_source_retry_delay_returns_nearest_backoff(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._source_backoff_until = {"cam1": 130.0, "cam2": 160.0}
    monkeypatch.setattr("src.core.deepstream_processor.time.monotonic", lambda: 100.0)

    assert proc.next_source_retry_delay() == 30.0


def test_deepstream_camera_id_from_error_debug():
    proc = object.__new__(DeepStreamProcessor)
    proc._cameras = {"camera_1": {}, "camera_2": {}}
    message = MagicMock()
    message.src = None

    camera_id = proc._camera_id_from_message(
        message,
        "/GstPipeline:cctv-deepstream/GstDsNvUriSrcBin:src-camera_1/GstRTSPSrc:src",
    )

    assert camera_id == "camera_1"


def test_deepstream_retry_camera_clears_backoff_and_attaches_dynamically():
    proc = object.__new__(DeepStreamProcessor)
    proc.running = True
    proc._source_backoff_until = {"cam1": 130.0}
    proc._source_last_error = {"cam1": "rtsp timeout"}
    proc._cameras = {"cam1": {"source": "rtsp://old/stream"}}
    proc._pipeline_restart_pending = False
    proc._add_camera_to_pipeline = MagicMock(return_value=True)
    restart = MagicMock()
    proc._restart_pipeline_async = restart

    proc._retry_camera("cam1", "rtsp://new/stream")

    assert "cam1" not in proc._source_backoff_until
    assert "cam1" not in proc._source_last_error
    assert proc._cameras["cam1"]["source"] == "rtsp://new/stream"
    assert proc._pipeline_restart_pending is False
    proc._add_camera_to_pipeline.assert_called_once_with("cam1")
    restart.assert_not_called()


def test_deepstream_retry_camera_falls_back_to_restart_when_dynamic_attach_fails():
    proc = object.__new__(DeepStreamProcessor)
    proc.running = True
    proc._source_backoff_until = {}
    proc._source_last_error = {}
    proc._cameras = {"cam1": {"source": "rtsp://old/stream"}}
    proc._pipeline_restart_pending = False
    proc._add_camera_to_pipeline = MagicMock(return_value=False)
    restart = MagicMock()
    proc._restart_pipeline_async = restart

    proc._retry_camera("cam1", "rtsp://new/stream")

    assert proc._pipeline_restart_pending is True
    restart.assert_called_once_with("camera_retry:cam1:fallback_restart")


def test_deepstream_retry_camera_registers_missing_camera_before_restart():
    proc = object.__new__(DeepStreamProcessor)
    proc.running = True
    proc._source_backoff_until = {}
    proc._source_last_error = {}
    proc._cameras = {}
    proc._pipeline_restart_pending = False
    proc.add_camera = MagicMock(return_value=True)
    proc._add_camera_to_pipeline = MagicMock(return_value=True)
    restart = MagicMock()
    proc._restart_pipeline_async = restart

    proc._retry_camera("cam1", "rtsp://new/stream")

    proc.add_camera.assert_called_once_with("cam1", "rtsp://new/stream")
    assert proc._pipeline_restart_pending is False
    proc._add_camera_to_pipeline.assert_called_once_with("cam1")
    restart.assert_not_called()


def test_deepstream_get_stats_uses_common_fields_without_runtime():
    proc = object.__new__(DeepStreamProcessor)
    proc._frames_processed = 12
    proc._frames_dropped = 1
    proc._events_detected = 3
    proc._events_filtered = 2
    proc._events_failed = 0
    proc._output_mode = "fakesink"
    proc._preview_enabled = True
    proc._preview_store = PreviewFrameStore(max_fps=5.0)
    proc._cameras = {"cam1": {"source": "rtsp://192.168.1.1/stream"}}

    stats = proc.get_stats()

    assert {
        "backend",
        "camera_count",
        "frames_processed",
        "frames_dropped",
        "events_detected",
        "events_sent",
        "events_filtered",
        "events_dropped",
        "events_failed",
        "inference_errors",
        "camera_errors",
        "fps",
        "uptime_seconds",
        "avg_inference_ms",
    } <= set(stats.keys())
    assert stats["backend"] == "deepstream"
    assert stats["camera_count"] == 1
    assert stats["frames_processed"] == 12
    assert stats["output_mode"] == "fakesink"


def test_frame_capture_stays_enabled_for_face_when_public_preview_is_disabled():
    proc = object.__new__(DeepStreamProcessor)
    proc._preview_enabled = False
    proc._face_enabled_default = True
    proc._appearance_enabled_default = False
    proc._camera_ai_flags = {}

    assert proc._frame_capture_enabled() is True


def test_frame_capture_stays_enabled_for_camera_appearance_override():
    proc = object.__new__(DeepStreamProcessor)
    proc._preview_enabled = False
    proc._face_enabled_default = False
    proc._appearance_enabled_default = False
    proc._camera_ai_flags = {
        "cam1": {"use_face": False, "use_appearance": True},
    }

    assert proc._frame_capture_enabled() is True


def test_frame_capture_can_be_disabled_when_no_consumer_needs_frames():
    proc = object.__new__(DeepStreamProcessor)
    proc._preview_enabled = False
    proc._face_enabled_default = False
    proc._appearance_enabled_default = False
    proc._camera_ai_flags = {}

    assert proc._frame_capture_enabled() is False


def test_read_preview_max_fps_defaults_to_stream_fps(monkeypatch):
    monkeypatch.delenv("DS_PREVIEW_MAX_FPS", raising=False)
    monkeypatch.setenv("STREAM_FPS", "20")

    assert DeepStreamProcessor._read_preview_max_fps() == 20.0


def test_read_preview_max_fps_clamps_high_values(monkeypatch):
    monkeypatch.setenv("DS_PREVIEW_MAX_FPS", "120")

    assert DeepStreamProcessor._read_preview_max_fps() == 60.0


def test_preview_sample_is_pulled_even_when_throttled(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._preview_store = PreviewFrameStore(max_fps=1.0)
    proc._preview_store.last_sample_at = 100.0
    monkeypatch.setattr("src.core.deepstream_processor.time.monotonic", lambda: 100.1)
    monkeypatch.setattr(
        "src.core.deepstream_processor.Gst",
        types.SimpleNamespace(FlowReturn=types.SimpleNamespace(OK="ok")),
    )

    sample = MagicMock()
    sink = MagicMock()
    sink.emit.return_value = sample

    result = proc._on_preview_sample(sink)

    assert result == "ok"
    sink.emit.assert_called_once_with("pull-sample")
    sample.get_buffer.assert_not_called()


def test_build_source_entries_skips_integer_sources_without_runtime():
    proc = object.__new__(DeepStreamProcessor)
    proc._source_backoff_until = {}
    proc._cameras = {
        "cam1": {"source": "rtsp://192.168.1.1/stream"},
        "cam2": {"source": 0},
    }

    entries = proc._build_source_entries()

    assert len(entries) == 1
    assert entries[0][0] == 0
    assert entries[0][1] == "cam1"
    assert entries[0][3] == "rtsp://192.168.1.1/stream"


def test_deepstream_model_flags_do_not_force_pose():
    flags = DeepStreamProcessor._normalize_model_flags(
        {"use_pose": False, "use_helmet": True, "use_face": True, "use_appearance": True}
    )

    assert flags == {
        "use_helmet": True,
        "use_pose": False,
        "use_person": False,
        "use_face": True,
        "use_appearance": True,
    }


def test_deepstream_filter_events_respects_all_off():
    proc = object.__new__(DeepStreamProcessor)
    proc._camera_ai_flags = {
        "cam1": {
            "use_helmet": False,
            "use_pose": False,
            "use_person": False,
            "use_face": False,
            "use_appearance": False,
        }
    }
    events = [
        DetectionEvent(EventType.PERSON, 0, 0, 10, 10, 0.9, 1.0),
        DetectionEvent(EventType.HEAD, 0, 0, 10, 10, 0.8, 1.0),
        DetectionEvent(EventType.FALL_DETECTED, 0, 0, 10, 10, 0.7, 1.0),
    ]

    assert proc._filter_events_for_camera(events, "cam1") == []


def test_context_event_store_merges_recent_non_person_events_without_duplicates():
    now = 100.0
    store = ContextEventStore(
        ttl_sec=2.0,
        maxlen=16,
        time_factory=lambda: now,
    )
    person = DetectionEvent(EventType.PERSON, 0, 0, 10, 10, 0.9, now, object_id=1)
    head = DetectionEvent(EventType.HEAD, 5, 5, 8, 8, 0.8, now, object_id=2)
    duplicate_head = DetectionEvent(EventType.HEAD, 5, 5, 8, 8, 0.7, now, object_id=2)

    store.remember("cam1", [person, head])
    merged = store.collect("cam1", [person, duplicate_head])

    assert merged == [person, duplicate_head]


def test_context_event_store_drops_stale_cached_events():
    current_time = [100.0]
    store = ContextEventStore(
        ttl_sec=1.0,
        maxlen=16,
        time_factory=lambda: current_time[0],
    )
    head = DetectionEvent(EventType.HEAD, 5, 5, 8, 8, 0.8, 100.0, object_id=2)
    person = DetectionEvent(EventType.PERSON, 0, 0, 10, 10, 0.9, 102.0, object_id=1)

    store.remember("cam1", [head])
    current_time[0] = 102.0

    assert store.collect("cam1", [person]) == [person]


def test_deepstream_fall_detector_uses_env_thresholds(mock_config, monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    monkeypatch.setenv("DS_FALL_HEIGHT_RATIO", "0.40")
    monkeypatch.setenv("DS_FALL_ANGLE_HORIZONTAL", "55")
    monkeypatch.setenv("DS_FALL_ANGLE_INVERTED", "125")
    monkeypatch.setenv("DS_FALL_BBOX_ASPECT_RATIO", "1.35")
    monkeypatch.setenv("DS_FALL_SPAN_BBOX_ASPECT_RATIO", "1.20")
    monkeypatch.setenv("DS_FALL_KEYPOINT_SPAN_RATIO", "0.55")
    monkeypatch.setenv("DS_FALL_SCORE_THRESHOLD", "3.1")
    monkeypatch.setenv("DS_FALL_ENABLE_FOLDED_POSE", "true")
    monkeypatch.setenv("DS_FALL_SUPPRESS_SITTING_LIKE_POSE", "true")
    monkeypatch.setenv("DS_FALL_SITTING_LIKE_ASPECT_RATIO", "1.50")
    monkeypatch.setenv("DS_FALL_MIN_KEYPOINT_CONFIDENCE", "0.25")
    monkeypatch.setenv("DS_FALL_MIN_HIP_CONFIDENCE", "0.25")
    monkeypatch.setenv("DS_FALL_MIN_LEG_CONFIDENCE", "0.35")

    proc._init_event_filters(mock_config)

    assert proc._fall_detector.fall_height_ratio == 0.40
    assert proc._fall_detector.angle_horizontal == 55.0
    assert proc._fall_detector.angle_inverted == 125.0
    assert proc._fall_detector.bbox_aspect_ratio == 1.35
    assert proc._fall_detector.span_bbox_aspect_ratio == 1.20
    assert proc._fall_detector.span_ratio == 0.55
    assert proc._fall_detector.score_threshold == 3.1
    assert proc._fall_detector.enable_folded_pose is True
    assert proc._fall_detector.suppress_sitting_like_pose is True
    assert proc._fall_detector.sitting_like_aspect_ratio == 1.50
    assert proc._fall_detector.min_keypoint_confidence == 0.25
    assert proc._fall_detector.min_hip_confidence == 0.25
    assert proc._fall_detector.min_leg_confidence == 0.35


def test_deepstream_falldata_aux_operational_settings_from_env(mock_config, monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._face_work_queue = Queue(maxsize=2)
    proc._context_event_store = ContextEventStore(ttl_sec=1.0, maxlen=10)
    monkeypatch.setenv("FALLDATA_AUX_CONFIRM_BORDERLINE", "true")
    monkeypatch.setenv("FALLDATA_AUX_CONFIRM_MAX_FALL_SCORE", "4.5")
    monkeypatch.setenv("FALLDATA_AUX_COMPARE_VETO_ENABLED", "true")
    monkeypatch.setenv("FALLDATA_AUX_COMPARE_VETO_MIN_FALL_SCORE", "5.0")

    proc._init_ai_context(mock_config)

    assert proc._falldata_aux.enabled is not None
    assert proc._fall_aux_confirm_borderline is True
    assert proc._fall_aux_confirm_max_fall_score == 4.5
    assert proc._fall_aux_compare_veto_enabled is True
    assert proc._fall_aux_compare_veto_min_fall_score == 5.0


def test_deepstream_fall_pose_returns_score_metadata():
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_detector = FallDetector(angle_horizontal=55, score_threshold=3.0)
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    keypoints[0] = [100.0, 80.0, 0.9]
    keypoints[5] = [40.0, 60.0, 0.9]
    keypoints[6] = [60.0, 60.0, 0.9]
    keypoints[11] = [120.0, 70.0, 0.9]
    keypoints[12] = [140.0, 70.0, 0.9]
    keypoints[13] = [130.0, 82.0, 0.9]
    keypoints[14] = [150.0, 84.0, 0.9]
    keypoints[15] = [145.0, 88.0, 0.9]
    keypoints[16] = [165.0, 90.0, 0.9]

    result = proc._is_fall_pose(keypoints, width=160, height=100)

    assert result["is_fall"] is True
    assert result["score"] >= 3.0
    assert any(reason.startswith("torso_horizontal:") for reason in result["reasons"])


def test_deepstream_fall_pose_reports_folded_floor_near_miss():
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_detector = FallDetector(enable_folded_pose=False)
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    keypoints[0] = [100.0, 20.0, 0.05]
    keypoints[5] = [90.0, 90.0, 0.9]
    keypoints[6] = [115.0, 92.0, 0.9]
    keypoints[11] = [100.0, 155.0, 0.9]
    keypoints[12] = [125.0, 158.0, 0.9]
    keypoints[13] = [112.0, 178.0, 0.9]
    keypoints[14] = [136.0, 176.0, 0.9]
    keypoints[15] = [130.0, 190.0, 0.7]
    keypoints[16] = [150.0, 188.0, 0.7]

    result = proc._is_fall_pose(keypoints, width=110, height=220)

    assert result["is_fall"] is False
    assert result["near_miss"]["type"] == "folded_floor_pose"


def test_deepstream_fall_pose_reports_low_score_near_miss():
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_detector = FallDetector(angle_horizontal=55, score_threshold=3.0)
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    keypoints[0] = [100.0, 80.0, 0.9]
    keypoints[5] = [40.0, 60.0, 0.9]
    keypoints[6] = [60.0, 60.0, 0.9]
    keypoints[11] = [120.0, 70.0, 0.9]
    keypoints[12] = [140.0, 70.0, 0.9]
    keypoints[13] = [130.0, 140.0, 0.9]

    result = proc._is_fall_pose(keypoints, width=80, height=180)

    assert result["is_fall"] is False
    assert result["near_miss"]["type"] == "low_score_pose"
    assert result["near_miss"]["score"] > 0.0


def test_deepstream_yolo_postprocess_mode_defaults_to_vectorized(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    monkeypatch.delenv("DS_YOLO_POSTPROCESS_MODE", raising=False)

    proc._init_yolo_settings()

    assert proc._yolo_postprocess_mode == "vectorized"


def test_deepstream_yolo_postprocess_mode_rejects_unknown_value(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    monkeypatch.setenv("DS_YOLO_POSTPROCESS_MODE", "surprise")

    with pytest.raises(ValueError, match="DS_YOLO_POSTPROCESS_MODE"):
        proc._init_yolo_settings()


def test_deepstream_yolo_postprocess_metrics_report_average_and_maximum():
    proc = object.__new__(DeepStreamProcessor)
    proc._yolo_postprocess_mode = "vectorized"
    proc._yolo_postprocess_calls = 0
    proc._yolo_postprocess_total_seconds = 0.0
    proc._yolo_postprocess_max_seconds = 0.0

    proc._record_yolo_postprocess_timing(0.002)
    proc._record_yolo_postprocess_timing(0.006)

    assert proc._yolo_postprocess_stats() == {
        "yolo_postprocess_mode": "vectorized",
        "yolo_postprocess_calls": 2,
        "yolo_postprocess_avg_ms": 4.0,
        "yolo_postprocess_max_ms": 6.0,
    }


def test_deepstream_cumulative_filter_does_not_gate_fall_events(mock_config):
    """낙상은 별도 지속시간 debouncer로 제어하므로 누적 위반 필터 대상에서 제외한다."""
    proc = object.__new__(DeepStreamProcessor)

    proc._init_event_filters(mock_config)

    assert proc.violation_filter.violation_types == {"head"}


class _FakeColor:
    def __init__(self):
        self.value = None

    def set(self, red, green, blue, alpha):
        self.value = (red, green, blue, alpha)


class _FakeDisplayMeta:
    def __init__(self):
        self.num_rects = 0
        self.num_labels = 0
        self.num_lines = 0
        self.num_circles = 0
        self.rect_params = [
            types.SimpleNamespace(
                left=0,
                top=0,
                width=0,
                height=0,
                border_width=0,
                has_bg_color=0,
                border_color=_FakeColor(),
            )
            for _ in range(16)
        ]
        self.text_params = [
            types.SimpleNamespace(
                display_text="",
                x_offset=0,
                y_offset=0,
                font_params=types.SimpleNamespace(
                    font_name="",
                    font_size=0,
                    font_color=_FakeColor(),
                ),
                set_bg_clr=0,
                text_bg_clr=_FakeColor(),
            )
            for _ in range(16)
        ]
        self.line_params = [
            types.SimpleNamespace(
                x1=0,
                y1=0,
                x2=0,
                y2=0,
                line_width=0,
                line_color=_FakeColor(),
            )
            for _ in range(16)
        ]
        self.circle_params = [
            types.SimpleNamespace(
                xc=0,
                yc=0,
                radius=0,
                circle_color=_FakeColor(),
                has_bg_color=0,
                bg_color=_FakeColor(),
            )
            for _ in range(16)
        ]


def test_deepstream_osd_draws_skeleton_for_fall_detection(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_detector = types.SimpleNamespace(min_keypoint_confidence=0.25)
    acquired = []
    frame_meta = types.SimpleNamespace(added_display_meta=[])

    def acquire(_batch_meta):
        meta = _FakeDisplayMeta()
        acquired.append(meta)
        return meta

    fake_pyds = types.SimpleNamespace(
        nvds_acquire_display_meta_from_pool=acquire,
        nvds_add_display_meta_to_frame=(
            lambda frame, meta: frame.added_display_meta.append(meta)
        ),
    )
    monkeypatch.setattr(deepstream_processor, "pyds", fake_pyds)

    keypoints = [[float(idx * 10), float(idx * 5), 0.9] for idx in range(17)]
    detection = {
        "box": (10, 20, 120, 80),
        "label": "person",
        "confidence": 0.88,
        "is_fall": True,
        "keypoints": keypoints,
    }

    proc._add_osd_overlays(object(), frame_meta, [detection])

    assert acquired[0].text_params[0].display_text == "fall_detected 0.88"
    assert any(meta.num_lines > 0 for meta in frame_meta.added_display_meta)
    assert any(meta.num_circles > 0 for meta in frame_meta.added_display_meta)


def test_deepstream_filter_events_respects_one_model_off():
    proc = object.__new__(DeepStreamProcessor)
    proc._camera_ai_flags = {
        "cam1": {
            "use_helmet": False,
            "use_pose": True,
            "use_person": False,
            "use_face": False,
            "use_appearance": False,
        }
    }
    person = DetectionEvent(EventType.PERSON, 0, 0, 10, 10, 0.9, 1.0)
    head = DetectionEvent(EventType.HEAD, 0, 0, 10, 10, 0.8, 1.0)

    assert proc._filter_events_for_camera([person, head], "cam1") == [person]


def test_deepstream_should_enqueue_event_uses_common_debouncer(mock_config):
    proc = object.__new__(DeepStreamProcessor)
    proc._debouncer = EventDebouncer(mock_config, proc._increment_stat)

    event = DetectionEvent(
        EventType.PERSON,
        0,
        0,
        10,
        10,
        0.9,
        1.0,
        object_id=7,
        metadata={"camera_id": "cam1"},
    )

    assert proc._should_enqueue_event(event, "fallback_cam") is True
    assert proc._should_enqueue_event(event, "fallback_cam") is False


def test_deepstream_event_queue_full_counts_event_drop_not_frame_drop():
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=1)
    proc.event_queue.put_nowait({"type": "existing"})
    proc._frames_dropped = 0
    proc._events_dropped = 0

    ok = proc._put_event_dict({"type": "person"}, "cam1")

    assert ok is False
    assert proc._frames_dropped == 0
    assert proc._events_dropped == 1


def test_deepstream_event_queue_success_counts_detected_event():
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=1)
    proc._events_detected = 0

    ok = proc._put_event_dict({"type": "person"}, "cam1")

    assert ok is True
    assert proc._events_detected == 1


def test_deepstream_detection_event_enqueue_uses_common_queue_stats(mock_config):
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=1)
    proc._debouncer = EventDebouncer(mock_config, proc._increment_stat)
    proc._events_detected = 0
    event = DetectionEvent(
        EventType.PERSON,
        0,
        0,
        10,
        10,
        0.9,
        1.0,
        object_id=7,
        metadata={"camera_id": "cam1"},
    )

    ok = proc._enqueue_event(event, "cam1")

    assert ok is True
    assert proc._events_detected == 1
    assert proc.event_queue.get_nowait() is event


def test_deepstream_falldata_aux_submission_only_for_fall_events():
    proc = object.__new__(DeepStreamProcessor)
    proc._falldata_aux_queue = Queue(maxsize=2)
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = True

    person = DetectionEvent(EventType.PERSON, 0, 0, 10, 10, 0.9, 1.0, object_id=1)
    fall = DetectionEvent(EventType.FALL_DETECTED, 1, 2, 30, 20, 0.8, 1.0, object_id=7)

    submitted = proc._submit_falldata_aux_work("cam1", [person])
    assert submitted is None
    assert proc._falldata_aux_queue.empty()

    submitted = proc._submit_falldata_aux_work("cam1", [person, fall])
    camera_name, payload = proc._falldata_aux_queue.get_nowait()

    assert submitted is fall
    assert camera_name == "cam1"
    assert payload["type"] == "fall_detected"
    assert payload["object_id"] == 7


def test_deepstream_falldata_aux_submission_prefers_pending_borderline_fall():
    proc = object.__new__(DeepStreamProcessor)
    proc._falldata_aux_queue = Queue(maxsize=2)
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = True

    clear_fall = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=8,
        metadata={"fall_score": 4.0},
    )
    pending_fall = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={
            "fall_score": 3.0,
            "falldata_aux_publish_pending": True,
        },
    )

    submitted = proc._submit_falldata_aux_work("cam1", [clear_fall, pending_fall])
    _, payload = proc._falldata_aux_queue.get_nowait()

    assert submitted is pending_fall
    assert payload["object_id"] == 7
    assert payload["metadata"]["falldata_aux_publish_pending"] is True


def test_deepstream_falldata_aux_submission_reports_queue_full():
    proc = object.__new__(DeepStreamProcessor)
    proc._falldata_aux_queue = Queue(maxsize=1)
    proc._falldata_aux_queue.put_nowait(("cam1", {"type": "existing"}))
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = True
    fall = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={"falldata_aux_publish_pending": True},
    )

    submitted = proc._submit_falldata_aux_work("cam1", [fall])

    assert submitted is None


def test_deepstream_falldata_aux_submission_ignored_when_disabled():
    proc = object.__new__(DeepStreamProcessor)
    proc._falldata_aux_queue = Queue(maxsize=2)
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = False
    fall = DetectionEvent(EventType.FALL_DETECTED, 1, 2, 30, 20, 0.8, 1.0, object_id=7)

    submitted = proc._submit_falldata_aux_work("cam1", [fall])

    assert submitted is None
    assert proc._falldata_aux_queue.empty()


def test_deepstream_borderline_fall_can_require_aux_before_publish():
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_aux_confirm_borderline = True
    proc._fall_aux_confirm_max_fall_score = None
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = True
    proc._fall_detector = types.SimpleNamespace(score_threshold=3.0)

    borderline = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={"fall_score": 3.0},
    )
    clear_fall = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=8,
        metadata={"fall_score": 4.0},
    )
    person = DetectionEvent(EventType.PERSON, 1, 2, 30, 20, 0.8, 1.0)

    assert proc._should_confirm_fall_with_aux_before_publish(borderline) is True
    assert proc._should_confirm_fall_with_aux_before_publish(clear_fall) is False
    assert proc._should_confirm_fall_with_aux_before_publish(person) is False


def test_deepstream_shadow_aux_never_delays_fall_publish():
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_aux_confirm_borderline = True
    proc._fall_aux_confirm_max_fall_score = 4.5
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = True
    proc._falldata_aux.config = types.SimpleNamespace(mode="shadow")
    proc._fall_detector = types.SimpleNamespace(score_threshold=3.0)
    borderline = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={"fall_score": 3.0},
    )

    assert proc._should_confirm_fall_with_aux_before_publish(borderline) is False


def test_deepstream_fall_aux_confirm_max_score_can_extend_pending_window():
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_aux_confirm_borderline = True
    proc._fall_aux_confirm_max_fall_score = 6.0
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = True
    proc._fall_detector = types.SimpleNamespace(score_threshold=3.0)
    high_score_fall = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={"fall_score": 5.5},
    )
    too_high_score_fall = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=8,
        metadata={"fall_score": 6.5},
    )

    assert proc._should_confirm_fall_with_aux_before_publish(high_score_fall) is True
    assert proc._should_confirm_fall_with_aux_before_publish(too_high_score_fall) is False


def test_deepstream_borderline_fall_is_deferred_to_aux_queue():
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._events_detected = 0
    proc._events_dropped = 0
    proc._falldata_aux_queue = Queue(maxsize=2)
    proc._fall_aux_confirm_borderline = True
    proc._fall_aux_confirm_max_fall_score = 4.5
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = True
    proc._fall_detector = types.SimpleNamespace(score_threshold=3.0)
    proc._debouncer = MagicMock()
    proc._debouncer.should_send.return_value = True
    proc._fall_shadow_review_log_path = Path("/tmp/not-used.jsonl")
    proc._fall_shadow_clip_dir = Path("/tmp")
    proc._fall_shadow_save_clips = False

    borderline = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={"fall_score": 4.5},
    )

    ok = proc._enqueue_event_or_defer_fall_aux(borderline, "cam1")
    camera_name, payload = proc._falldata_aux_queue.get_nowait()

    assert ok is False
    assert proc.event_queue.empty()
    assert camera_name == "cam1"
    assert payload["metadata"]["falldata_aux_publish_pending"] is True


def test_deepstream_repeated_borderline_fall_is_debounced_before_aux_queue():
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._events_detected = 0
    proc._events_dropped = 0
    proc._falldata_aux_queue = Queue(maxsize=2)
    proc._fall_aux_confirm_borderline = True
    proc._fall_aux_confirm_max_fall_score = 4.5
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = True
    proc._fall_detector = types.SimpleNamespace(score_threshold=3.0)
    proc._debouncer = MagicMock()
    proc._debouncer.should_send.side_effect = [True, False]
    proc._fall_shadow_review_log_path = Path("/tmp/not-used.jsonl")
    proc._fall_shadow_clip_dir = Path("/tmp")
    proc._fall_shadow_save_clips = False

    first = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={"fall_score": 4.0},
    )
    repeated = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={"fall_score": 4.0},
    )

    assert proc._enqueue_event_or_defer_fall_aux(first, "cam1") is False
    assert proc._enqueue_event_or_defer_fall_aux(repeated, "cam1") is False
    assert proc._falldata_aux_queue.qsize() == 1
    assert proc._debouncer.should_send.call_count == 2


def test_deepstream_clear_high_score_fall_bypasses_aux_defer(tmp_path):
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._events_detected = 0
    proc._events_dropped = 0
    proc._fall_aux_confirm_borderline = True
    proc._fall_aux_confirm_max_fall_score = 4.5
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.enabled = True
    proc._fall_detector = types.SimpleNamespace(score_threshold=3.0)
    proc._debouncer = MagicMock()
    proc._debouncer.should_send.return_value = True
    proc._fall_shadow_review_log_path = tmp_path / "fall_shadow_review.jsonl"
    proc._fall_shadow_clip_dir = tmp_path / "clips"
    proc._fall_shadow_save_clips = False

    clear_fall = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={"fall_score": 6.0},
    )

    assert proc._enqueue_event_or_defer_fall_aux(clear_fall, "cam1") is True
    assert proc.event_queue.get_nowait() is clear_fall


def test_deepstream_aux_confirmed_borderline_fall_is_enqueued():
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._events_detected = 0
    payload = {
        "type": "fall_detected",
        "object_id": 7,
        "metadata": {
            "fall_score": 3.0,
            "falldata_aux_publish_pending": True,
        },
    }
    result = {
        "status": "ok",
        "confirmed": True,
        "fall_probability": 0.82,
    }

    ok = proc._enqueue_aux_confirmed_fall_event("cam1", payload, result)
    enqueued = proc.event_queue.get_nowait()

    assert ok is True
    assert proc._events_detected == 1
    assert enqueued["metadata"]["falldata_aux"] == result
    assert enqueued["metadata"]["falldata_aux_confirmed"] is True
    assert "falldata_aux_publish_pending" not in enqueued["metadata"]


def test_deepstream_aux_rejected_borderline_fall_is_not_enqueued():
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._events_detected = 0
    payload = {
        "type": "fall_detected",
        "object_id": 7,
        "metadata": {
            "fall_score": 3.0,
            "falldata_aux_publish_pending": True,
        },
    }

    ok = proc._enqueue_aux_confirmed_fall_event(
        "cam1",
        payload,
        {"status": "ok", "confirmed": False},
    )

    assert ok is False
    assert proc.event_queue.empty()
    assert proc._events_detected == 0


def test_deepstream_aux_unavailable_fallback_fall_is_enqueued():
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._events_detected = 0
    payload = {
        "type": "fall_detected",
        "object_id": 7,
        "metadata": {
            "fall_score": 4.0,
            "falldata_aux_publish_pending": True,
        },
    }
    result = {
        "status": "no_frames",
        "confirmed": False,
    }

    ok = proc._enqueue_aux_fallback_fall_event("cam1", payload, result)
    enqueued = proc.event_queue.get_nowait()

    assert ok is True
    assert proc._events_detected == 1
    assert enqueued["metadata"]["falldata_aux"] == result
    assert enqueued["metadata"]["falldata_aux_confirm_fallback"] == "no_frames"
    assert "falldata_aux_publish_pending" not in enqueued["metadata"]


def test_deepstream_compare_veto_can_drop_aux_confirmed_high_score_fall():
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._events_detected = 0
    proc._fall_aux_compare_veto_enabled = True
    proc._fall_aux_compare_veto_min_fall_score = 5.0
    payload = {
        "type": "fall_detected",
        "object_id": 7,
        "metadata": {
            "fall_score": 6.0,
            "falldata_aux_publish_pending": True,
        },
    }
    result = {
        "status": "ok",
        "confirmed": True,
        "compare_model": {
            "status": "ok",
            "confirmed": False,
            "fall_probability": 0.24,
        },
    }

    ok = proc._enqueue_aux_confirmed_fall_event("cam1", payload, result)

    assert ok is False
    assert proc.event_queue.empty()
    assert proc._events_detected == 0


def test_deepstream_compare_veto_ignores_scores_below_minimum():
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._events_detected = 0
    proc._fall_aux_compare_veto_enabled = True
    proc._fall_aux_compare_veto_min_fall_score = 5.0
    payload = {
        "type": "fall_detected",
        "object_id": 7,
        "metadata": {
            "fall_score": 3.0,
            "falldata_aux_publish_pending": True,
        },
    }
    result = {
        "status": "ok",
        "confirmed": True,
        "compare_model": {
            "status": "ok",
            "confirmed": False,
            "fall_probability": 0.24,
        },
    }

    ok = proc._enqueue_aux_confirmed_fall_event("cam1", payload, result)
    enqueued = proc.event_queue.get_nowait()

    assert ok is True
    assert enqueued["metadata"]["falldata_aux"] == result
    assert proc._events_detected == 1


def test_deepstream_pa100k_sgie_backend_name_is_explicit(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._pphuman_label_map = {"model": "Rethinking_of_PAR PA100K resnet50"}
    monkeypatch.setenv("DS_PPHUMAN_INFER_CONFIG", "config/deepstream/config_infer_pa100k.txt")
    monkeypatch.setenv("APPEARANCE_LABEL_MAP_PATH", "config/appearance_pa100k_labels.json")

    assert proc._resolve_pphuman_sgie_backend_name() == "pa100k_sgie"


def test_deepstream_uses_configured_pphuman_infer_config(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    monkeypatch.setenv(
        "DS_PPHUMAN_INFER_CONFIG",
        "config/deepstream/config_infer_pa100k.txt",
    )

    assert proc._resolve_pphuman_infer_config() == Path(
        "config/deepstream/config_infer_pa100k.txt"
    )


def test_deepstream_sgie_injected_person_meta_has_label(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._pose_gie_id = 1

    injected_meta = types.SimpleNamespace(
        unique_component_id=0,
        class_id=-1,
        obj_label="",
        confidence=0.0,
        rect_params=types.SimpleNamespace(left=0.0, top=0.0, width=0.0, height=0.0),
    )
    added = []

    fake_pyds = types.SimpleNamespace(
        NVDSINFER_TENSOR_OUTPUT_META=100,
        NvDsUserMeta=types.SimpleNamespace(cast=lambda value: value),
        NvDsInferTensorMeta=types.SimpleNamespace(cast=lambda value: value),
        nvds_acquire_obj_meta_from_pool=lambda batch_meta: injected_meta,
        nvds_add_obj_meta_to_frame=lambda frame_meta, obj_meta, parent: added.append(obj_meta),
    )
    monkeypatch.setattr(deepstream_processor, "pyds", fake_pyds)
    monkeypatch.setattr(
        proc,
        "_detections_from_tensor",
        lambda tensor_meta, frame_meta: [
            {
                "label": "person",
                "class_id": 0,
                "confidence": 0.91,
                "box": (10, 20, 30, 40),
            }
        ],
    )
    monkeypatch.setattr(proc, "_filter_detections_for_camera", lambda detections, camera: detections)

    tensor_meta = types.SimpleNamespace(unique_id=1)
    user_meta = types.SimpleNamespace(
        base_meta=types.SimpleNamespace(meta_type=100),
        user_meta_data=tensor_meta,
    )
    frame_meta = types.SimpleNamespace(frame_user_meta_list=types.SimpleNamespace(data=user_meta, next=None))

    injected = proc._inject_primary_person_object_meta(object(), frame_meta, "cam1")

    assert injected == 1
    assert added == [injected_meta]
    assert injected_meta.unique_component_id == 1
    assert injected_meta.class_id == 0
    assert injected_meta.obj_label == "person"
    assert injected_meta.rect_params.width == 30


def test_deepstream_sgie_shrinks_oversized_person_roi(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._pose_gie_id = 1
    injected_meta = types.SimpleNamespace(
        unique_component_id=0,
        class_id=-1,
        obj_label="",
        confidence=0.0,
        rect_params=types.SimpleNamespace(left=0.0, top=0.0, width=0.0, height=0.0),
    )
    added = []

    fake_pyds = types.SimpleNamespace(
        NVDSINFER_TENSOR_OUTPUT_META=100,
        NvDsUserMeta=types.SimpleNamespace(cast=lambda value: value),
        NvDsInferTensorMeta=types.SimpleNamespace(cast=lambda value: value),
        nvds_acquire_obj_meta_from_pool=lambda batch_meta: injected_meta,
        nvds_add_obj_meta_to_frame=lambda frame_meta, obj_meta, parent: added.append(obj_meta),
    )
    monkeypatch.setattr(deepstream_processor, "pyds", fake_pyds)
    monkeypatch.setattr(
        proc,
        "_detections_from_tensor",
        lambda tensor_meta, frame_meta: [
            {
                "label": "person",
                "class_id": 0,
                "confidence": 0.91,
                "box": (0, 0, 900, 710),
                "keypoints": [
                    [0, 0, 0.0],
                    [0, 0, 0.0],
                    [0, 0, 0.0],
                    [0, 0, 0.0],
                    [0, 0, 0.0],
                    [500, 150, 0.9],
                    [650, 150, 0.9],
                    [470, 320, 0.8],
                    [680, 320, 0.8],
                    [450, 430, 0.7],
                    [700, 430, 0.7],
                    [520, 500, 0.8],
                    [640, 500, 0.8],
                ],
            }
        ],
    )
    monkeypatch.setattr(proc, "_filter_detections_for_camera", lambda detections, camera: detections)
    monkeypatch.delenv("DS_PPHUMAN_MAX_ROI_WIDTH_RATIO", raising=False)
    monkeypatch.delenv("DS_PPHUMAN_MAX_ROI_HEIGHT_RATIO", raising=False)

    tensor_meta = types.SimpleNamespace(unique_id=1)
    user_meta = types.SimpleNamespace(
        base_meta=types.SimpleNamespace(meta_type=100),
        user_meta_data=tensor_meta,
    )
    frame_meta = types.SimpleNamespace(
        source_frame_width=1280,
        source_frame_height=720,
        frame_user_meta_list=types.SimpleNamespace(data=user_meta, next=None),
    )

    injected = proc._inject_primary_person_object_meta(object(), frame_meta, "cam1")

    assert injected == 1
    assert added == [injected_meta]
    assert injected_meta.rect_params.left > 0
    assert injected_meta.rect_params.width < 900
    assert injected_meta.rect_params.height <= 720


def test_deepstream_fall_shadow_review_record_writes_jsonl(tmp_path):
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_shadow_review_log_path = tmp_path / "logs" / "fall_shadow_review.jsonl"
    proc._fall_shadow_clip_dir = tmp_path / "clips"
    proc._fall_shadow_save_clips = False
    proc._falldata_aux = MagicMock()

    record = proc._write_fall_shadow_review_record(
        "cam 1",
        {
            "type": "fall_detected",
            "object_id": 7,
            "bbox": {"x": 1, "y": 2, "width": 30, "height": 20},
            "confidence": 0.8,
        },
        {"status": "ok", "confirmed": True, "fall_probability": 0.82},
    )

    lines = proc._fall_shadow_review_log_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])

    assert record["review_status"] == "unlabeled"
    assert payload["camera_id"] == "cam 1"
    assert payload["falldata_aux"]["fall_probability"] == 0.82
    assert payload["clip_path"] is None
    assert " " not in payload["event_id"]


def test_deepstream_fall_shadow_review_record_can_save_clip(tmp_path):
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_shadow_review_log_path = tmp_path / "fall_shadow_review.jsonl"
    proc._fall_shadow_clip_dir = tmp_path / "clips"
    proc._fall_shadow_save_clips = True
    proc._falldata_aux = MagicMock()
    proc._falldata_aux.save_buffered_clip.return_value = 12

    record = proc._write_fall_shadow_review_record(
        "cam1",
        {"type": "fall_detected", "object_id": 3},
        {"status": "ok", "confirmed": False, "fall_probability": 0.4},
    )

    proc._falldata_aux.save_buffered_clip.assert_called_once()
    assert record["clip_frames"] == 12
    assert record["clip_path"].endswith(".mp4")


def test_deepstream_enqueue_fall_event_writes_review_record(tmp_path):
    proc = object.__new__(DeepStreamProcessor)
    proc.event_queue = Queue(maxsize=2)
    proc._events_detected = 0
    proc._events_dropped = 0
    proc._fall_shadow_review_log_path = tmp_path / "fall_shadow_review.jsonl"
    proc._fall_shadow_clip_dir = tmp_path / "clips"
    proc._fall_shadow_save_clips = False
    proc._falldata_aux = None
    proc._debouncer = MagicMock()
    proc._debouncer.should_send.return_value = True

    event = DetectionEvent(
        EventType.FALL_DETECTED,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={"fall_score": 4.5, "fall_reasons": ["torso_horizontal:5.0"]},
    )

    assert proc._enqueue_event(event, "cam1") is True

    payload = json.loads(proc._fall_shadow_review_log_path.read_text(encoding="utf-8"))
    assert payload["event_type"] == "fall_detected"
    assert payload["fall_score"] == 4.5
    assert payload["falldata_aux"]["status"] == "not_run"
    assert payload["falldata_aux"]["reason"] == "deepstream_event_only"


def test_deepstream_fall_near_miss_writes_review_record(tmp_path, monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_shadow_review_log_path = tmp_path / "fall_shadow_review.jsonl"
    proc._fall_shadow_clip_dir = tmp_path / "clips"
    proc._fall_shadow_save_clips = False
    proc._fall_shadow_near_miss_enabled = True
    proc._fall_shadow_near_miss_cooldown_sec = 10.0
    proc._fall_shadow_near_miss_last_at = {}
    proc._falldata_aux = None
    monkeypatch.setattr("src.core.deepstream_processor.time.monotonic", lambda: 100.0)

    event = DetectionEvent(
        EventType.PERSON,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={
            "fall_near_miss": {
                "type": "folded_floor_pose",
                "score": 3.0,
                "reasons": ["folded_floor_pose:0.20"],
            }
        },
    )

    proc._write_fall_near_miss_review_records("cam1", [event])
    proc._write_fall_near_miss_review_records("cam1", [event])

    lines = proc._fall_shadow_review_log_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])

    assert len(lines) == 1
    assert payload["event_type"] == "fall_near_miss"
    assert payload["review_source"] == "fall_near_miss"
    assert payload["near_miss"]["type"] == "folded_floor_pose"
    assert payload["falldata_aux"]["status"] == "not_run"


def test_deepstream_event_pipeline_writes_fall_near_miss_record(tmp_path, monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._fall_shadow_review_log_path = tmp_path / "fall_shadow_review.jsonl"
    proc._fall_shadow_clip_dir = tmp_path / "clips"
    proc._fall_shadow_save_clips = False
    proc._fall_shadow_near_miss_enabled = True
    proc._fall_shadow_near_miss_cooldown_sec = 10.0
    proc._fall_shadow_near_miss_last_at = {}
    proc._falldata_aux = None
    proc._assign_synthetic_object_ids = lambda events, camera_name: None
    proc.track_manager = MagicMock()
    proc.violation_filter = MagicMock()
    proc._submit_face_work = lambda camera_name, events: None
    proc.zone_manager = MagicMock()
    proc._enqueue_zone_events = lambda camera_name, events: None
    proc._enqueue_event = lambda event, camera_name: True
    proc._increment_stat = lambda name, delta=1: None
    monkeypatch.setattr("src.core.deepstream_processor.time.monotonic", lambda: 100.0)
    monkeypatch.setattr(
        "src.core.deepstream_processor.ds_apply_existing_event_pipeline",
        lambda **kwargs: None,
    )

    event = DetectionEvent(
        EventType.PERSON,
        1,
        2,
        30,
        20,
        0.8,
        1.0,
        object_id=7,
        metadata={
            "fall_near_miss": {
                "type": "low_score_pose",
                "score": 2.5,
                "reasons": ["torso_horizontal:40.0"],
            }
        },
    )

    proc._apply_existing_event_pipeline("cam1", [event])

    payload = json.loads(proc._fall_shadow_review_log_path.read_text(encoding="utf-8"))
    assert payload["event_type"] == "fall_near_miss"
    assert payload["near_miss"]["type"] == "low_score_pose"


def test_deepstream_publish_loop_success_counts_sent_not_detected(mock_config):
    proc = object.__new__(DeepStreamProcessor)
    proc.running = True
    proc.stop_event = threading.Event()
    proc.event_queue = Queue()
    proc.event_queue.put_nowait({"type": "person", "camera_id": "cam1"})
    proc.config = mock_config
    proc.event_publisher = MagicMock()
    proc._mqtt_publish = MagicMock(return_value=True)
    proc._events_detected = 1
    proc._events_sent = 0
    proc._events_failed = 0

    def stop_after_publish(*args, **kwargs):
        proc.running = False
        return True

    with patch("src.core._event_publish.publish_queue_item", side_effect=stop_after_publish):
        proc._publish_loop()

    assert proc._events_detected == 1
    assert proc._events_sent == 1
    assert proc._events_failed == 0


class _FakeElement:
    def __init__(self, name="element", link_ok=True):
        self.name = name
        self.link_ok = link_ok
        self.linked_to = []
        self.properties = {}
        self.state = None
        self.static_pad = None
        self.synced = False

    def set_property(self, name, value):
        self.properties[name] = value

    def set_state(self, state):
        self.state = state

    def link(self, other):
        self.linked_to.append(other)
        return self.link_ok

    def connect(self, *args):
        self.properties["connect_args"] = args

    def get_static_pad(self, name):
        return self.static_pad if name == "src" else None

    def sync_state_with_parent(self):
        self.synced = True

    def get_name(self):
        return self.name


class _FakePad:
    def __init__(self):
        self.linked = False
        self.linked_to = None

    def is_linked(self):
        return self.linked

    def link(self, sinkpad):
        self.linked = True
        self.linked_to = sinkpad
        return "ok"


class _FakeStreamMux(_FakeElement):
    def __init__(self):
        super().__init__("streammux")
        self.requested = {}
        self.released = []

    def get_request_pad(self, name):
        pad = _FakePad()
        self.requested[name] = pad
        return pad

    def release_request_pad(self, pad):
        self.released.append(pad)


class _FakePipeline:
    def __init__(self, streammux):
        self.streammux = streammux
        self.added = []
        self.removed = []

    def get_by_name(self, name):
        return self.streammux if name == "streammux" else None

    def add(self, element):
        self.added.append(element)

    def remove(self, element):
        self.removed.append(element)


def test_attach_camera_source_to_pipeline_adds_source_and_updates_pad_map(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._cameras = {"cam1": {"source": "rtsp://example/stream", "pad_id": None}}
    proc._pad_to_camera = {}
    proc._preview_camera_id = None
    streammux = _FakeStreamMux()
    pipeline = _FakePipeline(streammux)
    created = []

    def make_element(factory, name):
        element = _FakeElement(name)
        element.static_pad = _FakePad()
        created.append((factory, element))
        return element

    monkeypatch.setattr(proc, "_make_element", make_element)
    monkeypatch.setattr(
        "src.core.deepstream_processor.Gst",
        types.SimpleNamespace(
            PadLinkReturn=types.SimpleNamespace(OK="ok"),
            State=types.SimpleNamespace(NULL="null"),
        ),
    )

    ok = proc._attach_camera_source_to_pipeline(
        "cam1",
        pad_id=2,
        pipeline=pipeline,
        streammux=streammux,
    )

    assert ok is True
    assert created[0][0] == "nvurisrcbin"
    assert pipeline.added == [created[0][1]]
    assert streammux.requested["sink_2"].linked is False
    assert created[0][1].static_pad.linked_to is streammux.requested["sink_2"]
    assert proc._cameras["cam1"]["src_element"] is created[0][1]
    assert proc._cameras["cam1"]["sinkpad"] is streammux.requested["sink_2"]
    assert proc._cameras["cam1"]["pad_id"] == 2
    assert proc._pad_to_camera == {2: "cam1"}
    assert proc._preview_camera_id == "cam1"
    assert created[0][1].synced is True


def test_attach_camera_source_to_pipeline_detaches_existing_source(monkeypatch):
    old_src = _FakeElement("old-src")
    old_sinkpad = _FakePad()
    proc = object.__new__(DeepStreamProcessor)
    proc._cameras = {
        "cam1": {
            "source": "rtsp://example/new",
            "src_element": old_src,
            "sinkpad": old_sinkpad,
            "pad_id": 4,
        }
    }
    proc._pad_to_camera = {4: "cam1"}
    proc._preview_camera_id = "cam1"
    streammux = _FakeStreamMux()
    pipeline = _FakePipeline(streammux)

    def make_element(factory, name):
        element = _FakeElement(name)
        element.static_pad = _FakePad()
        return element

    monkeypatch.setattr(proc, "_make_element", make_element)
    monkeypatch.setattr(
        "src.core.deepstream_processor.Gst",
        types.SimpleNamespace(
            PadLinkReturn=types.SimpleNamespace(OK="ok"),
            State=types.SimpleNamespace(NULL="null"),
        ),
    )

    ok = proc._attach_camera_source_to_pipeline(
        "cam1",
        pipeline=pipeline,
        streammux=streammux,
        detach_existing=True,
    )

    assert ok is True
    assert old_src.state == "null"
    assert pipeline.removed == [old_src]
    assert streammux.released == [old_sinkpad]
    assert proc._cameras["cam1"]["pad_id"] == 4
    assert proc._pad_to_camera == {4: "cam1"}


def test_add_camera_to_pipeline_uses_restart_when_batch_size_would_grow():
    proc = object.__new__(DeepStreamProcessor)
    proc.running = True
    proc._pipeline = object()
    proc._built_source_count = 1
    proc._cameras = {
        "cam1": {"src_element": object(), "pad_id": 0},
        "cam2": {"src_element": None, "pad_id": None},
    }
    proc._attach_camera_source_to_pipeline = MagicMock(return_value=True)

    assert proc._add_camera_to_pipeline("cam2") is False
    proc._attach_camera_source_to_pipeline.assert_not_called()


def test_add_camera_to_pipeline_allows_same_camera_reconnect_with_existing_pad():
    proc = object.__new__(DeepStreamProcessor)
    proc.running = True
    proc._pipeline = object()
    proc._built_source_count = 1
    proc._cameras = {
        "cam1": {"src_element": object(), "pad_id": 0},
    }
    proc._attach_camera_source_to_pipeline = MagicMock(return_value=True)

    assert proc._add_camera_to_pipeline("cam1") is True
    proc._attach_camera_source_to_pipeline.assert_called_once_with(
        "cam1",
        detach_existing=True,
    )


def test_configure_streammux_sets_batch_and_frame_properties(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    streammux = _FakeElement()
    monkeypatch.setenv("DS_STREAM_WIDTH", "1280")
    monkeypatch.setenv("DS_STREAM_HEIGHT", "720")

    proc._configure_streammux(streammux, 3)

    assert streammux.properties["batch-size"] == 3
    assert streammux.properties["width"] == 1280
    assert streammux.properties["height"] == 720
    assert streammux.properties["live-source"] == 1


def test_configure_output_queue_sets_leaky_low_latency_properties():
    queue = _FakeElement()

    DeepStreamProcessor._configure_output_queue(queue)

    assert queue.properties == {
        "leaky": 2,
        "max-size-buffers": 2,
        "max-size-bytes": 0,
        "max-size-time": 0,
    }


def test_create_preview_elements_can_downscale_preview_caps(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    created = []

    def make_element(factory, name):
        element = _FakeElement(name)
        created.append(element)
        return element

    caps_from_string = MagicMock(side_effect=lambda value: value)
    monkeypatch.setattr(proc, "_make_element", make_element)
    monkeypatch.setenv("DS_PREVIEW_WIDTH", "1280")
    monkeypatch.setenv("DS_PREVIEW_HEIGHT", "720")
    monkeypatch.setattr(
        "src.core.deepstream_processor.Gst",
        types.SimpleNamespace(Caps=types.SimpleNamespace(from_string=caps_from_string)),
    )

    elements = proc._create_preview_elements()

    assert elements == created
    caps_from_string.assert_called_once_with("video/x-raw,format=BGRx,width=1280,height=720")
    assert created[0].properties["leaky"] == 2
    assert created[2].properties["caps"] == "video/x-raw,format=BGRx,width=1280,height=720"
    assert created[3].properties["drop"] is True


def test_create_output_elements_can_stream_h264_mpegts(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._output_mode = "h264-mpegts"
    created = []

    def make_element(factory, name):
        element = _FakeElement(name)
        element.factory = factory
        created.append(element)
        return element

    caps_from_string = MagicMock(side_effect=lambda value: value)
    monkeypatch.setattr(proc, "_make_element", make_element)
    monkeypatch.setenv("DS_H264_UDP_HOST", "media")
    monkeypatch.setenv("DS_H264_UDP_PORT", "1234")
    monkeypatch.setenv("DS_H264_WIDTH", "1280")
    monkeypatch.setenv("DS_H264_HEIGHT", "720")
    monkeypatch.setenv("DS_H264_BITRATE", "6000000")
    monkeypatch.setenv("DS_H264_IFRAME_INTERVAL", "30")
    monkeypatch.setattr(
        "src.core.deepstream_processor.Gst",
        types.SimpleNamespace(Caps=types.SimpleNamespace(from_string=caps_from_string)),
    )

    elements = proc._create_output_elements()

    assert [element.factory for element in elements] == [
        "nvvideoconvert",
        "capsfilter",
        "nvv4l2h264enc",
        "h264parse",
        "capsfilter",
        "mpegtsmux",
        "udpsink",
    ]
    assert caps_from_string.call_args_list[0].args == (
        "video/x-raw(memory:NVMM),format=NV12,width=1280,height=720",
    )
    assert caps_from_string.call_args_list[1].args == (
        "video/x-h264,stream-format=byte-stream,alignment=au",
    )
    assert elements[2].properties["bitrate"] == 6000000
    assert elements[2].properties["insert-sps-pps"] is True
    assert elements[2].properties["iframeinterval"] == 30
    assert elements[2].properties["poc-type"] == 2
    assert elements[6].properties["host"] == "media"
    assert elements[6].properties["port"] == 1234
    assert elements[6].properties["sync"] is False


def test_create_output_elements_can_enable_h264_poc_fix(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._output_mode = "h264-mpegts"
    created = []

    def make_element(factory, name):
        element = _FakeElement(name)
        element.factory = factory
        created.append(element)
        return element

    monkeypatch.setattr(proc, "_make_element", make_element)
    monkeypatch.setenv("DS_H264_POC_FIX_ENABLED", "1")
    monkeypatch.setattr(
        "src.core.deepstream_processor.Gst",
        types.SimpleNamespace(
            Caps=types.SimpleNamespace(from_string=lambda value: value),
            CLOCK_TIME_NONE=-1,
        ),
    )

    elements = proc._create_output_elements()

    assert [element.factory for element in elements] == [
        "nvvideoconvert",
        "capsfilter",
        "nvv4l2h264enc",
        "h264parse",
        "capsfilter",
        "identity",
        "mpegtsmux",
        "udpsink",
    ]
    assert elements[2].properties["poc-type"] == 0
    assert elements[5].properties["signal-handoffs"] is True


def test_create_output_elements_can_enable_h264_poc_fix_for_rtsp_publish(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._output_mode = "rtsp-publish"
    created = []

    def make_element(factory, name):
        element = _FakeElement(name)
        element.factory = factory
        created.append(element)
        return element

    monkeypatch.setattr(proc, "_make_element", make_element)
    monkeypatch.setenv("DS_H264_POC_FIX_ENABLED", "1")
    monkeypatch.setattr(
        "src.core.deepstream_processor.Gst",
        types.SimpleNamespace(
            Caps=types.SimpleNamespace(from_string=lambda value: value),
            CLOCK_TIME_NONE=-1,
        ),
    )

    elements = proc._create_output_elements()

    assert [element.factory for element in elements] == [
        "nvvideoconvert",
        "capsfilter",
        "nvv4l2h264enc",
        "h264parse",
        "capsfilter",
        "identity",
        "rtspclientsink",
    ]
    assert elements[2].properties["poc-type"] == 0
    assert elements[5].properties["signal-handoffs"] is True


def test_link_or_raise_raises_when_gstreamer_link_fails():
    first = _FakeElement("first", link_ok=False)
    second = _FakeElement("second")

    with pytest.raises(RuntimeError, match="first -> second link 실패"):
        DeepStreamProcessor._link_or_raise(first, second)


def test_link_preview_branch_links_output_and_preview_paths():
    proc = object.__new__(DeepStreamProcessor)
    osd = _FakeElement("osd")
    tee = _FakeElement("tee")
    output_queue = _FakeElement("output")
    preview = [_FakeElement("preview-1"), _FakeElement("preview-2")]

    previous = proc._link_preview_branch(
        osd=osd,
        tee=tee,
        output_queue=output_queue,
        preview_elements=preview,
    )

    assert previous is output_queue
    assert tee in osd.linked_to
    assert output_queue in tee.linked_to
    assert preview[0] in tee.linked_to
    assert preview[1] in preview[0].linked_to


def test_recognized_face_snapshot_disabled_by_default(tmp_path):
    proc = object.__new__(DeepStreamProcessor)
    proc._face_snapshot_enabled = False
    proc._face_snapshot_dir = tmp_path / "face_snapshots"
    proc._face_snapshot_cooldown_sec = 30.0
    proc._last_face_snapshot_at = {}
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    path = proc._save_recognized_face_snapshot(
        frame, "cam1", "홍길동", {"x": 5, "y": 5, "width": 20, "height": 20}, 0.91, 1000.0
    )

    assert path is None
    assert not proc._face_snapshot_dir.exists()


def test_recognized_face_snapshot_enabled_respects_cooldown(tmp_path):
    pytest.importorskip("cv2")

    proc = object.__new__(DeepStreamProcessor)
    proc._face_snapshot_enabled = True
    proc._face_snapshot_dir = tmp_path / "face_snapshots"
    proc._face_snapshot_cooldown_sec = 30.0
    proc._last_face_snapshot_at = {}
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    bbox = {"x": 5, "y": 5, "width": 20, "height": 20}

    first = proc._save_recognized_face_snapshot(frame, "cam1", "홍길동", bbox, 0.91, 1000.0)
    second = proc._save_recognized_face_snapshot(frame, "cam1", "홍길동", bbox, 0.92, 1010.0)

    assert first is not None
    assert Path(first).exists()
    assert second is None


# ---------------------------------------------------------------------------
# 4. Jetson 환경에서만 실행되는 테스트 (인터페이스 동작 검증)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DEEPSTREAM_AVAILABLE,
    reason="Jetson 환경 필요 — pyds 미설치",
)
def test_deepstream_add_camera(mock_config):
    """add_camera() 가 True 를 반환하고 cameras 에 등록되어야 한다."""
    proc = DeepStreamProcessor(mock_config)
    result = proc.add_camera("cam1", "rtsp://192.168.1.1/stream")
    assert result is True
    assert "cam1" in proc.cameras


@pytest.mark.skipif(
    not DEEPSTREAM_AVAILABLE,
    reason="Jetson 환경 필요 — pyds 미설치",
)
def test_deepstream_add_camera_duplicate(mock_config):
    """동일 camera_id 를 두 번 추가하면 두 번째는 False 를 반환해야 한다."""
    proc = DeepStreamProcessor(mock_config)
    proc.add_camera("cam1", "rtsp://192.168.1.1/stream")
    result = proc.add_camera("cam1", "rtsp://192.168.1.1/stream")
    assert result is False


@pytest.mark.skipif(
    not DEEPSTREAM_AVAILABLE,
    reason="Jetson 환경 필요 — pyds 미설치",
)
def test_deepstream_remove_camera(mock_config):
    """remove_camera() 후 cameras 에서 제거되어야 한다."""
    proc = DeepStreamProcessor(mock_config)
    proc.add_camera("cam1", "rtsp://192.168.1.1/stream")
    proc.remove_camera("cam1")
    assert "cam1" not in proc.cameras


@pytest.mark.skipif(
    not DEEPSTREAM_AVAILABLE,
    reason="Jetson 환경 필요 — pyds 미설치",
)
def test_deepstream_get_stats_returns_dict(mock_config):
    """get_stats() 는 딕셔너리를 반환해야 한다."""
    proc = DeepStreamProcessor(mock_config)
    stats = proc.get_stats()
    assert isinstance(stats, dict)
    assert stats.get("backend") == "deepstream"


@pytest.mark.skipif(
    not DEEPSTREAM_AVAILABLE,
    reason="Jetson 환경 필요 — pyds 미설치",
)
def test_deepstream_get_camera_status(mock_config):
    """get_camera_status() 는 등록된 카메라 정보를 반환해야 한다."""
    proc = DeepStreamProcessor(mock_config)
    proc.add_camera("cam1", "rtsp://192.168.1.1/stream")
    status = proc.get_camera_status()
    assert "cam1" in status
    assert "source" in status["cam1"]


# ---------------------------------------------------------------------------
# 5. create_processor 팩토리 분기
# ---------------------------------------------------------------------------


def test_create_processor_default_returns_video_processor():
    """USE_DEEPSTREAM 가 설정되지 않으면 VideoProcessor 를 반환해야 한다."""
    from src.bootstrap.runtime import create_processor

    cfg = MagicMock()
    cfg.events.queue_max_size = 10
    cfg.mqtt = MagicMock()
    cfg.zone_detection = False
    cfg.collect_dataset = False
    cfg.display = False
    cfg.processing = MagicMock()
    cfg.processing.min_track_frames = 1
    cfg.processing.detection_history_size = 10
    cfg.processing.violation_threshold = 3
    cfg.processing.cumulative_detection_enabled = True
    cfg.processing.consecutive_failure_threshold = 5
    cfg.processing.frame_skip = 1
    cfg.processing.camera_reconnect_delay = 5
    cfg.detection = MagicMock()
    cfg.detection.device = "cpu"
    cfg.detection.target_fps = 10

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("USE_DEEPSTREAM", None)
        proc = create_processor(cfg)

    assert isinstance(proc, VideoProcessor)


def test_create_processor_deepstream_fallback_when_unavailable():
    """USE_DEEPSTREAM=1 이지만 pyds 가 없으면 VideoProcessor 로 폴백해야 한다."""
    from src.bootstrap.runtime import create_processor

    cfg = MagicMock()
    cfg.events.queue_max_size = 10
    cfg.mqtt = MagicMock()
    cfg.zone_detection = False
    cfg.collect_dataset = False
    cfg.display = False
    cfg.processing = MagicMock()
    cfg.processing.min_track_frames = 1
    cfg.processing.detection_history_size = 10
    cfg.processing.violation_threshold = 3
    cfg.processing.cumulative_detection_enabled = True
    cfg.processing.consecutive_failure_threshold = 5
    cfg.processing.frame_skip = 1
    cfg.processing.camera_reconnect_delay = 5
    cfg.detection = MagicMock()
    cfg.detection.device = "cpu"
    cfg.detection.target_fps = 10

    with patch.dict(os.environ, {"USE_DEEPSTREAM": "1"}):
        with patch(
            "src.core.deepstream_processor.DEEPSTREAM_AVAILABLE",
            False,
        ):
            proc = create_processor(cfg)

    assert isinstance(proc, VideoProcessor)
