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

import os
import threading
import types
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.core.deepstream_processor as deepstream_processor
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
    proc._preview_frame_lock = threading.Lock()
    proc._preview_frames = {"cam1": frame}
    proc._preview_camera_id = None

    copied = proc.get_camera_frame("cam1")
    shared = proc.get_camera_frame("cam1", copy_frame=False)

    assert copied == frame
    assert copied is not frame
    assert shared is frame


def test_deepstream_get_camera_status_uses_common_fields_without_runtime():
    proc = object.__new__(DeepStreamProcessor)
    proc.running = False
    proc._preview_last_frame_at = None
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


def test_deepstream_get_stats_uses_common_fields_without_runtime():
    proc = object.__new__(DeepStreamProcessor)
    proc._frames_processed = 12
    proc._frames_dropped = 1
    proc._events_detected = 3
    proc._events_filtered = 2
    proc._events_failed = 0
    proc._output_mode = "fakesink"
    proc._preview_enabled = True
    proc._preview_max_fps = 5.0
    proc._preview_last_frame_at = None
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


def test_read_preview_max_fps_defaults_to_stream_fps(monkeypatch):
    monkeypatch.delenv("DS_PREVIEW_MAX_FPS", raising=False)
    monkeypatch.setenv("STREAM_FPS", "20")

    assert DeepStreamProcessor._read_preview_max_fps() == 20.0


def test_read_preview_max_fps_clamps_high_values(monkeypatch):
    monkeypatch.setenv("DS_PREVIEW_MAX_FPS", "120")

    assert DeepStreamProcessor._read_preview_max_fps() == 60.0


def test_preview_sample_is_pulled_even_when_throttled(monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    proc._preview_min_interval_sec = 1.0
    proc._preview_last_sample_at = 100.0
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


def test_deepstream_fall_detector_uses_env_thresholds(mock_config, monkeypatch):
    proc = object.__new__(DeepStreamProcessor)
    monkeypatch.setenv("DS_FALL_HEIGHT_RATIO", "0.40")
    monkeypatch.setenv("DS_FALL_ANGLE_HORIZONTAL", "55")
    monkeypatch.setenv("DS_FALL_ANGLE_INVERTED", "125")
    monkeypatch.setenv("DS_FALL_BBOX_ASPECT_RATIO", "1.35")
    monkeypatch.setenv("DS_FALL_SPAN_BBOX_ASPECT_RATIO", "1.20")
    monkeypatch.setenv("DS_FALL_KEYPOINT_SPAN_RATIO", "0.55")
    monkeypatch.setenv("DS_FALL_MIN_KEYPOINT_CONFIDENCE", "0.25")
    monkeypatch.setenv("DS_FALL_MIN_HIP_CONFIDENCE", "0.25")

    proc._init_event_filters(mock_config)

    assert proc._fall_detector.fall_height_ratio == 0.40
    assert proc._fall_detector.angle_horizontal == 55.0
    assert proc._fall_detector.angle_inverted == 125.0
    assert proc._fall_detector.bbox_aspect_ratio == 1.35
    assert proc._fall_detector.span_bbox_aspect_ratio == 1.20
    assert proc._fall_detector.span_ratio == 0.55
    assert proc._fall_detector.min_keypoint_confidence == 0.25
    assert proc._fall_detector.min_hip_confidence == 0.25


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

    with patch("src.core.deepstream_processor.publish_queue_item", side_effect=stop_after_publish):
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

    def set_property(self, name, value):
        self.properties[name] = value

    def link(self, other):
        self.linked_to.append(other)
        return self.link_ok

    def connect(self, *args):
        self.properties["connect_args"] = args

    def get_name(self):
        return self.name


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
        "identity",
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
    assert elements[5].properties["signal-handoffs"] is True
    assert elements[7].properties["host"] == "media"
    assert elements[7].properties["port"] == 1234
    assert elements[7].properties["sync"] is False


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
