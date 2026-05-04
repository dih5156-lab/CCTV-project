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
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.base_processor import BaseProcessor
from src.core.deepstream_processor import DEEPSTREAM_AVAILABLE, DeepStreamProcessor
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


def test_build_source_entries_skips_integer_sources_without_runtime():
    proc = object.__new__(DeepStreamProcessor)
    proc._cameras = {
        "cam1": {"source": "rtsp://192.168.1.1/stream"},
        "cam2": {"source": 0},
    }

    entries = proc._build_source_entries()

    assert len(entries) == 1
    assert entries[0][0] == 0
    assert entries[0][1] == "cam1"
    assert entries[0][3] == "rtsp://192.168.1.1/stream"


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
