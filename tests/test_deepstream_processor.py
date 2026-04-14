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
from unittest.mock import MagicMock, patch

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
    return cfg


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
