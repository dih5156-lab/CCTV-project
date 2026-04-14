"""deepstream_processor.py — NVIDIA DeepStream 기반 프로세서 (Jetson 전용).

[실행 환경 요구사항]
  - NVIDIA Jetson (또는 Linux + dGPU) 에서만 동작
  - DeepStream SDK 6.x / 7.x + Python bindings (pyds)
  - GStreamer 1.0 + gst-python (gi.repository.Gst)

[Windows / CPU 전용 환경]
  import 는 성공하지만 인스턴스 생성 시 RuntimeError 를 발생시킵니다.
  테스트는 @pytest.mark.skipif(not DEEPSTREAM_AVAILABLE, ...) 로 건너뜁니다.

[파이프라인 구조 (구현 예정)]
  nvurisrcbin  →  nvstreammux  →  nvinfer (TensorRT)
               →  nvtracker
               →  nvdsosd  →  fakesink
               probe 콜백에서 bbox 메타데이터 추출 → DetectionEvent 생성

[TODO: Jetson 환경 확보 후 구현할 항목]
  - _build_pipeline()          : GStreamer 파이프라인 조립
  - _on_pad_probe()            : pyds 메타데이터 → DetectionEvent 매핑
  - _publish_loop()            : event_queue → MQTT 발행
  - TensorRT .engine 경로 설정 : config_infer_primary.txt
  - nvtracker 파라미터 조정     : config_tracker.txt
"""

from __future__ import annotations

import logging
import os
from queue import Queue
from typing import Any, Dict, List, Optional, Union

from ..config import AppConfig
from ..utils.zone_drawer import ZoneDrawer
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepStream 가용성 탐지 (런타임 조건부 임포트)
# ---------------------------------------------------------------------------

DEEPSTREAM_AVAILABLE: bool = False

try:
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst  # type: ignore  # noqa: F401

    import pyds  # type: ignore  # noqa: F401

    DEEPSTREAM_AVAILABLE = True
    logger.debug("DeepStream Python bindings (pyds) 로드 성공")
except ImportError:
    logger.debug(
        "DeepStream Python bindings (pyds / gi) 를 찾을 수 없습니다. "
        "DeepStreamProcessor 는 이 환경에서 사용할 수 없습니다."
    )


# ---------------------------------------------------------------------------
# DeepStreamProcessor
# ---------------------------------------------------------------------------


class DeepStreamProcessor(BaseProcessor):
    """NVIDIA DeepStream SDK 기반 다중 카메라 처리 파이프라인.

    Jetson Orin (또는 Linux + dGPU) 에서만 동작합니다.
    VideoProcessor 와 동일한 BaseProcessor 인터페이스를 구현하므로
    runtime.py 의 팩토리 함수를 통해 투명하게 교체됩니다.

    사용 방법:
        USE_DEEPSTREAM=1 환경 변수를 설정하면 runtime.py 에서
        VideoProcessor 대신 이 클래스를 자동으로 선택합니다.
    """

    def __init__(self, config: AppConfig) -> None:
        if not DEEPSTREAM_AVAILABLE:
            raise RuntimeError(
                "DeepStreamProcessor 는 NVIDIA DeepStream SDK 와 pyds 바인딩이 "
                "설치된 환경(Jetson / Linux+GPU)에서만 실행할 수 있습니다.\n"
                "현재 환경에서는 USE_DEEPSTREAM=0 을 설정하거나 "
                "VideoProcessor 를 사용하세요."
            )

        super().__init__(config)

        # ── 상태 ─────────────────────────────────────────────────────
        self.running: bool = False
        self._cameras: Dict[str, Dict] = {}          # camera_id → 소스 정보
        self.event_queue: Queue = Queue(
            maxsize=config.events.queue_max_size * 3
        )

        # ── GStreamer 파이프라인 핸들 (구현 예정) ─────────────────────
        self._pipeline: Any = None                   # Gst.Pipeline
        self._main_loop: Any = None                  # GLib.MainLoop

        logger.info("DeepStreamProcessor 초기화됨 (Jetson 모드)")

    # ------------------------------------------------------------------
    # 필수 인터페이스 구현
    # ------------------------------------------------------------------

    @property
    def cameras(self) -> Dict:
        return dict(self._cameras)

    def add_camera(
        self,
        camera_id: str,
        source: Union[str, int],
        *,
        detections: Optional[List[str]] = None,
        model_paths: Optional[Dict[str, str]] = None,
        zones_data: Optional[List[Dict]] = None,
    ) -> bool:
        """카메라 소스를 등록한다.

        TODO: Jetson 환경에서 nvurisrcbin 동적 추가 구현
        """
        if camera_id in self._cameras:
            logger.warning("[%s] 이미 등록된 카메라입니다.", camera_id)
            return False

        self._cameras[camera_id] = {
            "source": source,
            "detections": detections,
            "model_paths": model_paths,
            "zones_data": zones_data,
        }
        logger.info("[%s] 카메라 등록됨 (DeepStream): %s", camera_id, source)
        return True

    def remove_camera(self, camera_id: str) -> None:
        """카메라를 파이프라인에서 제거한다.

        TODO: Jetson 환경에서 nvurisrcbin 동적 제거 구현
        """
        self._cameras.pop(camera_id, None)
        logger.info("[%s] 카메라 제거됨 (DeepStream)", camera_id)

    def enqueue_camera_retry(
        self,
        camera_id: str,
        source: Union[str, int],
        delay_seconds: float = 30.0,
    ) -> None:
        """카메라 재연결을 예약한다.

        TODO: GStreamer 버스 에러 핸들러 기반 자동 재연결 구현
        """
        logger.info(
            "[%s] 재연결 예약 — %.0f초 후 시도 (DeepStream)",
            camera_id,
            delay_seconds,
        )

    def start(self) -> None:
        """DeepStream 파이프라인을 시작한다.

        TODO: _build_pipeline() 호출 → pipeline.set_state(Gst.State.PLAYING)
              GLib.MainLoop 실행 (블로킹)
        """
        self.running = True
        logger.info("DeepStreamProcessor.start() — 파이프라인 구현 예정")
        # 구현 완료 전까지 블로킹 유지 (KeyboardInterrupt 로 종료)
        try:
            import time

            while self.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """DeepStream 파이프라인을 중지한다.

        TODO: pipeline.set_state(Gst.State.NULL) → main_loop.quit()
        """
        self.running = False
        if self._pipeline is not None:
            # Gst.State.NULL 전환은 Jetson 구현 시 추가
            pass
        logger.info("DeepStreamProcessor.stop() 호출됨")

    def get_stats(self) -> Dict:
        """처리 통계를 반환한다.

        TODO: nvdsanalytics 메타데이터 기반 통계 수집 구현
        """
        return {
            "frames_processed": 0,
            "frames_dropped": 0,
            "events_detected": 0,
            "backend": "deepstream",
        }

    def get_camera_status(self) -> Dict[str, dict]:
        """카메라별 상태를 반환한다."""
        return {
            camera_id: {
                "connected": self.running,
                "source": info["source"],
            }
            for camera_id, info in self._cameras.items()
        }
