"""deepstream_processor.py — NVIDIA DeepStream 기반 프로세서 (Jetson 전용).

[실행 환경 요구사항]
  - NVIDIA Jetson (또는 Linux + dGPU) 에서만 동작
  - DeepStream SDK 6.x / 7.x + Python bindings (pyds)
  - GStreamer 1.0 + gst-python (gi.repository.Gst)

[Windows / CPU 전용 환경]
  import 는 성공하지만 인스턴스 생성 시 RuntimeError 를 발생시킵니다.
  테스트는 @pytest.mark.skipif(not DEEPSTREAM_AVAILABLE, ...) 로 건너뜁니다.

[파이프라인 구조]
  nvurisrcbin  →  nvstreammux  →  nvinfer (TensorRT)
               →  nvtracker   →  nvdsosd  →  fakesink
               probe 콜백에서 bbox 메타데이터 추출 → DetectionEvent 생성

[설정 파일 경로]
  config/deepstream/config_infer_primary.txt  — nvinfer TensorRT 설정
  config/deepstream/config_tracker.txt        — nvtracker 설정
  config/deepstream/config_streammux.txt      — nvstreammux 설정
  config/deepstream/labels.txt                — 클래스 레이블

[구현 순서]
  1. _build_pipeline()          : GStreamer 파이프라인 엘리먼트 생성·연결
  2. _on_bus_message()          : EOS / Error 이벤트 처리
  3. _on_pad_probe()            : nvinfer 출력 패드 → DetectionEvent 변환
  4. _publish_loop()            : event_queue → MQTT 발행 스레드
  5. enqueue_camera_retry()     : GStreamer 버스 에러 핸들러 기반 자동 재연결
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Union

from ..config import AppConfig
from ..utils.zone_drawer import ZoneDrawer
from .base_processor import BaseProcessor
from .events import DetectionEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 설정 파일 경로
# ---------------------------------------------------------------------------

_DS_CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "deepstream"
_INFER_CONFIG   = _DS_CONFIG_DIR / "config_infer_primary.txt"
_TRACKER_CONFIG = _DS_CONFIG_DIR / "config_tracker.txt"
_STREAMMUX_CONFIG = _DS_CONFIG_DIR / "config_streammux.txt"
_LABELS_FILE    = _DS_CONFIG_DIR / "labels.txt"

# ---------------------------------------------------------------------------
# DeepStream 가용성 탐지 (런타임 조건부 임포트)
# ---------------------------------------------------------------------------

DEEPSTREAM_AVAILABLE: bool = False

try:
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst  # type: ignore  # noqa: F401
    from gi.repository import GLib  # type: ignore  # noqa: F401

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
        USE_DEEPSTREAM=1 환경변수를 설정하면 runtime.py 에서
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

        # ── 통계 카운터 ───────────────────────────────────────────────
        self._frames_processed: int = 0
        self._frames_dropped: int = 0
        self._events_detected: int = 0

        # ── GStreamer 파이프라인 핸들 ─────────────────────────────────
        self._pipeline: Any = None                   # Gst.Pipeline
        self._main_loop: Any = None                  # GLib.MainLoop
        self._publish_thread: Optional[threading.Thread] = None

        # ── MQTT 발행 콜백 (외부에서 주입) ───────────────────────────
        # set_mqtt_publish_callback(fn) 으로 설정
        # fn(topic: str, payload: dict) 형태
        self._mqtt_publish: Optional[Callable[[str, dict], None]] = None

        logger.info("DeepStreamProcessor 초기화됨 (Jetson 모드)")
        logger.info("설정 디렉터리: %s", _DS_CONFIG_DIR)
        if not _INFER_CONFIG.exists():
            logger.warning(
                "nvinfer 설정 파일이 없습니다: %s\n"
                "config/deepstream/ 디렉터리의 템플릿을 참고하세요.",
                _INFER_CONFIG,
            )

    # ------------------------------------------------------------------
    # 외부 주입 메서드
    # ------------------------------------------------------------------

    def set_mqtt_publish_callback(
        self, callback: Callable[[str, dict], None]
    ) -> None:
        """MQTT 발행 콜백을 설정한다.

        Args:
            callback: fn(topic: str, payload: dict) 형태의 함수.
                      ActionBridge 또는 MqttPublisher 에서 wrap 하여 주입.
        """
        self._mqtt_publish = callback
        logger.info("MQTT 발행 콜백 등록됨")

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

        구현 메모:
          - start() 전에 호출하면 nvstreammux 소스 패드에 정적으로 추가
          - start() 후에 호출하면 nvurisrcbin 을 동적으로 추가(STEP 1)
        """
        if camera_id in self._cameras:
            logger.warning("[%s] 이미 등록된 카메라입니다.", camera_id)
            return False

        self._cameras[camera_id] = {
            "source": source,
            "detections": detections or [],
            "model_paths": model_paths or {},
            "zones_data": zones_data or [],
            "src_element": None,   # Gst.Element — 동적 추가 시 저장
            "pad_id": None,        # nvstreammux 패드 번호
        }
        logger.info("[%s] 카메라 등록됨 (DeepStream): %s", camera_id, source)
        return True

    def remove_camera(self, camera_id: str) -> None:
        """카메라를 파이프라인에서 제거한다.

        구현 메모:
          - nvurisrcbin.set_state(NULL) → unrequest_pad → pipeline 에서 제거
        """
        self._cameras.pop(camera_id, None)
        logger.info("[%s] 카메라 제거됨 (DeepStream)", camera_id)

    def enqueue_camera_retry(
        self,
        camera_id: str,
        source: Union[str, int],
        delay_seconds: float = 30.0,
    ) -> None:
        """카메라 재연결을 지연 스레드로 예약한다.

        구현 메모:
          - GStreamar 버스 ERROR 메시지 → _on_bus_message() 에서 호출
          - threading.Timer(delay_seconds, add_camera, ...) 패턴 사용
        """
        logger.info(
            "[%s] %.0f초 후 재연결 예약 (DeepStream)", camera_id, delay_seconds
        )
        threading.Timer(
            delay_seconds, self._retry_camera, args=(camera_id, source)
        ).start()

    def _retry_camera(self, camera_id: str, source: Union[str, int]) -> None:
        """실제 재연결 실행 (Timer 콜백)."""
        if not self.running:
            return
        logger.info("[%s] 재연결 시도 중...", camera_id)
        self.add_camera(camera_id, source)
        # TODO: _add_camera_to_pipeline(camera_id) 호출 (파이프라인 실행 중 동적 추가)

    def start(self) -> None:
        """DeepStream 파이프라인을 시작한다 (블로킹).

        구현 순서:
          1. Gst.init(None)
          2. self._build_pipeline()
          3. self._pipeline.set_state(Gst.State.PLAYING)
          4. self._publish_thread 시작
          5. self._main_loop.run()  ← 블로킹
        """
        self.running = True
        logger.info("DeepStreamProcessor.start() 호출됨")

        try:
            self._build_pipeline()
            if self._pipeline is None:
                raise RuntimeError("_build_pipeline() 이 pipeline을 설정하지 않았습니다.")

            self._publish_thread = threading.Thread(
                target=self._publish_loop, daemon=True, name="ds-publish"
            )
            self._publish_thread.start()

            # GStreamer 파이프라인 재생 시작
            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("파이프라인을 PLAYING 상태로 전환하는 데 실패했습니다.")

            logger.info("DeepStream 파이프라인 시작됨 — GLib 루프 진입")
            self._main_loop.run()

        except KeyboardInterrupt:
            pass
        except Exception as exc:
            logger.exception("DeepStream 파이프라인 오류: %s", exc)
        finally:
            self.stop()

    def stop(self) -> None:
        """DeepStream 파이프라인을 중지한다."""
        self.running = False
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        if self._main_loop is not None and self._main_loop.is_running():
            self._main_loop.quit()
        logger.info("DeepStreamProcessor 중지됨")

    def get_stats(self) -> Dict:
        """처리 통계를 반환한다."""
        return {
            "frames_processed": self._frames_processed,
            "frames_dropped": self._frames_dropped,
            "events_detected": self._events_detected,
            "backend": "deepstream",
            "cameras": len(self._cameras),
        }

    def get_camera_status(self) -> Dict[str, dict]:
        """카메라별 상태를 반환한다."""
        return {
            camera_id: {
                "connected": self.running,
                "source": info["source"],
                "pad_id": info.get("pad_id"),
            }
            for camera_id, info in self._cameras.items()
        }

    # ------------------------------------------------------------------
    # 내부 파이프라인 구현 메서드 (스켈레톤)
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> None:
        """GStreamer 파이프라인을 조립한다.

        파이프라인 구조:
            [카메라별] nvurisrcbin → nvstreammux
                                     ↓
                                   nvinfer (TensorRT, _INFER_CONFIG)
                                     ↓
                                   nvtracker (_TRACKER_CONFIG)
                                     ↓
                             nvdsosd → fakesink
                             (pad probe → _on_pad_probe)

        [다중 카메라 핵심]
          batch-size 는 카메라 수에 따라 동적으로 설정해야 합니다:
            n_cams = len(self._cameras)
            streammux.set_property("batch-size", n_cams)
            nvinfer.set_property("batch-size", n_cams)

          nvurisrcbin 은 카메라마다 하나씩 생성:
            for pad_id, (cam_id, info) in enumerate(self._cameras.items()):
                src = Gst.ElementFactory.make("nvurisrcbin", f"src-{cam_id}")
                src.set_property("uri", info["source"])  # rtsp://... or file://...
                pipeline.add(src)
                # nvstreammux 에 sink pad 요청 → 연결
                sinkpad = streammux.get_request_pad(f"sink_{pad_id}")
                srcpad  = src.get_static_pad("src")
                srcpad.link(sinkpad)
                self._cameras[cam_id]["pad_id"] = pad_id

          카메라별 pad probe 부착 (카메라 ID 구분용):
            srcpad = nvinfer.get_static_pad("src")
            srcpad.add_probe(Gst.PadProbeType.BUFFER, self._on_pad_probe, None)
            # (pad probe 내부에서 frame_meta.source_id → pad_id → camera_id 역매핑)

        구현 체크리스트:
          [ ] Gst.init(None) 호출 (start() 에서도 가능)
          [ ] Gst.Pipeline.new("cctv-deepstream") 생성
          [ ] nvstreammux 생성 및 batch-size = len(cameras) 설정
          [ ] 카메라별 nvurisrcbin 생성 (uri = rtsp://... or file://...)
          [ ] nvinfer 생성 — config-file-path = str(_INFER_CONFIG)
          [ ] nvtracker 생성 — ll-config-file = str(_TRACKER_CONFIG)
          [ ] nvdsosd 생성
          [ ] fakesink 생성 (또는 실제 출력용 sink)
          [ ] 엘리먼트 파이프라인에 추가 후 link()
          [ ] nvinfer 출력 src_pad 에 _on_pad_probe probe 부착
          [ ] GLib.MainLoop 생성 및 self._main_loop 에 저장
          [ ] 버스 메시지 핸들러 등록: bus.add_signal_watch() + connect("message", _on_bus_message)
        """
        raise NotImplementedError(
            "_build_pipeline() 구현 필요 — "
            "Jetson 환경에서 GStreamer 파이프라인을 조립하세요."
        )

    def _on_bus_message(self, bus: Any, message: Any) -> bool:
        """GLib 메인 루프 버스 메시지 핸들러.

        Args:
            bus: Gst.Bus 인스턴스
            message: Gst.Message 인스턴스

        Returns:
            True 를 반환해야 GLib 이 핸들러를 계속 호출함

        구현 체크리스트:
          [ ] Gst.MessageType.EOS → self._main_loop.quit()
          [ ] Gst.MessageType.ERROR → err, debug 파싱 후 로그 → quit()
          [ ] Gst.MessageType.WARNING → 경고 로그만
          [ ] 카메라별 EOS(소스 종료) → enqueue_camera_retry() 호출
        """
        raise NotImplementedError("_on_bus_message() 구현 필요")

    def _on_pad_probe(
        self, pad: Any, info: Any, camera_id: str
    ) -> Any:  # Gst.PadProbeReturn
        """nvinfer 출력 패드 프로브 콜백 — bbox 메타데이터 → DetectionEvent 변환.

        Args:
            pad:       Gst.Pad (nvinfer src pad)
            info:      Gst.PadProbeInfo
            camera_id: 사용하지 않음 — frame_meta.source_id 로 카메라 구분

        Returns:
            Gst.PadProbeReturn.OK

        [다중 카메라 핵심]
          배치 내 각 프레임은 frame_meta.source_id 로 어느 카메라에서 왔는지 구분:
            frame_meta.source_id  →  nvstreammux 의 pad_id (sink_0, sink_1, ...)
            pad_id → camera_id 역매핑:
              _pad_to_camera = {info["pad_id"]: cam_id for cam_id, info in self._cameras.items()}
              cam_id = _pad_to_camera.get(frame_meta.source_id, "unknown")

        구현 체크리스트:
          [ ] info.get_buffer() → Gst.Buffer
          [ ] pyds.gst_buffer_get_nvds_batch_meta(hash(buffer)) 로 배치 메타 추출
          [ ] frame_meta_list 순회 → NvDsFrameMeta
          [ ] frame_meta.source_id → pad_id → camera_id 역매핑
          [ ] object_meta_list 순회 → NvDsObjectMeta
          [ ] NvDsObjectMeta.rect_params → x, y, width, height (절대 좌표)
          [ ] NvDsObjectMeta.class_id → EventType 매핑
          [ ] NvDsObjectMeta.confidence → DetectionEvent.confidence
          [ ] NvDsObjectMeta.object_id → track ID (nvtracker 결과)
          [ ] NvDsObjectMeta.classifier_meta_list → 헬멧 분류 결과
          [ ] DetectionEvent 생성 후 self.event_queue.put_nowait(event)
          [ ] self._frames_processed += 1
        """
        raise NotImplementedError("_on_pad_probe() 구현 필요")

    def _publish_loop(self) -> None:
        """event_queue 에서 DetectionEvent 를 꺼내 MQTT 로 발행하는 스레드.

        구현 체크리스트:
          [ ] while self.running: event_queue.get(timeout=1.0)
          [ ] MQTT 토픽: f"{topic_prefix}/{camera_id}/{event.event_type.value}"
          [ ] self._mqtt_publish(topic, event.to_dict()) 호출
          [ ] self._events_detected += 1
          [ ] queue.Empty 예외는 continue 로 처리
          [ ] 종료 시 잔여 이벤트 드레인 처리
        """
        logger.info("MQTT 발행 스레드 시작")
        while self.running:
            try:
                event: DetectionEvent = self.event_queue.get(timeout=1.0)
                if self._mqtt_publish is not None:
                    topic = (
                        f"{self._config.mqtt.topic_prefix}"
                        f"/deepstream/{event.event_type.value}"
                    )
                    self._mqtt_publish(topic, event.to_dict())
                    self._events_detected += 1
            except Empty:
                continue
            except Exception as exc:
                logger.error("MQTT 발행 오류: %s", exc)
        logger.info("MQTT 발행 스레드 종료")