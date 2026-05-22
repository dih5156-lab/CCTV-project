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
import json
import tempfile
import threading
import time
import ctypes
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from ..config import AppConfig
from ..protocols.mqtt_publisher import MqttEventPublisher
from ..services.appearance_conditions import AppearanceConditionStore
from ..utils.zone_detection import ZoneEvent, ZoneManager
from ..utils.zone_drawer import ZoneDrawer
from .ai._fall_detector import FallDetector
from .base_processor import BaseProcessor
from ._deepstream_event_factory import detections_to_events, object_meta_to_event
from ._deepstream_face_context import (
    remove_camera_face_cache,
    run_deepstream_face_recognition,
)
from ._event_context import events_to_nearby_objects
from ._event_publish import publish_queue_item
from ._face_snapshot import save_recognized_face_snapshot
from .event_filters import CumulativeViolationFilter, TrackManager
from .events import DetectionEvent, EventType
from ._synthetic_object_ids import SyntheticObjectIdAssigner, event_iou
from ._yolo_postprocess import (
    detections_from_yolo_output,
    map_yolo_box_to_frame,
    map_yolo_keypoints_to_frame,
    nms_detections,
)
from .ai._attribute_backends import decode_pphuman_scores
from .ai._appearance_analyzer import AppearanceAnalyzer, BAG_CLASSES
from .ai._appearance_pipeline import AppearancePipeline
from ..utils.face_recognition import FaceRecognitionEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 설정 파일 경로
# ---------------------------------------------------------------------------

_DS_CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "deepstream"
_INFER_CONFIG   = _DS_CONFIG_DIR / "config_infer_primary.txt"
_HELMET_INFER_CONFIG = _DS_CONFIG_DIR / "config_infer_helmet.txt"
_PPHUMAN_INFER_CONFIG = _DS_CONFIG_DIR / "config_infer_pphuman.txt"
_TRACKER_CONFIG = _DS_CONFIG_DIR / "config_tracker.txt"
_STREAMMUX_CONFIG = _DS_CONFIG_DIR / "config_streammux.txt"
_LABELS_FILE    = _DS_CONFIG_DIR / "labels.txt"
_HELMET_LABELS_FILE = _DS_CONFIG_DIR / "labels_helmet.txt"
_TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

# ---------------------------------------------------------------------------
# DeepStream 가용성 탐지 (런타임 조건부 임포트)
# ---------------------------------------------------------------------------

DEEPSTREAM_AVAILABLE: bool = False
Gst = None  # type: ignore[assignment]
GLib = None  # type: ignore[assignment]
pyds = None  # type: ignore[assignment]

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
# _H264PocFixer
# ---------------------------------------------------------------------------


class _H264PocFixer:
    """nvv4l2h264enc가 출력하는 H264 비트스트림의 poc_lsb 값을 순차적으로 수정한다.

    nvv4l2h264enc는 num-B-Frames=0, Baseline 프로파일이어도 B-frame 스타일의
    poc_lsb 값(IDR 이후 첫 P-frame = 2*(GOP-1))을 슬라이스 헤더에 기록한다.
    MediaMTX v1.18.2의 DTS 추출기는 이를 B-frame 재정렬로 오해하여
    "too many reordered frames" 오류를 낸다.

    이 클래스는 GStreamer identity 요소의 handoff 시그널에 연결하여 각 H264
    버퍼에서 슬라이스 헤더의 poc_lsb 필드를 순차값(0, 2, 4, ...)으로 덮어 쓴다.
    그러면 DTS 추출기가 pocDiff=−1 → DTS=PTS 로 올바르게 계산한다.
    """

    def __init__(self) -> None:
        self._log2_max_frame_num: int = 8   # SPS의 log2_max_frame_num_minus4+4
        self._poc_lsb_bits: int = 8         # SPS의 log2_max_pic_order_cnt_lsb_minus4+4
        self._poc_counter: int = 0          # 다음 non-IDR 프레임에 사용할 poc 값
        self._lock = threading.Lock()
    # ------------------------------------------------------------------
    # 비트 I/O 헬퍼
    # ------------------------------------------------------------------
    @staticmethod
    def _get_bit(data: bytes, pos: int) -> int:
        return (data[pos >> 3] >> (7 - (pos & 7))) & 1

    @staticmethod
    def _set_bit(data: bytearray, pos: int, bit: int) -> None:
        idx = pos >> 3
        shift = 7 - (pos & 7)
        if bit:
            data[idx] |= 1 << shift
        else:
            data[idx] &= ~(1 << shift)

    @classmethod
    def _read_ue(cls, data: bytes, pos: list) -> int:
        """Exp-Golomb ue(v) 읽기."""
        m = 0
        while pos[0] < len(data) * 8 and not cls._get_bit(data, pos[0]):
            m += 1
            pos[0] += 1
        pos[0] += 1  # stop bit
        val = (1 << m) - 1
        for i in range(m - 1, -1, -1):
            val += cls._get_bit(data, pos[0]) << i
            pos[0] += 1
        return val

    @classmethod
    def _read_u(cls, data: bytes, pos: list, n: int) -> int:
        """고정 n 비트 부호 없는 정수 읽기."""
        val = 0
        for _ in range(n):
            val = (val << 1) | cls._get_bit(data, pos[0])
            pos[0] += 1
        return val

    @classmethod
    def _write_u(cls, data: bytearray, bit_pos: int, n: int, val: int) -> None:
        """고정 n 비트 부호 없는 정수 쓰기."""
        for i in range(n - 1, -1, -1):
            cls._set_bit(data, bit_pos, (val >> i) & 1)
            bit_pos += 1

    # ------------------------------------------------------------------
    # SPS 파싱
    # ------------------------------------------------------------------
    def _parse_sps(self, nalu: bytes) -> None:
        """SPS NAL 유닛에서 log2_max_frame_num 및 poc_lsb_bits 추출."""
        try:
            # NAL 헤더(1B) + profile_idc(1B) + constraint_flags(1B) + level_idc(1B) 건너뜀
            body = nalu[4:]
            pos = [0]
            self._read_ue(body, pos)                          # seq_parameter_set_id
            self._log2_max_frame_num = self._read_ue(body, pos) + 4  # log2_max_frame_num_minus4
            poc_type = self._read_ue(body, pos)              # pic_order_cnt_type

            if poc_type == 0:
                self._poc_lsb_bits = self._read_ue(body, pos) + 4  # log2_max_poc_lsb_minus4
            else:
                self._poc_lsb_bits = 4  # poc_type=2: 필드 없음, 기본값 사용
        except Exception:
            pass  # 파싱 실패 시 기존 기본값 유지

    # ------------------------------------------------------------------
    # 슬라이스 헤더에서 poc_lsb 위치 탐색
    # ------------------------------------------------------------------
    def _poc_lsb_bit_pos(self, nalu: bytes, is_idr: bool):
        """슬라이스 NALU RBSP에서 poc_lsb 필드 시작 비트 위치 반환.

        반환값: (nalu 내 body 시작 비트 오프셋, body bytes) or (None, None)
        """
        try:
            body = nalu[1:]          # NAL 헤더(1B) 제거
            pos = [0]
            self._read_ue(body, pos)                      # first_mb_in_slice
            self._read_ue(body, pos)                      # slice_type
            self._read_ue(body, pos)                      # pic_parameter_set_id
            self._read_u(body, pos, self._log2_max_frame_num)  # frame_num
            if is_idr:
                self._read_ue(body, pos)                  # idr_pic_id
            return pos[0], body
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # NAL 유닛 분리
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_nals(data: bytes):
        """Annex B 바이트 스트림에서 (start_offset, end_offset) 쌍을 생성.

        start_offset은 시작 코드 **다음** 바이트(NAL 헤더 위치).
        end_offset은 다음 시작 코드 직전.
        """
        starts = []
        i = 0
        n = len(data)
        while i < n - 2:
            if i + 3 < n and data[i:i + 4] == b"\x00\x00\x00\x01":
                starts.append(i + 4)
                i += 4
            elif data[i:i + 3] == b"\x00\x00\x01":
                starts.append(i + 3)
                i += 3
            else:
                i += 1
        for j, s in enumerate(starts):
            # 다음 시작 코드 이전까지가 현재 NAL의 끝
            if j + 1 < len(starts):
                e = starts[j + 1]
                # 시작 코드 길이 빼기 (4바이트 또는 3바이트)
                e -= 4 if (e >= 4 and data[e - 4:e] == b"\x00\x00\x00\x01") else 3
            else:
                e = n
            yield s, e

    # ------------------------------------------------------------------
    # 버퍼 처리 (인플레이스)
    # ------------------------------------------------------------------
    def process_buffer(self, data: bytearray) -> None:
        """H264 버퍼 전체를 스캔하여 슬라이스 헤더의 poc_lsb를 순차값으로 수정."""
        raw = bytes(data)
        for s, e in self._iter_nals(raw):
            if s >= e or s >= len(raw):
                continue
            nal_type = raw[s] & 0x1F
            nalu = raw[s:e]

            if nal_type == 7:          # SPS
                self._parse_sps(nalu)

            elif nal_type == 5:        # IDR 슬라이스
                with self._lock:
                    self._poc_counter = 2          # 다음 non-IDR 프레임은 poc=2 부터 시작

                # IDR의 poc_lsb는 0 이어야 함 — 검증만, 수정 불필요

            elif nal_type == 1:        # Non-IDR 슬라이스
                with self._lock:
                    target_poc = self._poc_counter
                    self._poc_counter = (self._poc_counter + 2) % (1 << self._poc_lsb_bits)

                bit_pos, body = self._poc_lsb_bit_pos(nalu, False)
                if bit_pos is not None:
                    # body는 nalu[1:] — data 내 절대 비트 위치 계산
                    data_bit_pos = (s + 1) * 8 + bit_pos
                    self._write_u(data, data_bit_pos, self._poc_lsb_bits, target_poc)



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
        logging.getLogger("src.core.event_filters").setLevel(logging.WARNING)

        self._init_runtime_state(config)
        self._init_yolo_settings()
        self._init_event_filters(config)
        self._init_ai_context(config)
        self._init_pipeline_handles()
        self._init_event_publisher(config)

        logger.info("DeepStreamProcessor 초기화됨 (Jetson 모드)")
        logger.info(
            "DeepStream 이벤트 최소 발행 간격: %.2f초",
            self._event_min_interval_seconds,
        )
        logger.info("설정 디렉터리: %s", _DS_CONFIG_DIR)
        if not _INFER_CONFIG.exists():
            logger.warning(
                "nvinfer 설정 파일이 없습니다: %s\n"
                "config/deepstream/ 디렉터리의 템플릿을 참고하세요.",
                _INFER_CONFIG,
            )
        if self._helmet_enabled and not _HELMET_INFER_CONFIG.exists():
            logger.warning("helmet/head nvinfer 설정 파일이 없습니다: %s", _HELMET_INFER_CONFIG)

    def _init_runtime_state(self, config: AppConfig) -> None:
        """런타임 상태, preview, 통계 카운터를 초기화한다."""
        self.running: bool = False
        self.stop_event = Event()
        self._output_mode = os.environ.get("DS_OUTPUT_MODE", "fakesink").strip().lower()
        self._tensor_probe_warned = False
        self._preview_enabled = self._env_bool("DS_PREVIEW_ENABLED", True)
        self._preview_camera_id: Optional[str] = None
        self._preview_frame_lock = threading.Lock()
        self._preview_frames: Dict[str, Any] = {}
        self._preview_last_frame_at: Optional[float] = None
        self._preview_last_sample_at = 0.0
        self._preview_max_fps = self._read_preview_max_fps()
        self._preview_min_interval_sec = (
            1.0 / self._preview_max_fps if self._preview_max_fps > 0 else 0.0
        )
        self._pipeline_restart_lock = threading.Lock()
        self._cameras_json_lock = threading.Lock()   # cameras.json R/W 직렬화
        # 얼굴/외형 인식 비동기 워커 (GStreamer 메인루프 블로킹 방지)
        self._face_work_queue: Queue = Queue(maxsize=8)
        self._face_worker_thread: Optional[threading.Thread] = None
        # 실제 파이프라인에 빌드된 토폴로지 (primary, helmet, pphuman)
        # _build_pipeline() 호출 시 갱신됨. 재시작 필요 여부 판단에 사용.
        self._built_topology: Tuple[bool, bool, bool] = (False, False, False)
        self._pipeline_restart_pending: bool = False  # API 응답에 재시작 여부 전달용
        self._helmet_enabled = self._env_bool("DS_HELMET_ENABLED", True)
        self._pphuman_sgie_enabled = self._env_bool("DS_PPHUMAN_SGIE_ENABLED", True)
        self._face_enabled_default = self._env_bool("DS_FACE_ENABLED", False)
        self._appearance_enabled_default = self._env_bool(
            "DS_APPEARANCE_ENABLED",
            bool(config.appearance.enabled),
        )
        self._event_min_interval_seconds = float(
            os.environ.get(
                "DS_EVENT_MIN_INTERVAL_SEC",
                str(config.events.debounce_seconds if config.events.debounce_enabled else 0.0),
            )
        )
        self._last_event_emit_at: Dict[Tuple[str, str, int, Optional[int]], float] = {}
        self._cameras: Dict[str, Dict] = {}
        self._camera_ai_flags: Dict[str, Dict[str, bool]] = {}
        self._pad_to_camera: Dict[int, str] = {}  # pad_id → camera_id 캐시 (매 프레임 재생성 방지)
        self.event_queue: Queue = Queue(maxsize=config.events.queue_max_size * 3)
        self._frames_processed = 0
        self._frames_dropped = 0
        self._events_detected = 0
        self._events_filtered = 0
        self._events_failed = 0

    def _init_yolo_settings(self) -> None:
        """DeepStream nvinfer tensor 후처리 설정을 초기화한다."""
        self._pose_gie_id = int(os.environ.get("DS_POSE_GIE_ID", "1"))
        self._helmet_gie_id = int(os.environ.get("DS_HELMET_GIE_ID", "2"))
        self._pphuman_gie_id = int(os.environ.get("DS_PPHUMAN_GIE_ID", "3"))
        self._yolo_task = os.environ.get("DS_YOLO_TASK", "detect").strip().lower()
        self._yolo_conf_threshold = float(os.environ.get("DS_YOLO_CONFIDENCE", "0.35"))
        self._yolo_iou_threshold = float(os.environ.get("DS_YOLO_IOU_THRESHOLD", "0.45"))
        self._yolo_max_detections = int(os.environ.get("DS_YOLO_MAX_DETECTIONS", "100"))
        self._yolo_class_ids = self._parse_class_ids("DS_YOLO_CLASS_IDS", {0})
        self._yolo_labels = self._load_yolo_labels(_LABELS_FILE, "DS_YOLO_LABELS")
        self._task_by_gie = {
            self._pose_gie_id: self._yolo_task,
            self._helmet_gie_id: "detect",
        }
        self._labels_by_gie = {
            self._pose_gie_id: self._yolo_labels,
            self._helmet_gie_id: self._load_yolo_labels(
                _HELMET_LABELS_FILE,
                "DS_HELMET_LABELS",
                fallback=["helmet", "head"],
            ),
        }
        self._class_ids_by_gie = {
            self._pose_gie_id: self._yolo_class_ids,
            self._helmet_gie_id: self._parse_class_ids("DS_HELMET_CLASS_IDS", {0, 1}),
        }
        self._confidence_by_gie = {
            self._pose_gie_id: float(os.environ.get("DS_POSE_CONFIDENCE", str(self._yolo_conf_threshold))),
            self._helmet_gie_id: float(os.environ.get("DS_HELMET_CONFIDENCE", "0.35")),
        }

    def _init_event_filters(self, config: AppConfig) -> None:
        """기존 VideoProcessor 후처리 필터를 재사용하도록 초기화한다."""
        self._synthetic_track_iou = float(os.environ.get("DS_SYNTHETIC_TRACK_IOU", "0.30"))
        self._synthetic_track_timeout = float(os.environ.get("DS_SYNTHETIC_TRACK_TIMEOUT", "1.00"))
        self._synthetic_id_assigner = SyntheticObjectIdAssigner(
            track_iou=self._synthetic_track_iou,
            track_timeout=self._synthetic_track_timeout,
        )
        self._fall_detector = FallDetector(config.detection.fall_height_ratio)
        self.track_manager = TrackManager(
            track_timeout=self._synthetic_track_timeout,
            track_iou_threshold=float(os.environ.get("DS_TRACK_IOU_THRESHOLD", "0.50")),
            min_track_frames=int(os.environ.get(
                "DS_MIN_TRACK_FRAMES",
                str(config.processing.min_track_frames),
            )),
        )
        self.violation_filter = CumulativeViolationFilter(
            history_max_size=config.processing.detection_history_size,
            violation_threshold=config.processing.violation_threshold,
            enabled=config.processing.cumulative_detection_enabled,
        )

    def _init_ai_context(self, config: AppConfig) -> None:
        """얼굴/외형/구역 후처리 컨텍스트를 초기화한다."""
        self._face_snapshot_enabled = self._env_bool("FACE_SNAPSHOT_ENABLED", False)
        self._face_snapshot_dir = Path(os.environ.get("FACE_SNAPSHOT_DIR", "data/face_snapshots"))
        self._face_snapshot_cooldown_sec = float(os.environ.get("FACE_SNAPSHOT_COOLDOWN_SEC", "30.0"))
        self._last_face_snapshot_at: Dict[Tuple[str, str], float] = {}
        self.face_recognizer = FaceRecognitionEngine(
            device=os.environ.get("FACE_DEVICE", config.detection.device)
        )
        self._face_identity_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._appearance = AppearanceAnalyzer(
            backend_name=config.appearance.backend,
            backend_model_path=config.appearance.model_path,
            backend_label_map_path=config.appearance.label_map_path,
            backend_runtime=config.appearance.runtime,
            backend_device=os.environ.get("APPEARANCE_DEVICE", "cpu"),
            backend_input_size=config.appearance.input_size,
            backend_score_threshold=config.appearance.score_threshold,
            bbox_expand_ratio=config.appearance.bbox_expand_ratio,
        )
        self._appearance_pipeline = AppearancePipeline(
            self._appearance,
            Path(os.environ.get("APPEARANCE_CROP_DIR", "data/appearance_crops")),
            save_crops=self._env_bool("APPEARANCE_SAVE_CROPS", False),
        )
        self._appearance_db_path = Path(os.environ.get("APPEARANCES_DB", "/app/data/appearances.db"))
        self._appearance_conditions_mtime: Optional[float] = None
        self._appearance_conditions_checked_at = 0.0
        self._appearance_conditions_refresh_sec = float(
            os.environ.get("DS_APPEARANCE_CONDITION_REFRESH_SEC", "10.0")
        )
        self._appearance_capability_logged: set[str] = set()
        self._pphuman_label_map = self._load_pphuman_label_map(config.appearance.label_map_path)
        self.zone_manager: Optional[ZoneManager] = None
        if config.zone_detection:
            try:
                self.zone_manager = ZoneManager(config.zones_config)
            except Exception as exc:
                logger.warning("DeepStream ZoneManager 초기화 실패: %s", exc)

    def _init_pipeline_handles(self) -> None:
        self._pipeline: Any = None
        self._main_loop: Any = None
        self._publish_thread: Optional[threading.Thread] = None
        self._main_loop_thread: Optional[threading.Thread] = None
        self._mqtt_publish: Optional[Callable[[str, dict], None]] = None

    def _init_event_publisher(self, config: AppConfig) -> None:
        self.event_publisher = MqttEventPublisher(
            broker=config.mqtt.broker,
            port=config.mqtt.port,
            topic_prefix=config.mqtt.topic_prefix,
            client_id_prefix=f"{config.mqtt.client_id_prefix}-deepstream",
            qos=config.mqtt.qos,
            retain=config.mqtt.retain,
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

    @staticmethod
    def _normalize_model_flags(flags: Dict[str, object]) -> Dict[str, bool]:
        use_pose = bool(flags.get("use_pose", flags.get("pose", False)))
        use_helmet = bool(flags.get("use_helmet", flags.get("helmet", False)))
        use_person = bool(flags.get("use_person", flags.get("person", False)))
        use_face = bool(flags.get("use_face", flags.get("face", False)))
        use_appearance = bool(flags.get("use_appearance", flags.get("appearance", False)))

        return {
            "use_helmet": use_helmet,
            "use_pose": use_pose,
            "use_person": use_person,
            "use_face": use_face,
            "use_appearance": use_appearance,
        }

    @classmethod
    def _flags_to_detection_modes(cls, flags: Dict[str, object]) -> List[str]:
        normalized = cls._normalize_model_flags(flags)
        modes: List[str] = []
        if normalized["use_pose"]:
            modes.extend(["fall", "person"])
        if normalized["use_helmet"]:
            modes.append("helmet")
        if normalized["use_face"]:
            modes.append("face")
        if normalized["use_person"] and "person" not in modes:
            modes.append("person")
        if normalized["use_appearance"]:
            modes.append("appearance")
        return modes

    def _parse_detections(
        self,
        detections: Optional[Union[List[str], Mapping[str, object]]],
    ) -> Dict[str, bool]:
        if isinstance(detections, Mapping):
            return self._normalize_model_flags(dict(detections))

        if not detections:
            return {
                "use_helmet": self._helmet_enabled,
                "use_pose": True,
                "use_person": False,
                "use_face": self._face_enabled_default,
                "use_appearance": self._appearance_enabled_default,
            }

        modes = {str(item).lower() for item in detections}
        flags = {
            "use_helmet": "helmet" in modes,
            "use_pose": bool(modes & {"fall", "intrusion", "person", "pose"}),
            "use_person": "person_detector" in modes,
            "use_face": "face" in modes,
            "use_appearance": "appearance" in modes,
        }
        return self._normalize_model_flags(flags)

    def get_camera_model_settings(self, camera_id: str) -> Optional[Dict[str, bool]]:
        flags = self._camera_ai_flags.get(camera_id)
        return dict(flags) if flags is not None else None

    def update_camera_model_settings(
        self,
        camera_id: str,
        model_settings: Dict,
        cameras_json_path: str = "cameras.json",
    ) -> Optional[Dict[str, bool]]:
        if camera_id not in self._cameras:
            return None

        normalized = self._normalize_model_flags(model_settings)
        self._camera_ai_flags[camera_id] = normalized
        self._cameras[camera_id]["detections"] = self._flags_to_detection_modes(normalized)

        if cameras_json_path:
            self._save_camera_model_settings(camera_id, normalized, cameras_json_path)

        logger.info("[%s] DeepStream 모델 설정 업데이트: %s", camera_id, normalized)

        if self.running:
            next_topology = self._inference_topology_signature()
            built = self._built_topology
            # 현재 파이프라인에 없는 요소가 새로 필요해진 경우에만 재시작.
            # 모델을 끌 때(요소가 이미 파이프라인에 있음)는 재시작하지 않고
            # _filter_detections_for_camera가 런타임에 즉시 필터링한다.
            needs_restart = any(
                need and not have
                for need, have in zip(next_topology, built)
            )
            if needs_restart:
                logger.info(
                    "[%s] 파이프라인 재시작 필요: 빌드 토폴로지=%s → 필요 토폴로지=%s",
                    camera_id, built, next_topology,
                )
                self._pipeline_restart_pending = True
                self._restart_pipeline_async("model_settings_changed")
            else:
                self._pipeline_restart_pending = False
                logger.info(
                    "[%s] 런타임 필터 적용 (파이프라인 재시작 없음): 토폴로지=%s",
                    camera_id, next_topology,
                )

        return dict(normalized)

    def _save_camera_model_settings(
        self,
        camera_id: str,
        model_settings: Dict[str, object],
        cameras_json_path: str,
    ) -> None:
        normalized = self._normalize_model_flags(model_settings)
        detections = self._flags_to_detection_modes(normalized)

        with self._cameras_json_lock:
            with open(cameras_json_path, "r", encoding="utf-8") as fp:
                cameras = json.load(fp)

            updated = False
            for camera in cameras:
                if camera.get("id") == camera_id:
                    camera["model_settings"] = normalized
                    camera["detections"] = detections
                    updated = True
                    break

            if not updated:
                raise KeyError(f"camera_id '{camera_id}' not found in cameras config")

            # 원자적 쓰기: 임시 파일에 쓴 뒤 교체
            tmp_path = cameras_json_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(cameras, fp, ensure_ascii=False, indent=2)
            os.replace(tmp_path, cameras_json_path)

    def update_zones(
        self,
        camera_id: str,
        zones_data: Optional[List[Dict]],
        cameras_json_path: str = "cameras.json",
    ) -> bool:
        """카메라 구역을 갱신하고 DeepStream 런타임에 즉시 반영한다."""
        if camera_id not in self._cameras:
            logger.warning("[%s] 등록되지 않은 카메라의 구역 업데이트 요청", camera_id)
            return False

        if self.zone_manager is None:
            try:
                self.zone_manager = ZoneManager(self.config.zones_config)
                logger.info("[%s] DeepStream zone_manager on-demand 초기화", camera_id)
            except Exception as exc:
                logger.warning("[%s] DeepStream zone_manager 초기화 실패: %s", camera_id, exc)
                return False

        try:
            normalized_zones = zones_data or []
            self.zone_manager.save_zones(camera_id, normalized_zones, cameras_json_path)
            self._cameras[camera_id]["zones_data"] = normalized_zones
            return True
        except Exception as exc:
            logger.error("[%s] DeepStream 구역 업데이트 실패: %s", camera_id, exc)
            return False

    def list_registered_faces(self) -> List[Dict[str, str]]:
        return self.face_recognizer.list_faces()

    def register_face(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        entry = self.face_recognizer.register_face(*args, **kwargs)
        self.reload_face_gallery()
        return entry

    def delete_face(self, face_id: str) -> bool:
        deleted = self.face_recognizer.delete_face(face_id)
        if deleted:
            self.reload_face_gallery()
        return deleted

    def reload_face_gallery(self) -> None:
        try:
            self.face_recognizer.reload_gallery()
        except Exception as exc:
            logger.warning("DeepStream 얼굴 갤러리 리로드 실패: %s", exc)

    def add_camera(
        self,
        camera_id: str,
        source: Union[str, int],
        *,
        detections: Optional[Union[List[str], Mapping[str, object]]] = None,
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
        self._camera_ai_flags[camera_id] = self._parse_detections(detections)
        logger.info("[%s] DeepStream 감지 항목: %s", camera_id, self._camera_ai_flags[camera_id])
        if zones_data:
            if self.zone_manager is None:
                try:
                    self.zone_manager = ZoneManager(self.config.zones_config)
                    logger.info("[%s] DeepStream zone_manager on-demand 초기화", camera_id)
                except Exception as exc:
                    logger.warning("[%s] DeepStream zone_manager 초기화 실패: %s", camera_id, exc)
            if self.zone_manager is not None:
                try:
                    self.zone_manager.load_zones(camera_id, zones_data)
                except Exception as exc:
                    logger.warning("[%s] DeepStream 구역 로딩 실패: %s", camera_id, exc)
        elif self.zone_manager is not None:
            try:
                self.zone_manager.load_zones(camera_id, None)
            except Exception as exc:
                logger.warning("[%s] DeepStream 구역 로딩 실패: %s", camera_id, exc)
        logger.info("[%s] 카메라 등록됨 (DeepStream): %s", camera_id, source)
        return True

    def remove_camera(self, camera_id: str) -> None:
        """카메라를 파이프라인에서 제거한다.

        구현 메모:
          - nvurisrcbin.set_state(NULL) → unrequest_pad → pipeline 에서 제거
        """
        self._cameras.pop(camera_id, None)
        self._camera_ai_flags.pop(camera_id, None)
        self.track_manager.remove_camera(camera_id)
        self.violation_filter.purge(camera_id)
        self._synthetic_id_assigner.remove_camera(camera_id)
        remove_camera_face_cache(self._face_identity_cache, camera_id)
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
        """DeepStream 파이프라인을 시작한다.

        구현 순서:
          1. Gst.init(None)
          2. self._build_pipeline()
          3. self._pipeline.set_state(Gst.State.PLAYING)
          4. self._publish_thread 시작
          5. GLib 메인 루프를 백그라운드 스레드에서 실행
        """
        if self.running:
            return

        self.running = True
        self.stop_event.clear()
        logger.info("DeepStreamProcessor.start() 호출됨")

        try:
            self._build_pipeline()
            if self._pipeline is None:
                raise RuntimeError("_build_pipeline() 이 pipeline을 설정하지 않았습니다.")

            self._publish_thread = threading.Thread(
                target=self._publish_loop, daemon=True, name="ds-publish"
            )
            self._publish_thread.start()

            # 얼굴/외형 인식 워커 시작 (GStreamer 메인루프와 분리)
            self._face_worker_thread = threading.Thread(
                target=self._face_worker_loop, daemon=True, name="ds-face-worker"
            )
            self._face_worker_thread.start()

            # GStreamer 파이프라인 재생 시작
            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("파이프라인을 PLAYING 상태로 전환하는 데 실패했습니다.")

            self._main_loop_thread = threading.Thread(
                target=self._main_loop.run,
                daemon=True,
                name="ds-main-loop",
            )
            self._main_loop_thread.start()
            logger.info("DeepStream 파이프라인 시작됨")

        except Exception as exc:
            logger.exception("DeepStream 파이프라인 오류: %s", exc)
            self.stop()
            raise

    def stop(self) -> None:
        """DeepStream 파이프라인을 중지한다."""
        self.running = False
        self.stop_event.set()
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        if self._main_loop is not None and self._main_loop.is_running():
            self._main_loop.quit()
        if self._publish_thread and self._publish_thread.is_alive():
            self._publish_thread.join(timeout=2.0)
        if self._main_loop_thread and self._main_loop_thread.is_alive():
            self._main_loop_thread.join(timeout=2.0)
        if self._face_worker_thread and self._face_worker_thread.is_alive():
            self._face_worker_thread.join(timeout=2.0)
        self.event_publisher.disconnect()
        logger.info("DeepStreamProcessor 중지됨")

    def get_stats(self) -> Dict:
        """처리 통계를 반환한다."""
        return self._build_stats_payload(
            backend="deepstream",
            camera_count=len(self._cameras),
            frames_processed=self._frames_processed,
            frames_dropped=self._frames_dropped,
            events_detected=self._events_detected,
            events_filtered=self._events_filtered,
            events_failed=self._events_failed,
            output_mode=self._output_mode,
            preview_enabled=self._preview_enabled,
            preview_max_fps=self._preview_max_fps,
            preview_ready=self._preview_last_frame_at is not None,
            cameras=len(self._cameras),
        )

    def get_camera_status(self) -> Dict[str, dict]:
        """카메라별 상태를 반환한다."""
        return {
            camera_id: self._build_camera_status_entry(
                connected=self.running,
                source=info.get("source"),
                reconnect_attempts=int(info.get("reconnect_attempts", 0) or 0),
                last_frame_time=self._preview_last_frame_at,
                pad_id=info.get("pad_id"),
            )
            for camera_id, info in self._cameras.items()
        }

    def get_camera_frame(
        self,
        camera_id: str,
        *,
        annotated: bool = False,
        copy_frame: bool = True,
    ) -> Optional[Any]:
        """DeepStream OSD 이후 최신 preview 프레임을 반환한다.

        ``annotated``는 인터페이스 호환용 인자다. DeepStream preview branch가
        이미 nvdsosd 이후 프레임을 가져오므로 bbox/label이 포함된 상태다.
        내부 후처리는 ``copy_frame=False``로 같은 프레임을 읽어 전체 복사를 줄인다.
        """
        with self._preview_frame_lock:
            frame = self._preview_frames.get(camera_id)
            if frame is None and self._preview_camera_id:
                frame = self._preview_frames.get(self._preview_camera_id)
        if frame is None:
            return None
        return frame.copy() if copy_frame else frame

    # ------------------------------------------------------------------
    # 내부 파이프라인 구현 메서드 (스켈레톤)
    # ------------------------------------------------------------------

    def _make_element(self, factory: str, name: str) -> Any:
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"GStreamer element 생성 실패: {factory} ({name})")
        return element

    def _normalize_uri(self, source: Union[str, int]) -> str:
        if isinstance(source, int):
            raise ValueError("DeepStream nvurisrcbin은 현재 RTSP/HTTP/file URI만 지원합니다.")

        value = str(source)
        if "://" in value:
            return value

        path = Path(value).expanduser().resolve()
        return path.as_uri()

    def _on_source_pad_added(self, src: Any, pad: Any, sinkpad: Any) -> None:
        if sinkpad.is_linked():
            return
        ret = pad.link(sinkpad)
        if ret != Gst.PadLinkReturn.OK:
            logger.error("DeepStream source pad link 실패: %s -> %s", src.get_name(), ret)

    def _build_source_entries(self) -> List[Tuple[int, str, Dict, str]]:
        """카메라 설정을 DeepStream source entry 목록으로 변환한다."""
        source_entries: List[Tuple[int, str, Dict, str]] = []
        for camera_id, info in self._cameras.items():
            try:
                source_uri = self._normalize_uri(info["source"])
            except ValueError as exc:
                logger.warning("[%s] DeepStream 소스 제외: %s", camera_id, exc)
                continue
            source_entries.append((len(source_entries), camera_id, info, source_uri))
        return source_entries

    def _load_pphuman_label_map(self, label_map_path: Optional[str]) -> Dict[str, object]:
        """PP-Human SGIE tensor 디코딩용 라벨 맵을 로드한다."""
        candidates = [
            label_map_path,
            os.environ.get("APPEARANCE_LABEL_MAP_PATH"),
            "config/appearance_pphuman_labels.example.json",
        ]
        for value in candidates:
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if not path.exists():
                path = (Path.cwd() / str(value)).resolve()
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                logger.warning("PP-Human SGIE 라벨 맵 로드 실패: %s (%s)", path, exc)
        return {"labels": []}

    def _configure_streammux(self, streammux: Any, n_cams: int) -> None:
        streammux.set_property("batch-size", n_cams)
        streammux.set_property("width", int(os.environ.get("DS_STREAM_WIDTH", "1920")))
        streammux.set_property("height", int(os.environ.get("DS_STREAM_HEIGHT", "1080")))
        # 30fps 기준 배치 타임아웃: 1/30s = 33333µs
        # 카메라 수가 많으면 늘릴 것 (4cam: 33333 / 8cam+: 40000)
        streammux.set_property("batched-push-timeout", int(os.environ.get("DS_BATCH_TIMEOUT_US", "33333")))
        streammux.set_property("live-source", 1)
        streammux.set_property("enable-padding", 1)
        try:
            streammux.set_property("nvbuf-memory-type", int(os.environ.get("DS_NVBUF_MEMORY_TYPE", "0")))
        except TypeError:
            logger.debug("nvstreammux nvbuf-memory-type property 미지원")

    def _configure_infer_elements(
        self,
        nvinfer: Optional[Any],
        helmet_infer: Optional[Any],
        pphuman_infer: Optional[Any],
        n_cams: int,
    ) -> None:
        if nvinfer is not None:
            nvinfer.set_property("config-file-path", str(_INFER_CONFIG))
            nvinfer.set_property("batch-size", n_cams)
            self._set_optional_property(
                nvinfer,
                "interval",
                self._env_int("DS_PRIMARY_INTERVAL", 0),
            )
        if pphuman_infer is not None:
            pphuman_infer.set_property("config-file-path", str(_PPHUMAN_INFER_CONFIG))
            pphuman_infer.set_property("batch-size", n_cams)
            self._set_optional_property(
                pphuman_infer,
                "interval",
                self._env_int("DS_PPHUMAN_INTERVAL", 4),
            )
            self._set_optional_property(
                pphuman_infer,
                "secondary-reinfer-interval",
                self._env_int("DS_PPHUMAN_REINFER_INTERVAL", 15),
            )
        if helmet_infer is not None:
            helmet_infer.set_property("config-file-path", str(_HELMET_INFER_CONFIG))
            helmet_infer.set_property("batch-size", n_cams)
            self._set_optional_property(
                helmet_infer,
                "interval",
                self._env_int("DS_HELMET_INTERVAL", 1),
            )

    def _configure_tracker(self, tracker: Any) -> None:
        if Path(_TRACKER_LIB).exists():
            tracker.set_property("ll-lib-file", _TRACKER_LIB)
        tracker.set_property("ll-config-file", str(_TRACKER_CONFIG))
        tracker.set_property("tracker-width", int(os.environ.get("DS_TRACKER_WIDTH", "640")))
        tracker.set_property("tracker-height", int(os.environ.get("DS_TRACKER_HEIGHT", "384")))
        tracker.set_property("gpu-id", 0)
        try:
            tracker.set_property("enable-past-frame", 1)
        except TypeError:
            logger.debug("nvtracker enable-past-frame property 미지원")

    @staticmethod
    def _configure_output_queue(output_queue: Any) -> None:
        output_queue.set_property("leaky", 2)
        output_queue.set_property("max-size-buffers", 2)
        output_queue.set_property("max-size-bytes", 0)
        output_queue.set_property("max-size-time", 0)

    @staticmethod
    def _link_or_raise(first: Any, second: Any, message: Optional[str] = None) -> None:
        if not first.link(second):
            if message is None:
                message = f"{first.get_name()} -> {second.get_name()} link 실패"
            raise RuntimeError(message)

    def _link_preview_branch(
        self,
        *,
        osd: Any,
        tee: Any,
        output_queue: Any,
        preview_elements: List[Any],
    ) -> Any:
        self._link_or_raise(osd, tee, "nvdsosd -> preview-tee link 실패")
        self._link_or_raise(tee, output_queue, "preview-tee -> output-queue link 실패")
        if preview_elements:
            self._link_or_raise(
                tee,
                preview_elements[0],
                "preview-tee -> preview-queue link 실패",
            )
            preview_previous = preview_elements[0]
            for element in preview_elements[1:]:
                self._link_or_raise(preview_previous, element)
                preview_previous = element
        return output_queue

    def _event_type_for_label(self, label: str) -> EventType:
        normalized = (label or "").strip().lower().replace("-", "_")
        if normalized == "person":
            return EventType.PERSON
        if normalized in {"helmet", "hardhat", "head_protected"}:
            return EventType.HELMET
        if normalized in {"head", "hardhat_off", "no_helmet", "helmet_off", "helmet_missing"}:
            return EventType.HEAD
        if normalized in {"fall", "fall_detected"}:
            return EventType.FALL_DETECTED
        return EventType.OTHER

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str, default: int = 0) -> int:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            logger.warning("잘못된 %s 값입니다: %r, 기본값 %d 사용", name, raw_value, default)
            return default

    @staticmethod
    def _set_optional_property(element: Any, name: str, value: Any) -> None:
        try:
            element.set_property(name, value)
        except TypeError:
            logger.debug("%s property 미지원: %s", element.get_name(), name)

    @staticmethod
    def _read_preview_max_fps() -> float:
        """DeepStream preview 샘플링 FPS를 읽는다.

        별도 값이 없으면 MJPEG 스트림 FPS와 맞춰 브라우저 화면이 불필요하게
        낮은 FPS로 제한되지 않도록 한다.
        """
        raw_value = os.environ.get("DS_PREVIEW_MAX_FPS") or os.environ.get("STREAM_FPS") or "30.0"
        try:
            preview_fps = float(raw_value)
        except (TypeError, ValueError):
            logger.warning("잘못된 DS_PREVIEW_MAX_FPS/STREAM_FPS 값입니다: %r, 기본값 30.0 사용", raw_value)
            return 30.0
        return max(0.0, min(preview_fps, 60.0))

    @staticmethod
    def _parse_class_ids(name: str, default: Optional[set[int]] = None) -> set[int]:
        raw_value = os.environ.get(name)
        if raw_value is None or not raw_value.strip():
            return set(default or set())
        return {
            int(value.strip())
            for value in raw_value.split(",")
            if value.strip()
        }

    def _load_yolo_labels(
        self,
        labels_file: Path,
        env_name: str,
        fallback: Optional[List[str]] = None,
    ) -> List[str]:
        labels: List[str] = []
        if labels_file.exists():
            for line in labels_file.read_text(encoding="utf-8").splitlines():
                label = line.strip()
                if label and not label.startswith("#"):
                    labels.append(label)

        if not labels:
            labels = list(fallback or [])
        if not labels:
            labels = [f"class_{idx}" for idx in range(80)]
            labels[0] = "person"

        env_labels = [label.strip() for label in os.environ.get(env_name, "").split(",")]
        env_labels = [label for label in env_labels if label]
        if env_labels:
            labels = env_labels
        return labels

    def _layer_dims(self, layer: Any) -> List[int]:
        dims = getattr(layer, "inferDims", None)
        if dims is None:
            return []
        num_dims = int(getattr(dims, "numDims", 0) or 0)
        values = []
        for idx in range(num_dims):
            value = int(dims.d[idx])
            if value > 0:
                values.append(value)
        return values

    def _layer_to_numpy(self, layer: Any) -> Any:
        import numpy as np

        dims = self._layer_dims(layer)
        if not dims:
            return None
        size = 1
        for dim in dims:
            size *= dim
        if size <= 0:
            return None

        data_type = int(getattr(layer, "dataType", 0))
        if data_type == int(pyds.NvDsInferDataType.FLOAT):
            c_type = ctypes.c_float
            dtype = np.float32
        elif data_type == int(pyds.NvDsInferDataType.HALF):
            c_type = ctypes.c_uint16
            dtype = np.float16
        else:
            return None

        ptr = pyds.get_ptr(layer.buffer)
        array_type = c_type * size
        raw = array_type.from_address(ptr)
        return np.ctypeslib.as_array(raw).view(dtype).reshape(dims).astype(np.float32, copy=False)

    def _select_yolo_output(self, tensor_meta: Any) -> Any:
        for layer_idx in range(int(tensor_meta.num_output_layers)):
            layer = pyds.get_nvds_LayerInfo(tensor_meta, layer_idx)
            dims = self._layer_dims(layer)
            if len(dims) >= 2 and 4 < min(dims[-2:]) and max(dims[-2:]) >= 100:
                name = getattr(layer, "layerName", "") or ""
                if not name or "output" in str(name):
                    return layer
        return None

    def _select_pphuman_layer(self, tensor_meta: Any) -> Any:
        """PP-Human SGIE tensor에서 fetch_name_0 레이어를 선택한다."""
        for layer_idx in range(int(tensor_meta.num_output_layers)):
            layer = pyds.get_nvds_LayerInfo(tensor_meta, layer_idx)
            name = str(getattr(layer, "layerName", "") or "")
            if "fetch_name_0" in name:
                return layer
        # fallback: 첫 번째 레이어
        if int(tensor_meta.num_output_layers) > 0:
            return pyds.get_nvds_LayerInfo(tensor_meta, 0)
        return None

    def _read_pphuman_obj_scores(self, obj_meta: Any) -> List[float]:
        """NvDsObjectMeta.obj_user_meta_list에서 PP-Human SGIE 26개 score를 추출한다."""
        import numpy as np

        l_user = obj_meta.obj_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                if self._tensor_gie_id(tensor_meta) == self._pphuman_gie_id:
                    layer = self._select_pphuman_layer(tensor_meta)
                    if layer is not None:
                        output = self._layer_to_numpy(layer)
                        if output is not None:
                            return output.reshape(-1).tolist()
            try:
                l_user = l_user.next
            except StopIteration:
                break
        return []

    def _decode_pphuman_for_obj(self, obj_meta: Any) -> Dict[str, Any]:
        """obj_meta에서 PP-Human 26-score를 읽어 appearance 속성 dict로 반환한다."""
        scores = self._read_pphuman_obj_scores(obj_meta)
        if not scores:
            return {}
        try:
            return decode_pphuman_scores(scores, self._pphuman_label_map)
        except Exception as exc:
            logger.debug("PP-Human SGIE score 디코딩 실패: %s", exc)
            return {}

    def _map_yolo_box_to_frame(
        self, box: Any, frame_width: int, frame_height: int
    ) -> Tuple[int, int, int, int]:
        return map_yolo_box_to_frame(
            box,
            frame_width,
            frame_height,
            input_size=float(os.environ.get("DS_YOLO_INPUT_SIZE", "640")),
        )

    def _map_yolo_keypoints_to_frame(
        self, values: Any, frame_width: int, frame_height: int
    ) -> List[List[float]]:
        return map_yolo_keypoints_to_frame(
            values,
            frame_width,
            frame_height,
            input_size=float(os.environ.get("DS_YOLO_INPUT_SIZE", "640")),
        )

    def _is_fall_pose(
        self, keypoints: List[List[float]], width: int, height: int
    ) -> bool:
        import numpy as np

        if not keypoints:
            return False
        try:
            kpts = np.asarray(keypoints, dtype=np.float32)
            return self._fall_detector._check_fall(kpts, width, height)
        except Exception as exc:
            logger.debug("DeepStream pose 낙상 판단 실패: %s", exc)
            return False

    def _is_valid_person_pose(
        self, keypoints: List[List[float]]
    ) -> bool:
        import numpy as np

        if not keypoints:
            return True
        try:
            return self._fall_detector._check_person(np.asarray(keypoints, dtype=np.float32))
        except Exception as exc:
            logger.debug("DeepStream pose 사람 검증 실패: %s", exc)
            return True

    def _nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return nms_detections(
            detections,
            iou_threshold=self._yolo_iou_threshold,
            max_detections=self._yolo_max_detections,
        )

    def _tensor_gie_id(self, tensor_meta: Any) -> int:
        for attr_name in ("unique_id", "gie_unique_id"):
            value = getattr(tensor_meta, attr_name, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return self._pose_gie_id

    @staticmethod
    def _event_iou(first: DetectionEvent, second: DetectionEvent) -> float:
        return event_iou(first, second)

    def _assign_synthetic_object_ids(
        self, camera_name: str, events: List[DetectionEvent]
    ) -> List[DetectionEvent]:
        """Raw tensor 결과에 기존 후처리용 stable object_id를 붙인다."""
        return self._synthetic_id_assigner.assign(camera_name, events)

    def _put_event_dict(self, event_data: Dict[str, Any], camera_name: str) -> bool:
        try:
            self.event_queue.put_nowait(event_data)
            return True
        except Full:
            self._frames_dropped += 1
            logger.warning("[%s] DeepStream 이벤트 큐 가득 참", camera_name)
            return False

    def _enqueue_zone_events(
        self, camera_name: str, zone_events: List[ZoneEvent]
    ) -> None:
        for zone_event in zone_events:
            event_dict = zone_event.to_dict()
            if "type" not in event_dict:
                event_dict["type"] = event_dict.get("event_type")
            event_dict.setdefault("camera_id", camera_name)
            event_dict["backend"] = "deepstream"
            self._put_event_dict(event_dict, camera_name)

    def _refresh_appearance_conditions(self) -> None:
        if not self._appearance_enabled_default and not any(
            flags.get("use_appearance") for flags in self._camera_ai_flags.values()
        ):
            return

        now = time.monotonic()
        if now - self._appearance_conditions_checked_at < self._appearance_conditions_refresh_sec:
            return
        self._appearance_conditions_checked_at = now

        try:
            stat = self._appearance_db_path.stat()
        except FileNotFoundError:
            if self._appearance.conditions:
                self._appearance.set_conditions([])
            self._appearance_conditions_mtime = None
            return
        except OSError as exc:
            logger.debug("외형 조건 DB stat 실패: %s", exc)
            return

        if self._appearance_conditions_mtime == stat.st_mtime:
            return

        conditions = AppearanceConditionStore(self._appearance_db_path).list_all()
        self._appearance.set_conditions(conditions)
        self._appearance_conditions_mtime = stat.st_mtime

    def _feature_flags_for_camera(self, camera_name: str) -> Dict[str, bool]:
        flags = self._camera_ai_flags.get(camera_name)
        if flags is not None:
            return flags
        return {
            "use_helmet": self._helmet_enabled,
            "use_pose": True,
            "use_person": False,
            "use_face": self._face_enabled_default,
            "use_appearance": self._appearance_enabled_default,
        }

    def _any_camera_flag(self, *flag_names: str) -> bool:
        return any(
            bool(flags.get(flag_name))
            for flags in self._camera_ai_flags.values()
            for flag_name in flag_names
        )

    def _inference_topology_signature(self) -> Tuple[bool, bool, bool]:
        """현재 모델 플래그로 필요한 DeepStream nvinfer 구성을 계산한다."""
        primary_enabled = self._any_camera_flag(
            "use_pose",
            "use_person",
            "use_face",
            "use_appearance",
        )
        helmet_enabled = (
            self._helmet_enabled
            and self._any_camera_flag("use_helmet")
            and _HELMET_INFER_CONFIG.exists()
        )
        pphuman_enabled = (
            self._pphuman_sgie_enabled
            and self._any_camera_flag("use_appearance")
            and _PPHUMAN_INFER_CONFIG.exists()
        )
        return primary_enabled, helmet_enabled, pphuman_enabled

    def _restart_pipeline_async(self, reason: str) -> None:
        thread = threading.Thread(
            target=self._restart_pipeline,
            args=(reason,),
            daemon=True,
            name="ds-pipeline-restart",
        )
        thread.start()

    def _restart_pipeline(self, reason: str) -> None:
        with self._pipeline_restart_lock:
            logger.info("DeepStream 파이프라인 재시작 시작: %s", reason)
            try:
                self.stop()
                self.start()
                logger.info("DeepStream 파이프라인 재시작 완료: %s", reason)
            except Exception as exc:
                logger.exception("DeepStream 파이프라인 재시작 실패(%s): %s", reason, exc)
            finally:
                self._pipeline_restart_pending = False

    def _log_appearance_capability_hints(
        self,
        camera_name: str,
        flags: Dict[str, bool],
    ) -> None:
        """외형 검색 가능 여부를 카메라별로 1회 로그로 남긴다."""
        if camera_name in self._appearance_capability_logged:
            return

        backend_name = self._appearance.backend_name
        pphuman_sgie_active = (
            self._pphuman_sgie_enabled
            and flags.get("use_appearance", False)
            and _PPHUMAN_INFER_CONFIG.exists()
        )
        bag_labels = sorted(
            label for label in self._yolo_labels
            if str(label).strip().lower() in BAG_CLASSES
        )
        gender_ready = bool(flags.get("use_face")) and bool(self.face_recognizer.enabled)
        helmet_ready = bool(flags.get("use_helmet")) and self._helmet_enabled
        bag_ready = bool(bag_labels) or backend_name != "hsv"

        logger.info(
            "[%s] 외형 검색 컨텍스트: backend=%s, pphuman_sgie=%s, gender_ready=%s, helmet_ready=%s, bag_ready=%s",
            camera_name,
            backend_name,
            pphuman_sgie_active,
            gender_ready,
            helmet_ready,
            bag_ready,
        )

        if not gender_ready:
            logger.warning(
                "[%s] use_face가 꺼져 있거나 얼굴 인식이 비활성화되어 gender 값이 비어 있을 수 있습니다.",
                camera_name,
            )

        if not helmet_ready:
            logger.warning(
                "[%s] use_helmet 또는 DS_HELMET_ENABLED가 꺼져 있어 has_helmet 검색값이 채워지지 않을 수 있습니다.",
                camera_name,
            )

        if not bag_ready:
            logger.warning(
                "[%s] 현재 backend=%s 이고 bag class labels=%s 이라 backpack/handbag/suitcase 값이 채워지기 어렵습니다.",
                camera_name,
                backend_name,
                ",".join(bag_labels) if bag_labels else "none",
            )

        self._appearance_capability_logged.add(camera_name)

    @staticmethod
    def _nearby_objects_from_events(events: List[DetectionEvent]) -> List[Dict[str, Any]]:
        return events_to_nearby_objects(events)

    def _run_face_recognition(
        self,
        frame: Any,
        person_events: List[DetectionEvent],
        camera_name: str,
    ) -> List[DetectionEvent]:
        return run_deepstream_face_recognition(
            frame=frame,
            person_events=person_events,
            camera_name=camera_name,
            recognizer=self.face_recognizer,
            cache=self._face_identity_cache,
            timestamp_factory=time.time,
            snapshot_saver=self._save_recognized_face_snapshot,
        )

    def _save_recognized_face_snapshot(
        self,
        frame: Any,
        camera_name: str,
        face_name: str,
        face_bbox: Dict[str, int],
        confidence: float,
        now: float,
    ) -> Optional[str]:
        """등록 얼굴 인식 시 현재 프레임을 증거용 스냅샷으로 저장한다."""
        try:
            path = save_recognized_face_snapshot(
                frame=frame,
                camera_name=camera_name,
                face_name=face_name,
                face_bbox=face_bbox,
                confidence=confidence,
                now=now,
                enabled=self._face_snapshot_enabled,
                snapshot_dir=self._face_snapshot_dir,
                cooldown_sec=self._face_snapshot_cooldown_sec,
                last_saved_at=self._last_face_snapshot_at,
            )
            if path:
                logger.info("[%s] 등록 얼굴 스냅샷 저장: %s", camera_name, path)
            return path
        except Exception as exc:
            logger.debug("[%s] 등록 얼굴 스냅샷 저장 실패: %s", camera_name, exc)
            return None

    def _run_context_postprocessing(
        self, camera_name: str, filtered_events: List[DetectionEvent]
    ) -> List[DetectionEvent]:
        flags = self._feature_flags_for_camera(camera_name)
        if not (flags.get("use_face") or flags.get("use_appearance")):
            return []

        person_events = [
            event for event in filtered_events
            if event.event_type == EventType.PERSON
        ]
        if not person_events:
            return []

        frame = self.get_camera_frame(camera_name, copy_frame=False)
        if frame is None:
            logger.debug("[%s] DeepStream context 후처리 스킵: preview frame 없음", camera_name)
            return []

        face_events = (
            self._run_face_recognition(frame, person_events, camera_name)
            if flags.get("use_face")
            else []
        )

        appearance_events: List[DetectionEvent] = []
        if flags.get("use_appearance"):
            self._log_appearance_capability_hints(camera_name, flags)
            self._refresh_appearance_conditions()
            appearance_events = self._appearance_pipeline.run(
                frame,
                person_events,
                face_events,
                camera_id=camera_name,
                use_appearance=True,
                nearby_objects=self._nearby_objects_from_events(filtered_events),
            )
            for event in appearance_events:
                metadata = dict(event.metadata or {})
                metadata.setdefault("backend", "deepstream_context")
                metadata.setdefault("camera_id", camera_name)
                event.metadata = metadata

        return face_events + appearance_events

    def _apply_existing_event_pipeline(
        self, camera_name: str, events: List[DetectionEvent]
    ) -> List[DetectionEvent]:
        if not events:
            return []

        events = self._assign_synthetic_object_ids(camera_name, events)
        tracked_events, removed_ids = self.track_manager.update(camera_name, events)
        if removed_ids:
            self.violation_filter.purge(camera_name, removed_ids)

        filtered_events = self.violation_filter.filter(camera_name, tracked_events)
        self._events_filtered += max(0, len(tracked_events) - len(filtered_events))

        # 얼굴/외형 인식은 비동기 워커에 위임 (GStreamer 메인루프 블로킹 방지)
        self._submit_face_work(camera_name, filtered_events)

        zone_events: List[ZoneEvent] = []
        if self.zone_manager is not None:
            try:
                zone_events = self.zone_manager.check_zones(camera_name, filtered_events)
            except Exception as exc:
                logger.warning("[%s] DeepStream 구역 감지 오류: %s", camera_name, exc)
        self._enqueue_zone_events(camera_name, zone_events)

        for event in filtered_events:
            self._enqueue_event(event, camera_name)

    def _submit_face_work(self, camera_name: str, filtered_events: List[DetectionEvent]) -> None:
        """얼굴/외형 인식 작업을 비동기 워커 큐에 제출한다."""
        flags = self._feature_flags_for_camera(camera_name)
        if not (flags.get("use_face") or flags.get("use_appearance")):
            return
        person_events = [e for e in filtered_events if e.event_type == EventType.PERSON]
        if not person_events:
            return
        frame = self.get_camera_frame(camera_name, copy_frame=True)
        if frame is None:
            return
        try:
            self._face_work_queue.put_nowait((camera_name, person_events, frame, flags, filtered_events))
        except Full:
            logger.debug("[%s] 얼굴 인식 워커 큐 가득 참 — 프레임 건너뜀", camera_name)

    def _face_worker_loop(self) -> None:
        """얼굴/외형 인식 전용 백그라운드 워커 스레드."""
        logger.info("얼굴 인식 비동기 워커 시작")
        while not self.stop_event.is_set():
            try:
                task = self._face_work_queue.get(timeout=0.1)
            except Empty:
                continue
            camera_name, person_events, frame, flags, all_filtered_events = task
            try:
                face_events = (
                    self._run_face_recognition(frame, person_events, camera_name)
                    if flags.get("use_face")
                    else []
                )
                appearance_events: List[DetectionEvent] = []
                if flags.get("use_appearance"):
                    self._log_appearance_capability_hints(camera_name, flags)
                    self._refresh_appearance_conditions()
                    appearance_events = self._appearance_pipeline.run(
                        frame,
                        person_events,
                        face_events,
                        camera_id=camera_name,
                        use_appearance=True,
                        nearby_objects=self._nearby_objects_from_events(all_filtered_events),
                    )
                    for event in appearance_events:
                        metadata = dict(event.metadata or {})
                        metadata.setdefault("backend", "deepstream_context")
                        metadata.setdefault("camera_id", camera_name)
                        event.metadata = metadata
                for event in face_events + appearance_events:
                    self._enqueue_event(event, camera_name)
            except Exception as exc:
                logger.warning("[%s] 얼굴/외형 컨텍스트 후처리 실패: %s", camera_name, exc)

    def _should_enqueue_event(self, event: DetectionEvent) -> bool:
        """동일 이벤트가 프레임마다 MQTT로 발행되지 않도록 제한한다."""
        if self._event_min_interval_seconds <= 0:
            return True

        metadata = event.metadata or {}
        camera_id = str(metadata.get("camera_id", "unknown"))
        event_name = event.event_type.value
        class_idx = int(event.class_idx) if event.class_idx is not None else -1
        object_id = int(event.object_id) if event.object_id is not None else None
        throttle_key = (camera_id, event_name, class_idx, object_id)

        now = time.monotonic()
        last_emit_at = self._last_event_emit_at.get(throttle_key)
        if (
            last_emit_at is not None
            and now - last_emit_at < self._event_min_interval_seconds
        ):
            return False

        self._last_event_emit_at[throttle_key] = now
        return True

    def _enqueue_event(self, event: DetectionEvent, camera_name: str) -> bool:
        if not self._should_enqueue_event(event):
            return False
        try:
            self.event_queue.put_nowait(event)
            return True
        except Full:
            self._frames_dropped += 1
            logger.warning("[%s] DeepStream 이벤트 큐 가득 참", camera_name)
            return False

    def _label_color(self, label: str) -> Tuple[float, float, float, float]:
        normalized = (label or "").strip().lower().replace("-", "_")
        if normalized in {"fall", "fall_detected"}:
            return (1.0, 0.0, 1.0, 1.0)
        if normalized in {"head", "hardhat_off", "no_helmet", "helmet_off", "helmet_missing"}:
            return (1.0, 0.05, 0.05, 1.0)
        if normalized in {"helmet", "hardhat", "head_protected"}:
            return (0.05, 0.9, 0.2, 1.0)
        if normalized == "person":
            return (0.05, 0.55, 1.0, 1.0)
        return (1.0, 0.75, 0.05, 1.0)

    def _add_osd_overlays(
        self,
        batch_meta: Any,
        frame_meta: Any,
        detections: List[Dict[str, Any]],
    ) -> None:
        if not detections:
            return

        # NvDsDisplayMeta 한 개에는 보통 16개 요소만 담긴다.
        max_elements = int(os.environ.get("DS_OSD_MAX_ELEMENTS_PER_META", "16"))
        for start in range(0, len(detections), max_elements):
            chunk = detections[start : start + max_elements]
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_rects = len(chunk)
            display_meta.num_labels = len(chunk)

            for idx, detection in enumerate(chunk):
                x, y, width, height = detection["box"]
                label = str(detection["label"])
                confidence = float(detection["confidence"])
                red, green, blue, alpha = self._label_color(label)

                rect_params = display_meta.rect_params[idx]
                rect_params.left = float(x)
                rect_params.top = float(y)
                rect_params.width = float(width)
                rect_params.height = float(height)
                rect_params.border_width = 4
                rect_params.has_bg_color = 0
                rect_params.border_color.set(red, green, blue, alpha)

                text_params = display_meta.text_params[idx]
                text_params.display_text = f"{label} {confidence:.2f}"
                text_params.x_offset = int(x)
                text_params.y_offset = max(0, int(y) - 12)
                text_params.font_params.font_name = "Serif"
                text_params.font_params.font_size = 14
                text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                text_params.set_bg_clr = 1
                text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.75)

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    def _detections_from_tensor(self, tensor_meta: Any, frame_meta: Any) -> List[Dict[str, Any]]:
        import numpy as np

        gie_id = self._tensor_gie_id(tensor_meta)
        task = self._task_by_gie.get(gie_id, self._yolo_task)
        labels = self._labels_by_gie.get(gie_id, self._yolo_labels)
        class_ids_filter = self._class_ids_by_gie.get(gie_id, self._yolo_class_ids)
        confidence_threshold = self._confidence_by_gie.get(gie_id, self._yolo_conf_threshold)

        layer = self._select_yolo_output(tensor_meta)
        if layer is None:
            return []
        output = self._layer_to_numpy(layer)
        if output is None:
            return []

        frame_width = int(getattr(frame_meta, "source_frame_width", 0) or 0)
        frame_height = int(getattr(frame_meta, "source_frame_height", 0) or 0)
        if frame_width <= 0 or frame_height <= 0:
            frame_width = int(os.environ.get("DS_STREAM_WIDTH", "1920"))
            frame_height = int(os.environ.get("DS_STREAM_HEIGHT", "1080"))

        return detections_from_yolo_output(
            output,
            task=task,
            gie_id=gie_id,
            labels=labels,
            frame_width=frame_width,
            frame_height=frame_height,
            confidence_threshold=confidence_threshold,
            class_ids_filter=class_ids_filter,
            input_size=float(os.environ.get("DS_YOLO_INPUT_SIZE", "640")),
            iou_threshold=self._yolo_iou_threshold,
            max_detections=self._yolo_max_detections,
            fall_checker=self._is_fall_pose,
            person_pose_validator=self._is_valid_person_pose,
        )

    def _filter_detections_for_camera(
        self,
        detections: List[Dict[str, Any]],
        camera_name: str,
    ) -> List[Dict[str, Any]]:
        """카메라별 모델 on/off 설정에 맞지 않는 DeepStream tensor 결과를 제거한다."""
        flags = self._feature_flags_for_camera(camera_name)
        filtered: List[Dict[str, Any]] = []
        for detection in detections:
            event_type = self._event_type_for_label(str(detection.get("label", "")))
            if event_type == EventType.FALL_DETECTED and not flags.get("use_pose"):
                continue
            if event_type == EventType.PERSON and not (
                flags.get("use_pose") or flags.get("use_person")
            ):
                continue
            if event_type in {EventType.HELMET, EventType.HEAD} and not flags.get("use_helmet"):
                continue
            filtered.append(detection)
        return filtered

    def _filter_events_for_camera(
        self,
        events: List[DetectionEvent],
        camera_name: str,
    ) -> List[DetectionEvent]:
        """카메라별 모델 on/off 설정에 맞지 않는 DetectionEvent를 제거한다."""
        flags = self._feature_flags_for_camera(camera_name)
        filtered: List[DetectionEvent] = []
        for event in events:
            if event.event_type == EventType.FALL_DETECTED and not flags.get("use_pose"):
                continue
            if event.event_type == EventType.PERSON and not (
                flags.get("use_pose") or flags.get("use_person")
            ):
                continue
            if event.event_type in {EventType.HELMET, EventType.HEAD} and not flags.get("use_helmet"):
                continue
            filtered.append(event)
        return filtered

    def _emit_tensor_events(
        self, batch_meta: Any, frame_meta: Any, camera_name: str
    ) -> int:
        detected = 0
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                detections = self._filter_detections_for_camera(
                    self._detections_from_tensor(tensor_meta, frame_meta),
                    camera_name,
                )
                self._add_osd_overlays(batch_meta, frame_meta, detections)
                events = detections_to_events(
                    detections,
                    camera_name=camera_name,
                    source_id=int(frame_meta.source_id),
                    frame_num=int(frame_meta.frame_num),
                    timestamp_factory=time.time,
                    event_type_for_label=self._event_type_for_label,
                )
                detected += sum(
                    1 for event in events
                    if event.event_type != EventType.FALL_DETECTED
                )
                self._apply_existing_event_pipeline(camera_name, events)

            try:
                l_user = l_user.next
            except StopIteration:
                break
        return detected

    def _object_meta_events_from_frame(
        self,
        frame_meta: Any,
        camera_name: str,
    ) -> List[DetectionEvent]:
        """DeepStream object_meta_list를 DetectionEvent 목록으로 변환한다."""
        flags = self._feature_flags_for_camera(camera_name)
        attach_appearance = (
            self._pphuman_sgie_enabled
            and flags.get("use_appearance", False)
        )
        events: List[DetectionEvent] = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            event = object_meta_to_event(
                obj_meta,
                camera_name=camera_name,
                source_id=int(frame_meta.source_id),
                frame_num=int(frame_meta.frame_num),
                timestamp_factory=time.time,
                event_type_for_label=self._event_type_for_label,
            )
            if event is not None:
                # PP-Human SGIE 결과: person ROI(class_id=0)에만 appearance 부착
                if attach_appearance and int(obj_meta.class_id) == 0:
                    pphuman_attrs = self._decode_pphuman_for_obj(obj_meta)
                    if pphuman_attrs:
                        if event.metadata is None:
                            event.metadata = {}
                        event.metadata["appearance"] = pphuman_attrs
                        event.metadata["appearance_backend"] = "pphuman_sgie"
                events.append(event)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        return self._filter_events_for_camera(events, camera_name)

    def _create_output_elements(self) -> List[Any]:
        if self._output_mode in {"", "fake", "fakesink", "headless"}:
            sink = self._make_element("fakesink", "sink")
            sink.set_property("sync", False)
            sink.set_property("async", False)
            return [sink]

        if self._output_mode in {
            "mpegts",
            "h264",
            "h264_mpegts",
            "h264-mpegts",
            "rtsp",
            "rtsp_publish",
            "rtsp-publish",
        }:
            h264_elements = self._create_h264_encoder_elements()

            if self._output_mode in {"rtsp", "rtsp_publish", "rtsp-publish"}:
                sink = self._make_element("rtspclientsink", "h264-rtsp-sink")
                sink.set_property(
                    "location",
                    os.environ.get(
                        "DS_RTSP_LOCATION",
                        "rtsp://cctv-media-server:8554/camera_1",
                    ),
                )
                self._set_optional_property(sink, "protocols", "tcp")
                self._set_optional_property(sink, "latency", self._env_int("DS_RTSP_LATENCY_MS", 100))
                return [*h264_elements, sink]

            mux = self._make_element("mpegtsmux", "mpegts-mux")
            sink = self._make_element("udpsink", "mpegts-udp-sink")
            self._set_optional_property(mux, "alignment", 7)
            self._set_optional_property(mux, "pcr-interval", 9000)
            self._set_optional_property(mux, "pat-interval", 9000)
            self._set_optional_property(mux, "pmt-interval", 9000)
            sink.set_property("host", os.environ.get("DS_H264_UDP_HOST", "cctv-media-server"))
            sink.set_property("port", self._env_int("DS_H264_UDP_PORT", 1234))
            sink.set_property("sync", False)
            sink.set_property("async", False)

            # poc_lsb 순차 수정 identity: nvv4l2h264enc가 B-frame 스타일 poc_lsb를
            # 기록하여 MediaMTX DTS 추출기를 혼란케 하므로 0,2,4,... 로 강제 수정한다.
            poc_fixer = _H264PocFixer()
            poc_identity = self._make_element("identity", "poc-fix-identity")
            poc_identity.set_property("signal-handoffs", True)
            poc_identity.set_property("silent", True)

            clock_time_none = getattr(Gst, "CLOCK_TIME_NONE", -1)
            _prev_pts: list = [clock_time_none]  # PTS 단조증가 보정용
            h264_fps = max(1, self._env_int("DS_H264_FPS", 30))
            poc_frame_ns = int(1_000_000_000 / h264_fps)

            def _poc_handoff(element: Any, buf: Any) -> None:  # noqa: ANN401
                size = buf.get_size()
                ok, minfo = buf.map(Gst.MapFlags.READ)
                if not ok:
                    return
                try:
                    data = bytearray(minfo.data[:size])
                finally:
                    buf.unmap(minfo)
                poc_fixer.process_buffer(data)
                buf.fill(0, bytes(data))
                # PTS 단조증가 보장: PTS 역전 시 이전값 + 1프레임으로 보정
                cur_pts = buf.pts
                if cur_pts != clock_time_none:
                    if _prev_pts[0] == clock_time_none:
                        _prev_pts[0] = cur_pts
                    elif cur_pts <= _prev_pts[0]:
                        new_pts = _prev_pts[0] + poc_frame_ns
                        buf.pts = new_pts
                        buf.dts = new_pts
                        _prev_pts[0] = new_pts
                    else:
                        _prev_pts[0] = cur_pts

            poc_identity.connect("handoff", _poc_handoff)
            return [*h264_elements, poc_identity, mux, sink]

        if self._output_mode in {"display", "egl", "ui"}:
            transform = self._make_element("nvegltransform", "egl-transform")
            sink = self._make_element("nveglglessink", "egl-sink")
            sink.set_property("sync", False)
            return [transform, sink]

        raise ValueError(
            "지원하지 않는 DS_OUTPUT_MODE 입니다: "
            f"{self._output_mode}. 사용 가능: fakesink, display, h264-mpegts, rtsp-publish"
        )

    def _create_h264_encoder_elements(self) -> List[Any]:
        converter = self._make_element("nvvideoconvert", "h264-nvvidconv")
        capsfilter = self._make_element("capsfilter", "h264-caps")
        encoder_name = os.environ.get("DS_H264_ENCODER", "nvv4l2h264enc").strip().lower()
        use_x264 = encoder_name in {"x264", "x264enc", "software"}
        encoder = self._make_element("x264enc" if use_x264 else "nvv4l2h264enc", "h264-encoder")
        parser = self._make_element("h264parse", "h264-parser")
        parsed_capsfilter = self._make_element("capsfilter", "h264-parsed-caps")

        width = self._env_int("DS_H264_WIDTH", 1280)
        height = self._env_int("DS_H264_HEIGHT", 720)
        memory = "" if use_x264 else "(memory:NVMM)"
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw{memory},format=NV12,width={width},height={height}"
            ),
        )

        bitrate = self._env_int("DS_H264_BITRATE", 6000000)
        iframe_interval = self._env_int("DS_H264_IFRAME_INTERVAL", 30)
        idr_interval = self._env_int("DS_H264_IDR_INTERVAL", iframe_interval)
        if use_x264:
            encoder.set_property("bitrate", max(1, bitrate // 1000))
            self._set_optional_property(encoder, "speed-preset", "ultrafast")
            self._set_optional_property(encoder, "tune", "zerolatency")
            self._set_optional_property(encoder, "key-int-max", iframe_interval)
            self._set_optional_property(encoder, "byte-stream", True)
            self._set_optional_property(encoder, "bframes", 0)
            self._set_optional_property(encoder, "b-adapt", False)
            self._set_optional_property(encoder, "ref", 1)
            self._set_optional_property(encoder, "cabac", False)
            self._set_optional_property(encoder, "aud", True)
            self._set_optional_property(encoder, "insert-vui", True)
            self._set_optional_property(encoder, "sliced-threads", True)
        else:
            encoder.set_property("bitrate", bitrate)
            self._set_optional_property(encoder, "maxperf-enable", True)
            self._set_optional_property(encoder, "insert-aud", True)
            self._set_optional_property(encoder, "insert-sps-pps", True)
            self._set_optional_property(encoder, "insert-vui", False)  # VUI의 max_num_reorder_frames 오염 방지
            self._set_optional_property(encoder, "iframeinterval", iframe_interval)
            self._set_optional_property(encoder, "idrinterval", idr_interval)
            self._set_optional_property(encoder, "control-rate", 1)
            self._set_optional_property(encoder, "ratecontrol-enable", True)
            self._set_optional_property(encoder, "copy-timestamp", False)
            self._set_optional_property(encoder, "disable-cabac", True)
            self._set_optional_property(encoder, "num-B-Frames", 0)
            self._set_optional_property(encoder, "num-Ref-Frames", 1)
            self._set_optional_property(encoder, "profile", 0)   # Baseline
            self._set_optional_property(encoder, "poc-type", 0)   # poc_type=0: 슬라이스 헤더에 poc_lsb 포함

        self._set_optional_property(parser, "disable-passthrough", True)
        self._set_optional_property(parser, "config-interval", -1)
        parsed_capsfilter.set_property(
            "caps",
            Gst.Caps.from_string("video/x-h264,stream-format=byte-stream,alignment=au"),
        )
        return [converter, capsfilter, encoder, parser, parsed_capsfilter]

    def _create_preview_elements(self) -> List[Any]:
        """OSD 결과를 CPU BGR 프레임으로 복사하는 preview branch를 만든다."""
        queue = self._make_element("queue", "preview-queue")
        converter = self._make_element("nvvideoconvert", "preview-nvvidconv")
        capsfilter = self._make_element("capsfilter", "preview-caps")
        appsink = self._make_element("appsink", "preview-appsink")

        queue.set_property("leaky", 2)
        # 30fps에서 jitter 흡수용으로 2프레임 버퍼 유지 (~66ms)
        queue.set_property("max-size-buffers", 2)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-time", 0)

        caps_parts = ["video/x-raw", "format=BGRx"]
        preview_width = self._env_int("DS_PREVIEW_WIDTH", 0)
        preview_height = self._env_int("DS_PREVIEW_HEIGHT", 0)
        if preview_width > 0 and preview_height > 0:
            caps_parts.extend([f"width={preview_width}", f"height={preview_height}"])
        caps = Gst.Caps.from_string(",".join(caps_parts))
        capsfilter.set_property("caps", caps)

        appsink.set_property("emit-signals", True)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.set_property("sync", False)
        appsink.connect("new-sample", self._on_preview_sample)
        return [queue, converter, capsfilter, appsink]

    def _on_preview_sample(self, sink: Any) -> Any:
        now_monotonic = time.monotonic()
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        if (
            self._preview_min_interval_sec > 0
            and now_monotonic - self._preview_last_sample_at < self._preview_min_interval_sec
        ):
            return Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None or caps.get_size() == 0:
            return Gst.FlowReturn.OK

        structure = caps.get_structure(0)
        width = int(structure.get_value("width") or 0)
        height = int(structure.get_value("height") or 0)
        pixel_format = str(structure.get_value("format") or "")
        if width <= 0 or height <= 0:
            return Gst.FlowReturn.OK

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            import numpy as np

            data = np.frombuffer(map_info.data, dtype=np.uint8)
            if pixel_format == "BGRx":
                expected_size = width * height * 4
                if data.size < expected_size:
                    return Gst.FlowReturn.OK
                # ascontiguousarray: BGRx→BGR 슬라이스 + copy를 단일 패스로 처리
                frame = np.ascontiguousarray(
                    data[:expected_size].reshape((height, width, 4))[:, :, :3]
                )
            elif pixel_format == "BGR":
                expected_size = width * height * 3
                if data.size < expected_size:
                    return Gst.FlowReturn.OK
                frame = data[:expected_size].reshape((height, width, 3)).copy()
            else:
                logger.debug("지원하지 않는 DeepStream preview pixel format: %s", pixel_format)
                return Gst.FlowReturn.OK

            camera_id = self._preview_camera_id or next(iter(self._cameras.keys()), "camera_1")
            with self._preview_frame_lock:
                self._preview_frames[camera_id] = frame
                self._preview_last_frame_at = time.time()
                self._preview_last_sample_at = now_monotonic
        except Exception as exc:
            logger.debug("DeepStream preview sample 처리 실패: %s", exc)
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _build_pipeline(self) -> None:
        """GStreamer 파이프라인을 조립한다.

        파이프라인 구조:
            [카메라별] nvurisrcbin → nvstreammux
                                     ↓
                                   pose nvinfer (TensorRT, _INFER_CONFIG)
                                     ↓
                                   PP-Human SGIE (_PPHUMAN_INFER_CONFIG, optional)
                                     ↓
                                   helmet/head nvinfer (_HELMET_INFER_CONFIG)
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
          [ ] helmet nvinfer 생성 — config-file-path = str(_HELMET_INFER_CONFIG)
          [ ] nvtracker 생성 — ll-config-file = str(_TRACKER_CONFIG)
          [ ] nvdsosd 생성
          [ ] fakesink 생성 (또는 실제 출력용 sink)
          [ ] 엘리먼트 파이프라인에 추가 후 link()
          [ ] nvinfer 출력 src_pad 에 _on_pad_probe probe 부착
          [ ] GLib.MainLoop 생성 및 self._main_loop 에 저장
          [ ] 버스 메시지 핸들러 등록: bus.add_signal_watch() + connect("message", _on_bus_message)
        """
        if not self._cameras:
            raise RuntimeError("DeepStream 파이프라인을 만들 카메라가 없습니다.")
        if not _INFER_CONFIG.exists():
            raise FileNotFoundError(f"nvinfer 설정 파일 없음: {_INFER_CONFIG}")
        primary_enabled = self._any_camera_flag(
            "use_pose",
            "use_person",
            "use_face",
            "use_appearance",
        )
        helmet_enabled = (
            self._helmet_enabled
            and self._any_camera_flag("use_helmet")
            and _HELMET_INFER_CONFIG.exists()
        )
        pphuman_enabled = (
            self._pphuman_sgie_enabled
            and self._any_camera_flag("use_appearance")
            and _PPHUMAN_INFER_CONFIG.exists()
        )

        Gst.init(None)

        pipeline = Gst.Pipeline.new("cctv-deepstream")
        if pipeline is None:
            raise RuntimeError("Gst.Pipeline 생성 실패")

        source_entries = self._build_source_entries()
        if not source_entries:
            raise RuntimeError("DeepStream 파이프라인을 만들 지원 소스가 없습니다.")
        self._preview_camera_id = source_entries[0][1]

        n_cams = len(source_entries)
        streammux = self._make_element("nvstreammux", "streammux")
        nvinfer = self._make_element("nvinfer", "primary-infer") if primary_enabled else None
        pphuman_infer = (
            self._make_element("nvinfer", "pphuman-infer")
            if pphuman_enabled and nvinfer is not None
            else None
        )
        helmet_infer = (
            self._make_element("nvinfer", "helmet-infer")
            if helmet_enabled
            else None
        )
        tracker = self._make_element("nvtracker", "tracker") if (nvinfer or helmet_infer) else None
        converter = self._make_element("nvvideoconvert", "converter")
        osd = self._make_element("nvdsosd", "osd")
        tee = self._make_element("tee", "preview-tee") if self._preview_enabled else None
        output_queue = self._make_element("queue", "output-queue") if tee is not None else None
        output_elements = self._create_output_elements()
        preview_elements = self._create_preview_elements() if tee is not None else []

        self._configure_streammux(streammux, n_cams)
        self._configure_infer_elements(nvinfer, helmet_infer, pphuman_infer, n_cams)
        if tracker is not None:
            self._configure_tracker(tracker)
        if output_queue is not None:
            self._configure_output_queue(output_queue)

        pipeline_elements = [streammux]
        if nvinfer is not None:
            pipeline_elements.append(nvinfer)
        if pphuman_infer is not None:
            pipeline_elements.append(pphuman_infer)
        if helmet_infer is not None:
            pipeline_elements.append(helmet_infer)
        if tracker is not None:
            pipeline_elements.append(tracker)
        pipeline_elements.extend([converter, osd])
        if tee is not None and output_queue is not None:
            pipeline_elements.extend([tee, output_queue, *preview_elements])
        pipeline_elements.extend(output_elements)

        for element in pipeline_elements:
            pipeline.add(element)

        previous = streammux
        probe_element = streammux
        if nvinfer is not None:
            self._link_or_raise(previous, nvinfer, "nvstreammux -> nvinfer link 실패")
            previous = nvinfer
            probe_element = nvinfer
        if pphuman_infer is not None:
            primary_srcpad = nvinfer.get_static_pad("src") if nvinfer is not None else None
            if primary_srcpad is None:
                raise RuntimeError("primary-infer src pad를 찾을 수 없습니다.")
            primary_srcpad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._on_primary_tensor_probe,
                None,
            )
            self._link_or_raise(previous, pphuman_infer, "primary-infer -> pphuman-infer link 실패")
            previous = pphuman_infer
            probe_element = pphuman_infer
            logger.info(
                "PP-Human SGIE 파이프라인 연결 완료: gie_id=%d, config=%s",
                self._pphuman_gie_id,
                _PPHUMAN_INFER_CONFIG,
            )
        if helmet_infer is not None:
            self._link_or_raise(previous, helmet_infer, f"{previous.get_name()} -> helmet-infer link 실패")
            previous = helmet_infer
            probe_element = helmet_infer
        if tracker is not None:
            self._link_or_raise(previous, tracker, f"{previous.get_name()} -> nvtracker link 실패")
            previous = tracker
        self._link_or_raise(previous, converter, f"{previous.get_name()} -> nvvideoconvert link 실패")
        self._link_or_raise(converter, osd, "nvvideoconvert -> nvdsosd link 실패")
        previous = osd
        if tee is not None and output_queue is not None:
            previous = self._link_preview_branch(
                osd=osd,
                tee=tee,
                output_queue=output_queue,
                preview_elements=preview_elements,
            )
        for element in output_elements:
            self._link_or_raise(previous, element)
            previous = element

        for pad_id, camera_id, info, source_uri in source_entries:
            src = self._make_element("nvurisrcbin", f"src-{camera_id}")
            src.set_property("uri", source_uri)
            try:
                src.set_property("latency", int(os.environ.get("DS_RTSP_LATENCY_MS", "200")))
            except TypeError:
                logger.debug("nvurisrcbin latency property 미지원")

            pipeline.add(src)
            sinkpad = streammux.get_request_pad(f"sink_{pad_id}")
            if sinkpad is None:
                raise RuntimeError(f"nvstreammux sink_{pad_id} pad 요청 실패")

            src.connect("pad-added", self._on_source_pad_added, sinkpad)
            static_srcpad = src.get_static_pad("src")
            if static_srcpad is not None:
                self._on_source_pad_added(src, static_srcpad, sinkpad)

            info["src_element"] = src
            info["pad_id"] = pad_id

        srcpad = probe_element.get_static_pad("src")
        if srcpad is None:
            raise RuntimeError(f"{probe_element.get_name()} src pad를 찾을 수 없습니다.")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, self._on_pad_probe, None)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._pipeline = pipeline
        self._main_loop = GLib.MainLoop()
        # 실제 빌드된 파이프라인 토폴로지 기록
        self._built_topology = (
            nvinfer is not None,
            helmet_infer is not None,
            pphuman_infer is not None,
        )
        # pad_id → camera_id 역매핑 캐시 갱신 (매 프레임 재생성 방지)
        self._pad_to_camera = {
            cam_info.get("pad_id"): cam_id
            for cam_id, cam_info in self._cameras.items()
        }
        logger.info(
            "DeepStream 파이프라인 토폴로지: primary=%s, helmet=%s, pphuman=%s",
            *self._built_topology,
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
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            logger.warning("DeepStream EOS 수신")
            self.stop_event.set()
            if self._main_loop is not None:
                self._main_loop.quit()
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("DeepStream 오류: %s debug=%s", err, debug)
            self.stop_event.set()
            if self._main_loop is not None:
                self._main_loop.quit()
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning("DeepStream 경고: %s debug=%s", warn, debug)
        return True

    def _on_primary_tensor_probe(
        self, pad: Any, info: Any, user_data: Any
    ) -> Any:  # Gst.PadProbeReturn
        """primary nvinfer src pad probe — pphuman SGIE 활성 시 passthrough 역할.

        pphuman_infer가 활성화된 경우, 주 탐지(YOLO tensor)는 pphuman_infer
        src pad의 _on_pad_probe에서 처리된다. 이 probe는 primary GIE 직후
        버퍼가 올바르게 흐르는지 확인하는 passthrough 역할만 한다.
        """
        return Gst.PadProbeReturn.OK

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
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        pad_to_camera = self._pad_to_camera

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            camera_name = pad_to_camera.get(frame_meta.source_id, "unknown")
            self._frames_processed += 1

            # 주기적으로 만료된 throttle 키 정리 (메모리 누수 방지)
            if self._frames_processed % 1000 == 0:
                _cutoff = time.monotonic() - self._event_min_interval_seconds * 10
                self._last_event_emit_at = {
                    k: v for k, v in self._last_event_emit_at.items() if v > _cutoff
                }

            detected_from_tensor = self._emit_tensor_events(batch_meta, frame_meta, camera_name)
            self._apply_existing_event_pipeline(
                camera_name,
                self._object_meta_events_from_frame(frame_meta, camera_name),
            )

            if (
                detected_from_tensor == 0
                and frame_meta.frame_num % 300 == 0
                and not self._tensor_probe_warned
            ):
                logger.info(
                    "[%s] DeepStream tensor meta는 수신 중이나 필터 조건을 통과한 객체가 아직 없습니다.",
                    camera_name,
                )
                self._tensor_probe_warned = True

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

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
        while self.running and not self.stop_event.is_set():
            try:
                queue_item = self.event_queue.get(timeout=1.0)
                if publish_queue_item(
                    queue_item,
                    topic_prefix=self.config.mqtt.topic_prefix,
                    mqtt_publish=self._mqtt_publish,
                    event_publisher=self.event_publisher,
                ):
                    self._events_detected += 1
                    continue
                self._events_failed += 1
            except Empty:
                continue
            except Exception as exc:
                logger.error("MQTT 발행 오류: %s", exc)
                self._events_failed += 1
        logger.info("MQTT 발행 스레드 종료")

    def print_stats(self) -> None:
        stats = self.get_stats()
        logger.info(
            "DeepStream stats: frames=%s dropped=%s events=%s cameras=%s",
            stats["frames_processed"],
            stats["frames_dropped"],
            stats["events_detected"],
            stats["cameras"],
        )

    def release_all_cameras(self) -> None:
        self.stop()
