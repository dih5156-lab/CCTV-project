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

import errno
import importlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from ..config import AppConfig
from ..protocols.mqtt_publisher import MqttEventPublisher
from ..utils.face_recognition import FaceRecognitionEngine
from ..utils.zone_detection import ZoneEvent, ZoneManager
from . import _deepstream_element_config as ds_element_config
from . import _deepstream_env as ds_env
from ._context_event_store import ContextEventStore
from ._deepstream_context_worker import DeepStreamContextWorker
from ._deepstream_event_factory import (
    emit_tensor_events,
    filter_detections_for_camera,
    filter_events_for_camera,
    object_meta_events_from_frame,
    process_batch_frames,
)
from ._deepstream_event_queue import (
    apply_existing_event_pipeline as ds_apply_existing_event_pipeline,
)
from ._deepstream_event_queue import enqueue_queue_item as ds_enqueue_queue_item
from ._deepstream_event_queue import enqueue_zone_events as ds_enqueue_zone_events
from ._deepstream_face_context import (
    remove_camera_face_cache,
    run_deepstream_face_recognition,
)
from ._deepstream_labels import (
    event_type_for_label,
    load_pphuman_label_map,
    load_yolo_labels,
    resolve_pphuman_sgie_backend_name,
)
from ._deepstream_model_flags import flags_to_detection_modes, normalize_model_flags
from ._deepstream_osd import add_osd_overlays as ds_add_osd_overlays
from ._deepstream_pipeline_builder import (
    add_pipeline_elements,
    configure_pipeline_elements_bundle,
    create_h264_encoder_elements,
    create_output_elements,
    create_pipeline_elements_bundle,
    create_preview_elements,
    link_deepstream_pipeline_path,
    register_pipeline_runtime_hooks,
    start_pipeline_runtime,
    stop_pipeline_runtime,
    validate_pipeline_prerequisites,
)
from ._deepstream_source_health import (
    build_camera_status_map,
    build_deepstream_stats_fields,
    build_source_entries,
    camera_id_from_message,
    execute_pipeline_restart,
    handle_bus_message,
    mark_restart_pending_if_allowed,
    mark_source_failed,
    start_pipeline_restart_thread,
)
from ._deepstream_source_health import (
    next_source_retry_delay as compute_next_source_retry_delay,
)
from ._deepstream_source_manager import (
    attach_camera_source_to_pipeline,
    attach_camera_sources_batch,
)
from ._deepstream_source_state import rebuild_pad_to_camera
from ._deepstream_tensor_utils import (
    layer_to_numpy,
    select_yolo_output,
    tensor_gie_id,
)
from ._deepstream_tensor_utils import (
    read_pphuman_obj_scores as tensor_read_pphuman_obj_scores,
)
from ._deepstream_topology import (
    any_camera_flag,
    feature_flags_for_camera,
    inference_topology_signature,
)
from ._event_context import (
    log_appearance_capability_hints as event_context_log_appearance_capability_hints,
)
from ._event_context import (
    refresh_appearance_conditions as event_context_refresh_appearance_conditions,
)
from ._event_publish import run_publish_loop
from ._face_snapshot import save_recognized_face_snapshot
from ._fall_shadow_review import (
    FallShadowReviewConfig,
    FallShadowReviewRecorder,
    fall_shadow_event_id,
)
from ._h264_poc_fixer import H264PocFixer
from ._preview_frame_store import PreviewFrameStore, process_preview_sample
from ._synthetic_object_ids import SyntheticObjectIdAssigner, event_iou
from ._yolo_postprocess import (
    detections_from_yolo_output,
    map_yolo_box_to_frame,
    nms_detections,
)
from .ai._appearance_analyzer import BAG_CLASSES, AppearanceAnalyzer
from .ai._appearance_pipeline import AppearancePipeline
from .ai._attribute_backends import decode_pphuman_scores
from .ai._fall_detector import FallDetector
from .ai._falldata_aux import FallDataAuxVerifier
from .base_processor import BaseProcessor
from .event_debouncer import EventDebouncer
from .event_filters import CumulativeViolationFilter, TrackManager
from .events import DetectionEvent, EventType

logger = logging.getLogger(__name__)

_DS_CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "deepstream"
_INFER_CONFIG   = _DS_CONFIG_DIR / "config_infer_primary.txt"
_HELMET_INFER_CONFIG = _DS_CONFIG_DIR / "config_infer_helmet.txt"
_PPHUMAN_INFER_CONFIG = _DS_CONFIG_DIR / "config_infer_pphuman.txt"
_TRACKER_CONFIG = _DS_CONFIG_DIR / "config_tracker.txt"
_STREAMMUX_CONFIG = _DS_CONFIG_DIR / "config_streammux.txt"
_LABELS_FILE    = _DS_CONFIG_DIR / "labels.txt"
_HELMET_LABELS_FILE = _DS_CONFIG_DIR / "labels_helmet.txt"
_TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

Gst: Any = None
GLib: Any = None
pyds: Any = None


def _has_deepstream_modules() -> bool:
    """네이티브 모듈을 import하지 않고 DeepStream Python 모듈 존재만 확인한다."""
    return (
        importlib.util.find_spec("gi") is not None
        and importlib.util.find_spec("pyds") is not None
    )


DEEPSTREAM_AVAILABLE: bool = _has_deepstream_modules()


def _ensure_deepstream_loaded() -> bool:
    """실제 DeepStream/GStreamer 사용 직전에 네이티브 모듈을 로드한다."""
    global DEEPSTREAM_AVAILABLE, GLib, Gst, pyds

    if Gst is not None and GLib is not None and pyds is not None:
        return True
    if not DEEPSTREAM_AVAILABLE:
        return False

    try:
        gi = importlib.import_module("gi")
        gi.require_version("Gst", "1.0")
        pyds = importlib.import_module("pyds")
        GLib = importlib.import_module("gi.repository.GLib")
        Gst = importlib.import_module("gi.repository.Gst")
    except (ImportError, ValueError, OSError) as exc:
        Gst = None
        GLib = None
        pyds = None
        DEEPSTREAM_AVAILABLE = False
        logger.debug(
            "DeepStream 환경을 로드할 수 없습니다 (%s). "
            "DeepStreamProcessor 는 이 환경에서 비활성화됩니다.", exc
        )
        return False

    DEEPSTREAM_AVAILABLE = True
    logger.debug("DeepStream Python bindings (pyds) 로드 성공")
    return True


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
            "DeepStream 이벤트 디바운싱: %s (간격: %.2f초)",
            config.events.debounce_enabled,
            config.events.debounce_seconds,
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
        self._preview_max_fps = self._read_preview_max_fps()
        self._preview_store = PreviewFrameStore(self._preview_max_fps)
        self._pipeline_restart_lock = threading.Lock()
        self._restart_request_lock = threading.Lock()
        self._cameras_json_lock = threading.Lock()
        self._face_work_queue: Queue = Queue(maxsize=8)
        self._face_worker_thread: Optional[threading.Thread] = None
        self._falldata_aux_queue: Queue = Queue(
            maxsize=int(os.environ.get("FALLDATA_AUX_QUEUE_MAXSIZE", "4"))
        )
        self._falldata_aux_thread: Optional[threading.Thread] = None
        self._built_topology: Tuple[bool, bool, bool] = (False, False, False)
        self._pipeline_restart_pending: bool = False
        self._pipeline_restart_min_interval_sec = float(
            os.environ.get("DS_PIPELINE_RESTART_MIN_INTERVAL_SEC", "5.0")
        )
        self._last_pipeline_restart_at = 0.0
        self._source_failure_backoff_sec = float(os.environ.get("DS_SOURCE_FAILURE_BACKOFF_SEC", "60"))
        self._source_backoff_until: Dict[str, float] = {}
        self._source_last_error: Dict[str, str] = {}
        self._helmet_enabled = self._env_bool("DS_HELMET_ENABLED", True)
        self._pphuman_sgie_enabled = self._env_bool("DS_PPHUMAN_SGIE_ENABLED", True)
        self._face_enabled_default = self._env_bool("DS_FACE_ENABLED", False)
        self._appearance_enabled_default = self._env_bool(
            "DS_APPEARANCE_ENABLED",
            bool(config.appearance.enabled),
        )
        self._cameras: Dict[str, Dict] = {}
        self._camera_ai_flags: Dict[str, Dict[str, bool]] = {}
        self._pad_to_camera: Dict[int, str] = {}
        self._context_event_store = ContextEventStore(
            ttl_sec=float(os.environ.get("DS_CONTEXT_EVENT_TTL_SEC", "2.0")),
            maxlen=int(os.environ.get("DS_CONTEXT_EVENTS_MAXLEN", "256")),
        )
        self.event_queue: Queue = Queue(maxsize=config.events.queue_max_size * 3)
        self._debouncer = EventDebouncer(config, self._increment_stat)
        self._frames_processed = 0
        self._frames_dropped = 0
        self._events_detected = 0
        self._events_sent = 0
        self._events_dropped = 0
        self._events_filtered = 0
        self._events_failed = 0
        self._yolo_postprocess_calls = 0
        self._yolo_postprocess_total_seconds = 0.0
        self._yolo_postprocess_max_seconds = 0.0

    def _increment_stat(self, field_name: str, delta: int = 1) -> int:
        """지정한 통계 카운터 값을 증가시키고 최신 값을 반환한다."""
        attr_name = f"_{field_name}"
        current = getattr(self, attr_name, 0)
        new_val = current + delta
        setattr(self, attr_name, new_val)
        return new_val

    def _record_yolo_postprocess_timing(self, elapsed_seconds: float) -> None:
        """YOLO 후처리 지연을 누적해 운영 통계에 노출한다."""
        self._yolo_postprocess_calls += 1
        self._yolo_postprocess_total_seconds += elapsed_seconds
        self._yolo_postprocess_max_seconds = max(
            self._yolo_postprocess_max_seconds,
            elapsed_seconds,
        )

    def _yolo_postprocess_stats(self) -> Dict[str, Any]:
        calls = int(getattr(self, "_yolo_postprocess_calls", 0))
        total_seconds = float(getattr(self, "_yolo_postprocess_total_seconds", 0.0))
        max_seconds = float(getattr(self, "_yolo_postprocess_max_seconds", 0.0))
        average_ms = total_seconds * 1000.0 / calls if calls else 0.0
        return {
            "yolo_postprocess_mode": getattr(self, "_yolo_postprocess_mode", "unknown"),
            "yolo_postprocess_calls": calls,
            "yolo_postprocess_avg_ms": round(average_ms, 3),
            "yolo_postprocess_max_ms": round(max_seconds * 1000.0, 3),
        }

    def _init_yolo_settings(self) -> None:
        """DeepStream nvinfer tensor 후처리 설정을 초기화한다."""
        self._pose_gie_id = int(os.environ.get("DS_POSE_GIE_ID", "1"))
        self._helmet_gie_id = int(os.environ.get("DS_HELMET_GIE_ID", "2"))
        self._pphuman_gie_id = int(os.environ.get("DS_PPHUMAN_GIE_ID", "3"))
        self._yolo_task = os.environ.get("DS_YOLO_TASK", "detect").strip().lower()
        self._yolo_conf_threshold = float(os.environ.get("DS_YOLO_CONFIDENCE", "0.35"))
        self._yolo_iou_threshold = float(os.environ.get("DS_YOLO_IOU_THRESHOLD", "0.45"))
        self._yolo_max_detections = int(os.environ.get("DS_YOLO_MAX_DETECTIONS", "100"))
        self._yolo_postprocess_mode = os.environ.get(
            "DS_YOLO_POSTPROCESS_MODE", "vectorized"
        ).strip().lower()
        if self._yolo_postprocess_mode not in {"vectorized", "legacy"}:
            raise ValueError(
                "DS_YOLO_POSTPROCESS_MODE는 'vectorized' 또는 'legacy'여야 합니다: "
                f"{self._yolo_postprocess_mode}"
            )
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
            self._helmet_gie_id: float(os.environ.get("DS_HELMET_CONFIDENCE", "0.65")),
        }

    def _init_event_filters(self, config: AppConfig) -> None:
        """기존 VideoProcessor 후처리 필터를 재사용하도록 초기화한다."""
        self._synthetic_track_iou = float(os.environ.get("DS_SYNTHETIC_TRACK_IOU", "0.30"))
        self._synthetic_track_timeout = self._read_float_setting(
            "DS_SYNTHETIC_TRACK_TIMEOUT",
            config.processing.track_timeout_seconds,
            1.00,
        )
        self._synthetic_id_assigner = SyntheticObjectIdAssigner(
            track_iou=self._synthetic_track_iou,
            track_timeout=self._synthetic_track_timeout,
        )
        self._fall_detector = FallDetector(
            float(os.environ.get("DS_FALL_HEIGHT_RATIO", config.detection.fall_height_ratio)),
            angle_horizontal=float(os.environ.get("DS_FALL_ANGLE_HORIZONTAL", "55")),
            angle_inverted=float(os.environ.get("DS_FALL_ANGLE_INVERTED", "125")),
            bbox_aspect_ratio=float(os.environ.get("DS_FALL_BBOX_ASPECT_RATIO", "1.35")),
            span_bbox_aspect_ratio=float(os.environ.get("DS_FALL_SPAN_BBOX_ASPECT_RATIO", "1.20")),
            span_ratio=float(os.environ.get("DS_FALL_KEYPOINT_SPAN_RATIO", "0.55")),
            score_threshold=float(os.environ.get("DS_FALL_SCORE_THRESHOLD", "3.0")),
            enable_folded_pose=self._env_bool("DS_FALL_ENABLE_FOLDED_POSE", False),
            suppress_sitting_like_pose=self._env_bool(
                "DS_FALL_SUPPRESS_SITTING_LIKE_POSE",
                False,
            ),
            sitting_like_aspect_ratio=float(
                os.environ.get("DS_FALL_SITTING_LIKE_ASPECT_RATIO", "1.45")
            ),
            min_keypoint_confidence=float(os.environ.get("DS_FALL_MIN_KEYPOINT_CONFIDENCE", "0.25")),
            min_hip_confidence=float(os.environ.get("DS_FALL_MIN_HIP_CONFIDENCE", "0.25")),
            min_leg_confidence=float(os.environ.get("DS_FALL_MIN_LEG_CONFIDENCE", "0.35")),
        )
        self.track_manager = TrackManager(
            track_timeout=self._synthetic_track_timeout,
            track_iou_threshold=self._read_float_setting(
                "DS_TRACK_IOU_THRESHOLD",
                config.processing.track_iou_threshold,
                0.50,
            ),
            min_track_frames=self._read_int_setting(
                "DS_MIN_TRACK_FRAMES",
                config.processing.min_track_frames,
                3,
            ),
            max_missed_frames=self._read_int_setting(
                "DS_TRACK_MAX_MISSED_FRAMES",
                config.processing.track_max_missed_frames,
                30,
            ),
        )
        self.violation_filter = CumulativeViolationFilter(
            history_max_size=config.processing.detection_history_size,
            violation_threshold=config.processing.violation_threshold,
            violation_types={"head"},
            enabled=config.processing.cumulative_detection_enabled,
        )

    @staticmethod
    def _read_float_setting(env_name: str, config_value: Any, default: float) -> float:
        """환경변수/설정값/기본값 우선순위로 float 설정값을 읽는다."""
        return ds_env.read_float_setting(env_name, config_value, default)

    @staticmethod
    def _read_int_setting(env_name: str, config_value: Any, default: int) -> int:
        """환경변수/설정값/기본값 우선순위로 int 설정값을 읽는다."""
        return ds_env.read_int_setting(env_name, config_value, default)

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
        appearance_models_enabled = bool(config.appearance.enabled) or bool(
            getattr(self, "_appearance_enabled_default", False)
        )
        self._appearance = AppearanceAnalyzer(
            backend_name=config.appearance.backend,
            backend_model_path=(
                config.appearance.model_path if appearance_models_enabled else None
            ),
            backend_label_map_path=(
                config.appearance.label_map_path if appearance_models_enabled else None
            ),
            backend_runtime=config.appearance.runtime,
            backend_device=os.environ.get("APPEARANCE_DEVICE", "cpu"),
            backend_input_size=config.appearance.input_size,
            backend_score_threshold=config.appearance.score_threshold,
            bbox_expand_ratio=config.appearance.bbox_expand_ratio,
            color_model_path=(
                os.environ.get("APPEARANCE_COLOR_MODEL_PATH")
                if appearance_models_enabled
                else None
            ),
            color_label_map_path=(
                os.environ.get("APPEARANCE_COLOR_LABEL_MAP_PATH")
                if appearance_models_enabled
                else None
            ),
            color_input_size=int(os.environ.get("APPEARANCE_COLOR_INPUT_SIZE", "160")),
            color_score_threshold=float(os.environ.get("APPEARANCE_COLOR_SCORE_THRESHOLD", "0.75")),
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
        self._fall_shadow_review_log_path = Path(
            os.environ.get(
                "FALL_SHADOW_REVIEW_LOG_PATH",
                "/app/data/logs/fall_shadow_review.jsonl",
            )
        )
        self._fall_shadow_clip_dir = Path(
            os.environ.get("FALL_SHADOW_CLIP_DIR", "/app/data/fall_review_clips")
        )
        self._fall_shadow_save_clips = self._env_bool("FALL_SHADOW_SAVE_CLIPS", False)
        self._fall_shadow_near_miss_enabled = self._env_bool(
            "FALL_SHADOW_NEAR_MISS_LOG",
            False,
        )
        self._fall_shadow_near_miss_cooldown_sec = float(
            os.environ.get("FALL_SHADOW_NEAR_MISS_COOLDOWN_SECONDS", "10.0")
        )
        self._fall_shadow_near_miss_last_at: Dict[Tuple[str, int], float] = {}
        self._falldata_aux = FallDataAuxVerifier()
        self._fall_aux_confirm_borderline = self._env_bool(
            "FALLDATA_AUX_CONFIRM_BORDERLINE",
            False,
        )
        raw_confirm_max_score = os.environ.get("FALLDATA_AUX_CONFIRM_MAX_FALL_SCORE", "").strip()
        self._fall_aux_confirm_max_fall_score = (
            float(raw_confirm_max_score) if raw_confirm_max_score else None
        )
        self._fall_aux_compare_veto_enabled = self._env_bool(
            "FALLDATA_AUX_COMPARE_VETO_ENABLED",
            False,
        )
        self._fall_aux_compare_veto_min_fall_score = float(
            os.environ.get("FALLDATA_AUX_COMPARE_VETO_MIN_FALL_SCORE", "0") or "0"
        )
        self._pphuman_label_map = self._load_pphuman_label_map(config.appearance.label_map_path)
        self._context_worker = DeepStreamContextWorker(
            queue=self._face_work_queue,
            feature_flags_for_camera=self._feature_flags_for_camera,
            remember_context_events=self._remember_context_events,
            collect_context_events=self._collect_context_events,
            get_camera_frame=self.get_camera_frame,
            run_face_recognition=self._run_face_recognition,
            log_appearance_capability_hints=self._log_appearance_capability_hints,
            refresh_appearance_conditions=self._refresh_appearance_conditions,
            appearance_pipeline=self._appearance_pipeline,
            enqueue_event=self._enqueue_event,
        )
        self.zone_manager: Optional[ZoneManager] = None
        if config.zone_detection:
            try:
                self.zone_manager = ZoneManager(config.zones_config)
            except Exception as exc:
                logger.warning("DeepStream ZoneManager 초기화 실패: %s", exc)

    def _init_pipeline_handles(self) -> None:
        """파이프라인/루프/스레드 핸들 멤버를 초기 상태로 만든다."""
        self._pipeline: Any = None
        self._main_loop: Any = None
        self._publish_thread: Optional[threading.Thread] = None
        self._main_loop_thread: Optional[threading.Thread] = None
        self._mqtt_publish: Optional[Callable[[str, dict], None]] = None

    def _init_event_publisher(self, config: AppConfig) -> None:
        """DeepStream 이벤트 전송용 MQTT 퍼블리셔를 초기화한다."""
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
        """현재 등록된 카메라 설정 사본을 반환한다."""
        return dict(self._cameras)

    @staticmethod
    def _normalize_model_flags(flags: Dict[str, object]) -> Dict[str, bool]:
        """카메라 모델 플래그를 bool 기반 표준 형태로 정규화한다."""
        return normalize_model_flags(flags)

    @classmethod
    def _flags_to_detection_modes(cls, flags: Dict[str, object]) -> List[str]:
        """정규화 플래그를 카메라 detections 모드 목록으로 변환한다."""
        return flags_to_detection_modes(flags)

    def _parse_detections(
        self,
        detections: Optional[Union[List[str], Mapping[str, object]]],
    ) -> Dict[str, bool]:
        """입력 detections 값을 카메라별 기능 플래그로 해석한다."""
        if isinstance(detections, Mapping):
            return self._normalize_model_flags(dict(detections))

        if not detections:
            return {
                "use_helmet": getattr(self, "_helmet_enabled", True),
                "use_pose": True,
                "use_person": False,
                "use_face": getattr(self, "_face_enabled_default", False),
                "use_appearance": getattr(self, "_appearance_enabled_default", False),
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
        """카메라별 현재 모델 활성화 플래그를 반환한다."""
        flags = self._camera_ai_flags.get(camera_id)
        return dict(flags) if flags is not None else None

    def update_camera_model_settings(
        self,
        camera_id: str,
        model_settings: Dict,
        cameras_json_path: str = "cameras.json",
    ) -> Optional[Dict[str, bool]]:
        """카메라 모델 설정을 갱신하고 필요 시 파이프라인 재시작을 예약한다."""
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
                self._request_pipeline_restart("model_settings_changed")
            else:
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
        """카메라 모델 플래그를 cameras.json에 원자적으로 저장한다."""
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

            tmp_path = cameras_json_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(cameras, fp, ensure_ascii=False, indent=2)
            try:
                os.replace(tmp_path, cameras_json_path)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise
                logger.warning(
                    "cameras.json 원자적 교체 실패(EBUSY) - bind mount 파일로 판단하여 직접 저장으로 재시도: %s",
                    cameras_json_path,
                )
                with open(cameras_json_path, "w", encoding="utf-8") as fp:
                    json.dump(cameras, fp, ensure_ascii=False, indent=2)
                    fp.write("\n")
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass

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
        """등록된 얼굴 목록을 조회한다."""
        return self.face_recognizer.list_faces()

    def register_face(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        """얼굴을 등록한 뒤 갤러리를 즉시 다시 로드한다."""
        entry = self.face_recognizer.register_face(*args, **kwargs)
        self.reload_face_gallery()
        return entry

    def delete_face(self, face_id: str) -> bool:
        """지정 얼굴을 삭제하고 성공 시 갤러리를 다시 로드한다."""
        deleted = self.face_recognizer.delete_face(face_id)
        if deleted:
            self.reload_face_gallery()
        return deleted

    def reload_face_gallery(self) -> None:
        """얼굴 인식 갤러리 메모리를 강제로 새로고침한다."""
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
            "reconnect_attempts": 0,
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
        self._context_event_store.clear_camera(camera_id)
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
        self._source_backoff_until.pop(camera_id, None)
        self._source_last_error.pop(camera_id, None)
        if camera_id in self._cameras:
            self._cameras[camera_id]["source"] = source
        elif not self.add_camera(camera_id, source):
            return
        if not self._add_camera_to_pipeline(camera_id):
            self._pipeline_restart_pending = True
            self._restart_pipeline_async(f"camera_retry:{camera_id}:fallback_restart")

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

            (
                self._publish_thread,
                self._main_loop_thread,
                self._face_worker_thread,
            ) = start_pipeline_runtime(
                pipeline=self._pipeline,
                main_loop=self._main_loop,
                gst_module=Gst,
                publish_loop_target=self._publish_loop,
                face_worker_loop_target=self._face_worker_loop,
            )
            self._falldata_aux_thread = threading.Thread(
                target=self._falldata_aux_worker_loop,
                name="DeepStreamFallDataAuxWorker",
                daemon=True,
            )
            self._falldata_aux_thread.start()
            logger.info("DeepStream 파이프라인 시작됨")

        except Exception as exc:
            if self.next_source_retry_delay() is not None and "지원 소스가 없습니다" in str(exc):
                logger.warning("DeepStream 파이프라인 대기: %s", exc)
            else:
                logger.exception("DeepStream 파이프라인 오류: %s", exc)
            self.stop()
            raise

    def stop(self) -> None:
        """DeepStream 파이프라인을 중지한다."""
        self.running = False
        self.stop_event.set()

        stop_pipeline_runtime(
            pipeline=self._pipeline,
            main_loop=self._main_loop,
            publish_thread=self._publish_thread,
            main_loop_thread=self._main_loop_thread,
            face_worker_thread=self._face_worker_thread,
            gst_module=Gst,
            join_timeout_sec=2.0,
        )
        if self._falldata_aux_thread and self._falldata_aux_thread.is_alive():
            self._falldata_aux_thread.join(timeout=2.0)
        self._falldata_aux_thread = None
        self._pipeline = None
        self.event_publisher.disconnect()
        logger.info("DeepStreamProcessor 중지됨")

    def get_stats(self) -> Dict:
        """처리 통계를 반환한다."""
        stats_fields = build_deepstream_stats_fields(
            cameras_count=len(self._cameras),
            frames_processed=self._frames_processed,
            frames_dropped=self._frames_dropped,
            events_detected=self._events_detected,
            events_sent=getattr(self, "_events_sent", 0),
            events_dropped=getattr(self, "_events_dropped", 0),
            events_filtered=self._events_filtered,
            events_failed=self._events_failed,
            output_mode=self._output_mode,
            preview_enabled=self._preview_enabled,
            preview_max_fps=getattr(self, "_preview_max_fps", 0.0),
            preview_ready=getattr(self, "_preview_store", None) is not None
            and self._preview_store.last_frame_at is not None,
        )
        stats_fields.update(self._yolo_postprocess_stats())
        return self._build_stats_payload(
            backend="deepstream",
            **stats_fields,
        )

    def get_camera_status(self) -> Dict[str, dict]:
        """카메라별 상태를 반환한다."""
        return build_camera_status_map(
            cameras=self._cameras,
            running=self.running,
            source_backoff_until=self._source_backoff_until,
            source_last_error=self._source_last_error,
            preview_last_frame_at=self._preview_store.last_frame_at,
            now_monotonic=time.monotonic(),
            build_status_entry=self._build_camera_status_entry,
        )

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
        return self._preview_store.get_frame(
            camera_id,
            fallback_camera_id=getattr(self, "_preview_camera_id", None),
            copy_frame=copy_frame,
        )

    def get_detection_snapshot(self) -> Dict[str, dict]:
        """카메라별 최신 탐지 스냅샷을 반환한다."""
        return self._snapshot_store.snapshot()

    # ------------------------------------------------------------------
    # 내부 파이프라인 구현 메서드 (스켈레톤)
    # ------------------------------------------------------------------

    def _make_element(self, factory: str, name: str) -> Any:
        """GStreamer 엘리먼트를 생성하고 실패 시 예외를 발생시킨다."""
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"GStreamer element 생성 실패: {factory} ({name})")
        return element

    def _normalize_uri(self, source: Union[str, int]) -> str:
        """카메라 source 값을 DeepStream에서 사용할 URI 형태로 정규화한다."""
        if isinstance(source, int):
            raise ValueError("DeepStream nvurisrcbin은 현재 RTSP/HTTP/file URI만 지원합니다.")

        value = str(source)
        if "://" in value:
            return value

        path = Path(value).expanduser().resolve()
        return path.as_uri()

    def _on_source_pad_added(self, src: Any, pad: Any, sinkpad: Any) -> None:
        """동적 소스 패드를 streammux sink pad에 안전하게 링크한다."""
        if sinkpad.is_linked():
            return
        ret = pad.link(sinkpad)
        if ret != Gst.PadLinkReturn.OK:
            logger.error("DeepStream source pad link 실패: %s -> %s", src.get_name(), ret)

    def next_source_retry_delay(self) -> Optional[float]:
        """소스 backoff 종료까지 남은 최소 대기 시간을 반환한다."""
        return compute_next_source_retry_delay(
            self._source_backoff_until,
            now=time.monotonic(),
        )

    def _build_source_entries(self) -> List[Tuple[int, str, Dict, str]]:
        return build_source_entries(
            cameras=self._cameras,
            source_backoff_until=self._source_backoff_until,
            now=time.monotonic(),
            normalize_uri=self._normalize_uri,
        )

    def _mark_source_failed(self, camera_id: str, reason: str) -> None:
        mark_source_failed(
            cameras=self._cameras,
            source_backoff_until=self._source_backoff_until,
            source_last_error=self._source_last_error,
            source_failure_backoff_sec=self._source_failure_backoff_sec,
            camera_id=camera_id,
            reason=reason,
            now=time.monotonic(),
        )

    def _camera_id_from_message(self, message: Any, debug: object) -> Optional[str]:
        return camera_id_from_message(cameras=self._cameras, message=message, debug=debug)

    def _attach_camera_source_to_pipeline(
        self,
        camera_id: str,
        *,
        pad_id: Optional[int] = None,
        pipeline: Optional[Any] = None,
        streammux: Optional[Any] = None,
        detach_existing: bool = False,
    ) -> bool:
        pipeline = pipeline or self._pipeline
        streammux = streammux or (pipeline.get_by_name("streammux") if pipeline else None)
        if pipeline is None or streammux is None or camera_id not in self._cameras:
            return False
        attached = attach_camera_source_to_pipeline(
            camera_id=camera_id,
            info=self._cameras[camera_id],
            pad_to_camera=self._pad_to_camera,
            gst_module=Gst,
            pipeline=pipeline,
            streammux=streammux,
            pad_id=pad_id,
            make_element=self._make_element,
            normalize_uri=self._normalize_uri,
            on_source_pad_added=self._on_source_pad_added,
            next_pad_id=lambda: max(self._pad_to_camera.keys(), default=-1) + 1,
            detach_existing=detach_existing,
        )
        if attached and self._preview_camera_id is None:
            self._preview_camera_id = camera_id
        return attached

    def _add_camera_to_pipeline(self, camera_id: str) -> bool:
        if not self.running or self._pipeline is None or camera_id not in self._cameras:
            return False
        info = self._cameras[camera_id]
        existing_source = info.get("src_element") is not None
        built_source_count = getattr(
            self, "_built_source_count", len(getattr(self, "_pad_to_camera", {}))
        )
        if not existing_source and len(self._cameras) > built_source_count:
            return False
        return self._attach_camera_source_to_pipeline(
            camera_id,
            detach_existing=existing_source,
        )

    def _load_pphuman_label_map(self, label_map_path: Optional[str]) -> Dict[str, object]:
        """PP-Human 속성 디코딩용 라벨 맵을 로드한다."""
        return load_pphuman_label_map(label_map_path)

    def _put_event_dict(self, event_data: Dict[str, Any], camera_name: str) -> bool:
        return ds_enqueue_queue_item(
            event_queue=self.event_queue,
            queue_item=event_data,
            camera_name=camera_name,
            increment_stat=self._increment_stat,
        )

    def _resolve_pphuman_sgie_backend_name(self) -> str:
        return resolve_pphuman_sgie_backend_name(
            pphuman_infer_config=self._resolve_pphuman_infer_config(),
            pphuman_label_map=self._pphuman_label_map,
        )

    @staticmethod
    def _resolve_pphuman_infer_config() -> Path:
        return Path(
            os.environ.get("DS_PPHUMAN_INFER_CONFIG", str(_PPHUMAN_INFER_CONFIG))
        )

    def _fall_shadow_recorder(self) -> FallShadowReviewRecorder:
        config = FallShadowReviewConfig(
            review_log_path=Path(getattr(self, "_fall_shadow_review_log_path", "/app/data/logs/fall_shadow_review.jsonl")),
            clip_dir=Path(getattr(self, "_fall_shadow_clip_dir", "/app/data/fall_review_clips")),
            save_clips=bool(getattr(self, "_fall_shadow_save_clips", False)),
            near_miss_enabled=bool(getattr(self, "_fall_shadow_near_miss_enabled", False)),
            near_miss_cooldown_sec=float(getattr(self, "_fall_shadow_near_miss_cooldown_sec", 10.0)),
        )
        near_miss_last_at = getattr(self, "_fall_shadow_near_miss_last_at", None)
        if near_miss_last_at is None:
            near_miss_last_at = {}
            self._fall_shadow_near_miss_last_at = near_miss_last_at
        recorder = getattr(self, "_fall_shadow", None)
        if recorder is None:
            recorder = FallShadowReviewRecorder(
                config,
                falldata_aux=getattr(self, "_falldata_aux", None),
                near_miss_last_at=near_miss_last_at,
            )
            self._fall_shadow = recorder
        else:
            recorder.config = config
            recorder.falldata_aux = getattr(self, "_falldata_aux", None)
            recorder.near_miss_last_at = near_miss_last_at
        return recorder

    def _submit_falldata_aux_work(
        self, camera_name: str, filtered_events: List[DetectionEvent]
    ) -> Optional[DetectionEvent]:
        return self._fall_shadow_recorder().submit_aux_work(
            self._falldata_aux_queue, camera_name, filtered_events
        )

    def _write_fall_near_miss_review_records(
        self, camera_name: str, filtered_events: List[DetectionEvent]
    ) -> None:
        self._fall_shadow_recorder().write_near_miss_records(
            camera_name, filtered_events, now_monotonic=time.monotonic()
        )

    def _write_fall_shadow_review_record(
        self,
        camera_name: str,
        event_payload: Dict[str, Any],
        result: Dict[str, Any],
        *,
        near_miss: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._fall_shadow_recorder().write_record(
            camera_name, event_payload, result, near_miss=near_miss
        )

    def _write_fall_event_review_record(
        self, camera_name: str, event: DetectionEvent
    ) -> None:
        if event.event_type != EventType.FALL_DETECTED:
            return
        try:
            self._write_fall_shadow_review_record(
                camera_name,
                event.to_dict(),
                {
                    "status": "not_run",
                    "reason": "deepstream_event_only",
                    "confirmed": None,
                },
            )
        except Exception as exc:
            logger.warning("[%s] fall event review 기록 실패: %s", camera_name, exc)

    @staticmethod
    def _fall_shadow_event_id(camera_name: str, event_payload: Dict[str, Any], created_at: Any) -> str:
        return fall_shadow_event_id(camera_name, event_payload, created_at)

    def _should_confirm_fall_with_aux_before_publish(self, event: DetectionEvent) -> bool:
        if event.event_type != EventType.FALL_DETECTED:
            return False
        if not getattr(self, "_fall_aux_confirm_borderline", False):
            return False
        falldata_aux = getattr(self, "_falldata_aux", None)
        if falldata_aux is None or not falldata_aux.enabled:
            return False
        score = float((event.metadata or {}).get("fall_score", 0.0))
        max_score = getattr(self, "_fall_aux_confirm_max_fall_score", None)
        if max_score is not None:
            return score <= float(max_score)
        return score <= float(self._fall_detector.score_threshold)

    def _enqueue_aux_confirmed_fall_event(
        self, camera_name: str, event_payload: Dict[str, Any], result: Dict[str, Any]
    ) -> bool:
        if result.get("status") != "ok" or result.get("confirmed") is not True:
            return False
        metadata = dict(event_payload.get("metadata") or {})
        if (
            getattr(self, "_fall_aux_compare_veto_enabled", False)
            and float(metadata.get("fall_score", 0.0))
            >= float(getattr(self, "_fall_aux_compare_veto_min_fall_score", 0.0))
            and (result.get("compare_model") or {}).get("status") == "ok"
            and (result.get("compare_model") or {}).get("confirmed") is False
        ):
            return False
        metadata.pop("falldata_aux_publish_pending", None)
        metadata["falldata_aux"] = result
        metadata["falldata_aux_confirmed"] = True
        queue_item = dict(event_payload)
        queue_item["metadata"] = metadata
        return self._put_event_dict(queue_item, camera_name)

    def _configure_streammux(self, streammux: Any, n_cams: int) -> None:
        """카메라 수에 맞춰 nvstreammux 속성을 설정한다."""
        ds_element_config.configure_streammux(streammux, n_cams)

    def _configure_infer_elements(
        self,
        nvinfer: Optional[Any],
        helmet_infer: Optional[Any],
        pphuman_infer: Optional[Any],
        n_cams: int,
    ) -> None:
        """Primary/Helmet/PP-Human 추론 엘리먼트 속성을 일괄 설정한다."""
        ds_element_config.configure_infer_elements(
            nvinfer=nvinfer,
            helmet_infer=helmet_infer,
            pphuman_infer=pphuman_infer,
            n_cams=n_cams,
            infer_config=_INFER_CONFIG,
            helmet_infer_config=_HELMET_INFER_CONFIG,
            pphuman_infer_config=self._resolve_pphuman_infer_config(),
            env_int=self._env_int,
            set_property_optional=self._set_optional_property,
        )

    def _configure_tracker(self, tracker: Any) -> None:
        """nvtracker 라이브러리/설정 경로 및 공통 옵션을 적용한다."""
        ds_element_config.configure_tracker(
            tracker=tracker,
            tracker_lib=_TRACKER_LIB,
            tracker_config=_TRACKER_CONFIG,
        )

    @staticmethod
    def _configure_output_queue(output_queue: Any) -> None:
        """출력 큐의 누수/버퍼 정책을 설정해 지연을 제어한다."""
        ds_element_config.configure_output_queue(output_queue)

    @staticmethod
    def _link_or_raise(first: Any, second: Any, message: Optional[str] = None) -> None:
        """두 엘리먼트를 링크하고 실패 시 예외로 중단한다."""
        ds_element_config.link_or_raise(first, second, message)

    def _link_preview_branch(
        self,
        *,
        osd: Any,
        tee: Any,
        output_queue: Any,
        preview_elements: List[Any],
    ) -> Any:
        """OSD 이후 tee에서 preview 브랜치를 연결한다."""
        return ds_element_config.link_preview_branch(
            osd=osd,
            tee=tee,
            output_queue=output_queue,
            preview_elements=preview_elements,
            link=self._link_or_raise,
        )

    def _event_type_for_label(self, label: str) -> EventType:
        """라벨 문자열을 내부 EventType으로 변환한다."""
        return event_type_for_label(label)

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        """환경변수를 bool로 파싱한다."""
        return ds_env.env_bool(name, default)

    @staticmethod
    def _env_int(name: str, default: int = 0) -> int:
        """환경변수를 int로 파싱한다."""
        return ds_env.env_int(name, default)

    @staticmethod
    def _set_optional_property(element: Any, name: str, value: Any) -> None:
        """지원되는 경우에만 GStreamer 속성을 설정한다."""
        ds_element_config.set_optional_property(element, name, value)

    @staticmethod
    def _read_preview_max_fps() -> float:
        """DeepStream preview 샘플링 FPS를 읽는다.

        별도 값이 없으면 MJPEG 스트림 FPS와 맞춰 브라우저 화면이 불필요하게
        낮은 FPS로 제한되지 않도록 한다.
        """
        return ds_env.read_preview_max_fps()

    @staticmethod
    def _parse_class_ids(name: str, default: Optional[set[int]] = None) -> set[int]:
        """클래스 ID 환경변수를 정수 집합으로 파싱한다."""
        return ds_env.parse_class_ids(name, default)

    def _load_yolo_labels(
        self,
        labels_file: Path,
        env_name: str,
        fallback: Optional[List[str]] = None,
    ) -> List[str]:
        """라벨 파일/환경변수/기본값 순으로 YOLO 라벨 목록을 구성한다."""
        return load_yolo_labels(labels_file, env_name, fallback)

    def _decode_pphuman_for_obj(self, obj_meta: Any) -> Dict[str, Any]:
        """obj_meta에서 PP-Human 26-score를 읽어 appearance 속성 dict로 반환한다."""
        scores = tensor_read_pphuman_obj_scores(
            obj_meta,
            pyds_module=pyds,
            pphuman_gie_id=self._pphuman_gie_id,
            default_gie_id=self._pose_gie_id,
        )
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
        """모델 입력 좌표계 bbox를 원본 프레임 좌표계로 변환한다."""
        return map_yolo_box_to_frame(
            box,
            frame_width,
            frame_height,
            input_size=float(os.environ.get("DS_YOLO_INPUT_SIZE", "640")),
        )

    def _is_fall_pose(
        self, keypoints: List[List[float]], width: int, height: int
    ) -> Dict[str, Any]:
        """키포인트와 bbox 크기를 이용해 낙상 자세 여부를 판정한다."""
        import numpy as np

        if not keypoints:
            return {"is_fall": False, "score": 0.0, "reasons": ["missing_keypoints"]}
        try:
            kpts = np.asarray(keypoints, dtype=np.float32)
            score = self._fall_detector._score_fall(kpts, width, height)
            is_fall = score.score >= self._fall_detector.score_threshold
            near_miss = None
            if not is_fall:
                folded_score = self._fall_detector.folded_floor_pose_score(kpts, height)
                if folded_score is not None:
                    near_miss = {
                        "type": "folded_floor_pose",
                        "score": score.score,
                        "reasons": [f"folded_floor_pose:{folded_score:.2f}"],
                    }
                elif score.score > 0.0:
                    near_miss = {
                        "type": "low_score_pose",
                        "score": score.score,
                        "reasons": list(score.reasons),
                    }
            return {
                "is_fall": is_fall,
                "score": score.score,
                "reasons": list(score.reasons),
                "near_miss": near_miss,
            }
        except Exception as exc:
            logger.debug("DeepStream pose 낙상 판단 실패: %s", exc)
            return {"is_fall": False, "score": 0.0, "reasons": ["error"]}

    def _is_valid_person_pose(
        self, keypoints: List[List[float]]
    ) -> bool:
        """키포인트가 유효한 사람 자세인지 검증한다."""
        import numpy as np

        if not keypoints:
            return True
        try:
            return self._fall_detector._check_person(np.asarray(keypoints, dtype=np.float32))
        except Exception as exc:
            logger.debug("DeepStream pose 사람 검증 실패: %s", exc)
            return True

    def _nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """탐지 결과에 Non-Maximum Suppression을 적용한다."""
        return nms_detections(
            detections,
            iou_threshold=self._yolo_iou_threshold,
            max_detections=self._yolo_max_detections,
        )

    @staticmethod
    def _event_iou(first: DetectionEvent, second: DetectionEvent) -> float:
        """두 이벤트 bbox의 IoU를 계산한다."""
        return event_iou(first, second)

    def _assign_synthetic_object_ids(
        self, camera_name: str, events: List[DetectionEvent]
    ) -> List[DetectionEvent]:
        """Raw tensor 결과에 기존 후처리용 stable object_id를 붙인다."""
        return self._synthetic_id_assigner.assign(camera_name, events)

    def _enqueue_queue_item(self, queue_item: Any, camera_name: str) -> bool:
        return ds_enqueue_queue_item(
            event_queue=self.event_queue,
            queue_item=queue_item,
            camera_name=camera_name,
            increment_stat=self._increment_stat,
        )

    def _enqueue_zone_events(
        self, camera_name: str, zone_events: List[ZoneEvent]
    ) -> None:
        ds_enqueue_zone_events(
            camera_name=camera_name,
            zone_events=zone_events,
            enqueue_event_dict=self._enqueue_queue_item,
        )

    def _refresh_appearance_conditions(self) -> None:
        (
            self._appearance_conditions_checked_at,
            self._appearance_conditions_mtime,
        ) = event_context_refresh_appearance_conditions(
            appearance_enabled_default=self._appearance_enabled_default,
            camera_ai_flags=self._camera_ai_flags,
            appearance=self._appearance,
            appearance_db_path=self._appearance_db_path,
            current_mtime=self._appearance_conditions_mtime,
            checked_at=self._appearance_conditions_checked_at,
            refresh_sec=self._appearance_conditions_refresh_sec,
            now_monotonic=time.monotonic(),
        )

    def _feature_flags_for_camera(self, camera_name: str) -> Dict[str, bool]:
        return feature_flags_for_camera(
            camera_ai_flags=getattr(self, "_camera_ai_flags", {}),
            camera_name=camera_name,
            helmet_enabled=getattr(self, "_helmet_enabled", True),
            face_enabled_default=getattr(self, "_face_enabled_default", False),
            appearance_enabled_default=getattr(self, "_appearance_enabled_default", False),
        )

    def _any_camera_flag(self, *flag_names: str) -> bool:
        return any_camera_flag(self._camera_ai_flags, *flag_names)

    def _inference_topology_signature(self) -> Tuple[bool, bool, bool]:
        return inference_topology_signature(
            camera_ai_flags=self._camera_ai_flags,
            helmet_enabled=self._helmet_enabled,
            pphuman_sgie_enabled=self._pphuman_sgie_enabled,
            helmet_config_exists=_HELMET_INFER_CONFIG.exists(),
            pphuman_config_exists=self._resolve_pphuman_infer_config().exists(),
        )

    def _set_pipeline_restart_pending(self, pending: bool) -> None:
        self._pipeline_restart_pending = pending

    def _set_last_pipeline_restart_at(self, timestamp: float) -> None:
        self._last_pipeline_restart_at = timestamp

    def _request_pipeline_restart(self, reason: str) -> bool:
        """중복/폭주를 막으면서 DeepStream 파이프라인 재시작을 예약한다."""
        now = time.monotonic()
        with self._restart_request_lock:
            if not mark_restart_pending_if_allowed(
                running=self.running,
                restart_pending=self._pipeline_restart_pending,
                now=now,
                last_restart_at=self._last_pipeline_restart_at,
                min_interval_sec=self._pipeline_restart_min_interval_sec,
                reason=reason,
                set_pending_cb=self._set_pipeline_restart_pending,
            ):
                return False

        self._restart_pipeline_async(reason)
        return True

    def _restart_pipeline_async(self, reason: str) -> None:
        """재시작 작업 스레드를 시작한다.

        원칙적으로 _request_pipeline_restart()를 통해서만 호출되어야 한다.
        향후 직접 호출이 추가되더라도 게이트를 우회하지 않도록 보호한다.
        """
        with self._restart_request_lock:
            pending = self._pipeline_restart_pending

        start_pipeline_restart_thread(
            pending=pending,
            reason=reason,
            request_pipeline_restart_cb=self._request_pipeline_restart,
            restart_pipeline_cb=self._restart_pipeline,
        )

    def _restart_pipeline(self, reason: str) -> None:
        with self._pipeline_restart_lock:
            try:
                execute_pipeline_restart(
                    reason=reason,
                    stop_cb=self.stop,
                    start_cb=self.start,
                    monotonic_now=time.monotonic,
                    set_last_restart_at_cb=self._set_last_pipeline_restart_at,
                )
            finally:
                with self._restart_request_lock:
                    self._set_pipeline_restart_pending(False)

    def _log_appearance_capability_hints(
        self,
        camera_name: str,
        flags: Dict[str, bool],
    ) -> None:
        """카메라별 외형 분석 활성 상태와 제약 정보를 로그로 남긴다."""
        event_context_log_appearance_capability_hints(
            logged_cameras=self._appearance_capability_logged,
            camera_name=camera_name,
            flags=flags,
            backend_name=self._appearance.backend_name,
            pphuman_sgie_enabled=self._pphuman_sgie_enabled,
            pphuman_config_exists=self._resolve_pphuman_infer_config().exists(),
            yolo_labels=self._yolo_labels,
            bag_classes=set(BAG_CLASSES),
            face_recognizer_enabled=bool(self.face_recognizer.enabled),
            helmet_enabled=self._helmet_enabled,
        )

    def _remember_context_events(self, camera_name: str, events: List[DetectionEvent]) -> None:
        """카메라별 최근 이벤트를 컨텍스트 저장소에 기록한다."""
        self._context_event_store.remember(camera_name, events)

    def _collect_context_events(
        self,
        camera_name: str,
        current_events: List[DetectionEvent],
    ) -> List[DetectionEvent]:
        """현재 이벤트에 병합할 과거 컨텍스트 이벤트를 수집한다."""
        return self._context_event_store.collect(camera_name, current_events)

    def _run_face_recognition(
        self,
        frame: Any,
        person_events: List[DetectionEvent],
        camera_name: str,
    ) -> List[DetectionEvent]:
        """프레임 내 사람 이벤트에 얼굴 인식 결과를 결합한다."""
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

    def _apply_existing_event_pipeline(
        self, camera_name: str, events: List[DetectionEvent]
    ) -> None:
        """기존 트래킹/필터/구역/얼굴 후처리 파이프라인을 적용한다."""
        self._write_fall_near_miss_review_records(camera_name, events)
        ds_apply_existing_event_pipeline(
            camera_name=camera_name,
            events=events,
            assign_synthetic_object_ids=self._assign_synthetic_object_ids,
            track_manager=self.track_manager,
            violation_filter=self.violation_filter,
            submit_face_work=self._submit_face_work,
            zone_manager=self.zone_manager,
            enqueue_zone_events_cb=self._enqueue_zone_events,
            enqueue_event=self._enqueue_event_or_defer_fall_aux,
            add_filtered_event_count=lambda delta: self._increment_stat("events_filtered", delta),
        )

    def _cleanup_event_filters(self) -> None:
        """디바운서와 위반 누적 필터의 오래된 상태를 정리한다."""
        self._debouncer.cleanup()
        self.violation_filter.cleanup(self._synthetic_track_timeout * 10)

    def _log_tensor_probe_waiting(self, camera_name: str) -> None:
        """텐서 메타는 있으나 유효 객체가 없을 때 상태를 로그로 기록한다."""
        logger.info(
            "[%s] DeepStream tensor meta는 수신 중이나 필터 조건을 통과한 객체가 아직 없습니다.",
            camera_name,
        )

    def _submit_face_work(self, camera_name: str, filtered_events: List[DetectionEvent]) -> None:
        """얼굴/외형 후처리 작업을 백그라운드 워커 큐에 전달한다."""
        self._context_worker.submit(camera_name, filtered_events)

    def _face_worker_loop(self) -> None:
        """얼굴/외형 컨텍스트 워커 루프를 실행한다."""
        self._context_worker.run_loop(self.stop_event)

    def _falldata_aux_worker_loop(self) -> None:
        """borderline 낙상 후보를 보조 모델로 검증하고 confirmed일 때만 발행한다."""
        logger.info("falldata aux 비동기 워커 시작")
        while not self.stop_event.is_set():
            try:
                camera_name, event_payload = self._falldata_aux_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                result, _ = self._fall_shadow_recorder().verify_and_write_aux_record(
                    camera_name,
                    event_payload,
                )
                if self._enqueue_aux_confirmed_fall_event(camera_name, event_payload, result):
                    continue
                if self._should_fail_open_falldata_aux_result(result):
                    self._enqueue_aux_fallback_fall_event(camera_name, event_payload, result)
            except Exception as exc:
                logger.warning("[%s] falldata aux 워커 처리 실패: %s", camera_name, exc)

    def _should_fail_open_falldata_aux_result(self, result: Dict[str, Any]) -> bool:
        falldata_aux = getattr(self, "_falldata_aux", None)
        if falldata_aux is None:
            return False
        return falldata_aux._should_fail_open(result)

    def _enqueue_aux_fallback_fall_event(
        self,
        camera_name: str,
        event_payload: Dict[str, Any],
        result: Dict[str, Any],
    ) -> bool:
        metadata = dict(event_payload.get("metadata") or {})
        metadata.pop("falldata_aux_publish_pending", None)
        metadata["falldata_aux"] = result
        metadata["falldata_aux_confirm_fallback"] = result.get("status")
        queue_item = dict(event_payload)
        queue_item["metadata"] = metadata
        logger.warning(
            "[%s] falldata aux 사용 불가로 borderline 낙상 후보 fail-open 발행: %s",
            camera_name,
            result.get("status"),
        )
        return self._put_event_dict(queue_item, camera_name)

    def _enqueue_event_or_defer_fall_aux(
        self,
        event: DetectionEvent,
        camera_name: str,
    ) -> bool:
        if not self._should_confirm_fall_with_aux_before_publish(event):
            return self._enqueue_event(event, camera_name)

        metadata = dict(event.metadata or {})
        metadata["falldata_aux_publish_pending"] = True
        event.metadata = metadata
        submitted = self._submit_falldata_aux_work(camera_name, [event])
        if submitted is not None:
            logger.info(
                "[%s] borderline fall 후보 aux 확인 대기: object_id=%s score=%s",
                camera_name,
                event.object_id,
                metadata.get("fall_score"),
            )
            return False

        metadata["falldata_aux_confirm_fallback"] = "submit_failed"
        event.metadata = metadata
        logger.warning("[%s] falldata aux 제출 실패 - borderline 후보 fail-open 발행", camera_name)
        return self._enqueue_event(event, camera_name)

    def _should_enqueue_event(self, event: DetectionEvent, camera_name: str) -> bool:
        """동일 이벤트가 프레임마다 MQTT로 발행되지 않도록 제한한다."""
        metadata = event.metadata or {}
        camera_id = str(metadata.get("camera_id") or camera_name)
        object_id = int(event.object_id) if event.object_id is not None else 0
        return self._debouncer.should_send(camera_id, event.event_type.value, object_id)

    def _enqueue_event(self, event: DetectionEvent, camera_name: str) -> bool:
        """디바운싱 검증 후 이벤트를 내부 큐에 적재한다."""
        if not self._should_enqueue_event(event, camera_name):
            return False
        enqueued = self._enqueue_queue_item(event, camera_name)
        if enqueued:
            self._write_fall_event_review_record(camera_name, event)
        return enqueued

    def _add_osd_overlays(
        self,
        batch_meta: Any,
        frame_meta: Any,
        detections: List[Dict[str, Any]],
    ) -> None:
        """탐지 결과를 OSD 오버레이로 프레임 메타에 반영한다."""
        ds_add_osd_overlays(
            pyds_module=pyds,
            batch_meta=batch_meta,
            frame_meta=frame_meta,
            detections=detections,
            min_keypoint_confidence=float(
                os.environ.get("DS_OSD_KEYPOINT_CONFIDENCE", "0.35")
            ),
        )

    def _detections_from_tensor(self, tensor_meta: Any, frame_meta: Any) -> List[Dict[str, Any]]:
        """nvinfer 텐서 메타를 후처리해 공통 탐지 dict 목록으로 변환한다."""

        gie_id = tensor_gie_id(tensor_meta, self._pose_gie_id)
        task = self._task_by_gie.get(gie_id, self._yolo_task)
        labels = self._labels_by_gie.get(gie_id, self._yolo_labels)
        class_ids_filter = self._class_ids_by_gie.get(gie_id, self._yolo_class_ids)
        confidence_threshold = self._confidence_by_gie.get(gie_id, self._yolo_conf_threshold)

        layer = select_yolo_output(tensor_meta, pyds)
        if layer is None:
            return []
        output = layer_to_numpy(layer, pyds)
        if output is None:
            return []

        frame_width = int(getattr(frame_meta, "source_frame_width", 0) or 0)
        frame_height = int(getattr(frame_meta, "source_frame_height", 0) or 0)
        if frame_width <= 0 or frame_height <= 0:
            frame_width = int(os.environ.get("DS_STREAM_WIDTH", "1920"))
            frame_height = int(os.environ.get("DS_STREAM_HEIGHT", "1080"))

        postprocess_started_at = time.perf_counter()
        detections = detections_from_yolo_output(
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
            postprocess_mode=self._yolo_postprocess_mode,
        )
        self._record_yolo_postprocess_timing(time.perf_counter() - postprocess_started_at)
        return detections

    @staticmethod
    def _pphuman_roi_for_detection(
        detection: Mapping[str, Any], frame_meta: Any
    ) -> Tuple[int, int, int, int]:
        x, y, width, height = [int(value) for value in detection.get("box", (0, 0, 0, 0))]
        frame_width = int(getattr(frame_meta, "source_frame_width", 0) or 0)
        frame_height = int(getattr(frame_meta, "source_frame_height", 0) or 0)
        max_width = int(frame_width * float(os.environ.get("DS_PPHUMAN_MAX_ROI_WIDTH_RATIO", "0.65")))
        max_height = int(frame_height * float(os.environ.get("DS_PPHUMAN_MAX_ROI_HEIGHT_RATIO", "1.0")))
        keypoints = [point for point in detection.get("keypoints", []) if len(point) >= 3 and point[2] > 0]
        if frame_width and width > max_width and keypoints:
            xs = [int(point[0]) for point in keypoints]
            margin = max(10, int((max(xs) - min(xs)) * 0.35))
            x = max(0, min(xs) - margin)
            width = min(frame_width - x, max(xs) - min(xs) + margin * 2)
        if frame_height and height > max_height:
            height = max_height
        return x, y, width, height

    def _inject_primary_person_object_meta(
        self, batch_meta: Any, frame_meta: Any, camera_name: str
    ) -> int:
        injected = 0
        user_meta_list = frame_meta.frame_user_meta_list
        while user_meta_list is not None:
            user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
            if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                if tensor_gie_id(tensor_meta, self._pose_gie_id) == self._pose_gie_id:
                    detections = self._filter_detections_for_camera(
                        self._detections_from_tensor(tensor_meta, frame_meta), camera_name
                    )
                    for detection in detections:
                        if self._event_type_for_label(str(detection.get("label", ""))) != EventType.PERSON:
                            continue
                        obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                        x, y, width, height = self._pphuman_roi_for_detection(detection, frame_meta)
                        obj_meta.unique_component_id = self._pose_gie_id
                        obj_meta.class_id = int(detection.get("class_id", 0))
                        obj_meta.obj_label = str(detection.get("label") or "person")
                        obj_meta.confidence = float(detection.get("confidence", 0.0))
                        obj_meta.rect_params.left = float(x)
                        obj_meta.rect_params.top = float(y)
                        obj_meta.rect_params.width = float(width)
                        obj_meta.rect_params.height = float(height)
                        pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)
                        injected += 1
            user_meta_list = getattr(user_meta_list, "next", None)
        return injected

    def _filter_detections_for_camera(
        self,
        detections: List[Dict[str, Any]],
        camera_name: str,
    ) -> List[Dict[str, Any]]:
        """카메라 기능 플래그에 따라 탐지 dict를 필터링한다."""
        return filter_detections_for_camera(
            detections,
            camera_name=camera_name,
            feature_flags_for_camera=self._feature_flags_for_camera,
            event_type_for_label=self._event_type_for_label,
        )

    def _filter_events_for_camera(
        self,
        events: List[DetectionEvent],
        camera_name: str,
    ) -> List[DetectionEvent]:
        """카메라별 기능 설정에 맞지 않는 이벤트를 제거한다."""
        return filter_events_for_camera(
            events,
            camera_name=camera_name,
            feature_flags_for_camera=self._feature_flags_for_camera,
        )

    def _emit_tensor_events(
        self, batch_meta: Any, frame_meta: Any, camera_name: str
    ) -> int:
        """텐서 메타 기반 이벤트 생성/후처리를 수행하고 처리 건수를 반환한다."""
        return emit_tensor_events(
            batch_meta=batch_meta,
            frame_meta=frame_meta,
            camera_name=camera_name,
            pyds_module=pyds,
            detections_from_tensor=self._detections_from_tensor,
            add_osd_overlays=self._add_osd_overlays,
            apply_existing_event_pipeline=self._apply_existing_event_pipeline,
            feature_flags_for_camera=self._feature_flags_for_camera,
            event_type_for_label=self._event_type_for_label,
        )

    def _object_meta_events_from_frame(
        self,
        frame_meta: Any,
        camera_name: str,
    ) -> List[DetectionEvent]:
        """object_meta를 읽어 DetectionEvent 목록으로 변환한다."""
        return object_meta_events_from_frame(
            frame_meta=frame_meta,
            camera_name=camera_name,
            pyds_module=pyds,
            pphuman_sgie_enabled=self._pphuman_sgie_enabled,
            feature_flags_for_camera=self._feature_flags_for_camera,
            decode_pphuman_for_obj=self._decode_pphuman_for_obj,
            event_type_for_label=self._event_type_for_label,
        )

    def _create_output_elements(
        self,
        *,
        rtsp_location: Optional[str] = None,
        element_name_suffix: str = "",
        include_output_queue: bool = False,
    ) -> List[Any]:
        """출력 모드에 맞는 sink 브랜치 엘리먼트 집합을 생성한다."""
        return create_output_elements(
            output_mode=self._output_mode,
            make_element=self._make_element,
            set_optional_property=self._set_optional_property,
            env_int=self._env_int,
            gst_module=Gst,
            create_h264_encoder_elements_fn=self._create_h264_encoder_elements,
            poc_fixer_factory=H264PocFixer,
            rtsp_location=rtsp_location,
            element_name_suffix=element_name_suffix,
            include_output_queue=include_output_queue,
        )

    def _create_h264_encoder_elements(
        self,
        element_name_suffix: str = "",
    ) -> List[Any]:
        """H.264 인코딩 브랜치 엘리먼트를 생성한다."""
        return create_h264_encoder_elements(
            make_element=self._make_element,
            env_int=self._env_int,
            set_optional_property=self._set_optional_property,
            gst_module=Gst,
            element_name_suffix=element_name_suffix,
        )

    def _create_preview_elements(self) -> List[Any]:
        """미리보기용 샘플 추출 브랜치 엘리먼트를 생성한다."""
        return create_preview_elements(
            make_element=self._make_element,
            env_int=self._env_int,
            gst_module=Gst,
            on_preview_sample=self._on_preview_sample,
        )

    def _on_preview_sample(self, sink: Any) -> Any:
        """appsink 샘플을 읽어 카메라별 최신 프레임 캐시에 저장한다."""
        result = process_preview_sample(
            sink=sink,
            preview_store=self._preview_store,
            preview_camera_id=getattr(self, "_preview_camera_id", None),
            cameras=getattr(self, "_cameras", {}),
            gst_module=Gst,
        )
        try:
            camera_id = getattr(self, "_preview_camera_id", None) or next(
                iter(getattr(self, "_cameras", {}).keys()),
                None,
            )
            if camera_id:
                frame = self.get_camera_frame(camera_id, copy_frame=False)
                if frame is not None:
                    self._falldata_aux.add_frame(frame)
        except Exception as exc:
            logger.debug("falldata aux preview frame 추가 실패: %s", exc)
        return result

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
        validate_pipeline_prerequisites(
            deepstream_loaded=_ensure_deepstream_loaded(),
            has_cameras=bool(self._cameras),
            infer_config_exists=_INFER_CONFIG.exists(),
            infer_config_path=_INFER_CONFIG,
        )
        primary_enabled, helmet_enabled, pphuman_enabled = self._inference_topology_signature()

        Gst.init(None)

        pipeline = Gst.Pipeline.new("cctv-deepstream")
        if pipeline is None:
            raise RuntimeError("Gst.Pipeline 생성 실패")

        source_entries = build_source_entries(
            cameras=self._cameras,
            source_backoff_until=self._source_backoff_until,
            now=time.monotonic(),
            normalize_uri=self._normalize_uri,
        )
        if not source_entries:
            raise RuntimeError("DeepStream 파이프라인을 만들 지원 소스가 없습니다.")
        self._preview_camera_id = source_entries[0][1]

        n_cams = len(source_entries)
        output_elements = self._create_output_elements()
        preview_elements = self._create_preview_elements() if self._preview_enabled else []

        elements = create_pipeline_elements_bundle(
            make_element=self._make_element,
            preview_enabled=self._preview_enabled,
            primary_enabled=primary_enabled,
            helmet_enabled=helmet_enabled,
            pphuman_enabled=pphuman_enabled,
            output_elements=output_elements,
            preview_elements=preview_elements,
        )

        configure_pipeline_elements_bundle(
            elements=elements,
            n_cams=n_cams,
            configure_streammux=self._configure_streammux,
            configure_infer_elements=self._configure_infer_elements,
            configure_tracker=self._configure_tracker,
            configure_output_queue=self._configure_output_queue,
        )
        add_pipeline_elements(pipeline, elements)

        probe_element = link_deepstream_pipeline_path(
            elements,
            link_or_raise=self._link_or_raise,
            gst_module=Gst,
            primary_probe_callback=self._on_primary_tensor_probe,
            link_preview_branch=self._link_preview_branch,
            pphuman_gie_id=self._pphuman_gie_id,
            pphuman_infer_config=self._resolve_pphuman_infer_config(),
        )

        attach_camera_sources_batch(
            source_entries=source_entries,
            pad_to_camera=self._pad_to_camera,
            gst_module=Gst,
            pipeline=pipeline,
            streammux=elements.streammux,
            make_element=self._make_element,
            normalize_uri=self._normalize_uri,
            on_source_pad_added=self._on_source_pad_added,
        )

        register_pipeline_runtime_hooks(
            probe_element=probe_element,
            pipeline=pipeline,
            gst_module=Gst,
            on_pad_probe=self._on_pad_probe,
            on_bus_message=self._on_bus_message,
        )

        self._pipeline = pipeline
        self._main_loop = GLib.MainLoop()
        # 실제 빌드된 파이프라인 토폴로지 기록
        self._built_topology = elements.topology()
        # pad_id → camera_id 역매핑 캐시 갱신 (매 프레임 재생성 방지)
        self._pad_to_camera = rebuild_pad_to_camera(self._cameras)
        logger.info(
            "DeepStream 파이프라인 토폴로지: primary=%s, helmet=%s, pphuman=%s",
            *self._built_topology,
        )

    def _stop_runtime_loop(self) -> None:
        """버스 오류/EOS 수신 시 메인 루프와 런타임 종료 플래그를 정리한다."""
        self.stop_event.set()
        if self._main_loop is not None:
            self._main_loop.quit()

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
        return handle_bus_message(
            cameras=self._cameras,
            source_backoff_until=self._source_backoff_until,
            source_last_error=self._source_last_error,
            source_failure_backoff_sec=self._source_failure_backoff_sec,
            message=message,
            gst_module=Gst,
            monotonic_now=time.monotonic,
            request_pipeline_restart_cb=self._request_pipeline_restart,
            stop_runtime_cb=self._stop_runtime_loop,
        )

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
          [ ] DetectionEvent 생성 후 _enqueue_event() 로 디바운싱/큐 적재
          [ ] self._frames_processed += 1
        """
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        self._frames_processed, self._tensor_probe_warned = process_batch_frames(
            batch_meta=batch_meta,
            pyds_module=pyds,
            pad_to_camera=self._pad_to_camera,
            frames_processed=self._frames_processed,
            tensor_probe_warned=self._tensor_probe_warned,
            cleanup_interval=1000,
            cleanup_callback=self._cleanup_event_filters,
            emit_tensor_events_for_frame=self._emit_tensor_events,
            object_meta_events_for_frame=self._object_meta_events_from_frame,
            apply_existing_event_pipeline=self._apply_existing_event_pipeline,
            tensor_warn_log=self._log_tensor_probe_waiting,
        )

        return Gst.PadProbeReturn.OK

    def _publish_loop(self) -> None:
        """event_queue 에서 DetectionEvent 를 꺼내 MQTT 로 발행하는 스레드.

        구현 체크리스트:
          [ ] while self.running: event_queue.get(timeout=1.0)
          [ ] MQTT 토픽: f"{topic_prefix}/{camera_id}/{event.event_type.value}"
          [ ] publish_queue_item() 로 DetectionEvent/dict 발행
          [ ] 성공 시 events_sent, 실패 시 events_failed 증가
          [ ] queue.Empty 예외는 continue 로 처리
        """
        logger.info("MQTT 발행 스레드 시작")
        run_publish_loop(
            is_running=lambda: self.running,
            stop_event=self.stop_event,
            event_queue=self.event_queue,
            topic_prefix=self.config.mqtt.topic_prefix,
            mqtt_publish=self._mqtt_publish,
            event_publisher=self.event_publisher,
            increment_stat=self._increment_stat,
        )
        logger.info("MQTT 발행 스레드 종료")

    def print_stats(self) -> None:
        """현재 누적 처리 통계를 로그 한 줄로 출력한다."""
        stats = self.get_stats()
        logger.info(
            "DeepStream stats: frames=%s frame_dropped=%s "
            "events_detected=%s sent=%s filtered=%s event_dropped=%s failed=%s cameras=%s "
            "yolo_postprocess=%s avg_ms=%.3f max_ms=%.3f calls=%s",
            stats["frames_processed"],
            stats["frames_dropped"],
            stats["events_detected"],
            stats["events_sent"],
            stats["events_filtered"],
            stats["events_dropped"],
            stats["events_failed"],
            stats["cameras"],
            stats["yolo_postprocess_mode"],
            stats["yolo_postprocess_avg_ms"],
            stats["yolo_postprocess_max_ms"],
            stats["yolo_postprocess_calls"],
        )


    def release_all_cameras(self) -> None:
        """호환 인터페이스용으로 전체 카메라 처리를 중지한다."""
        self.stop()
