"""
processor.py - 실시간 CCTV 객체 감지 프로세서
다중 카메라 처리, RTSP 재연결, 이벤트 필터링 및 서버 전송

클래스 구성:
  ProcessorStats      - 처리 통계 DTO
  _EventDebouncer     - 이벤트 디바운싱 + 로컬 백업  (VideoProcessor 내부용)
  _DisplayGrid        - 다중 카메라 그리드 디스플레이 (VideoProcessor 내부용)
  _CameraRegistry     - 카메라·스레드·재시도 큐 관리  (VideoProcessor 내부용)
  _InferencePipeline  - AI 추론·구역 탐지·이벤트 큐 처리 (VideoProcessor 내부용)
  VideoProcessor      - 파이프라인 오케스트레이터 (공개 API)

[God Class 대응]
  VideoProcessor 가 담당하던 추론/이벤트 로직을 _InferencePipeline 으로 분리.
  VideoProcessor 는 라이프사이클(start/stop), 카메라 관리(add/remove),
  공개 API(get_stats/get_camera_status 등) 만 담당한다.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import cv2

from ..config import AppConfig
from ..protocols.mqtt_publisher import MqttEventPublisher
from ..utils.camera_input import RTSPCamera
from ..utils.dataset_collector import DatasetCollector
from ..utils.zone_detection import ZoneEvent, ZoneManager
from ..utils.zone_drawer import ZoneDrawer
from ._display_event_mapper import DisplayEventMapper
from ._display_grid import _DisplayGrid
from ..utils.visualizer import draw_events
from ._camera_registry import _CameraRegistry
from ._inference_pipeline import _InferencePipeline
from .ai.analyzer import AIAnalyzer
from .base_processor import BaseProcessor
from .event_filters import CumulativeViolationFilter, TrackManager
from .events import DetectionEvent, EventType

logger = logging.getLogger(__name__)


# ===========================================================================
# 통계 DTO
# ===========================================================================


@dataclass
class ProcessorStats:
    """처리 통계 추적기."""

    frames_processed: int = 0
    frames_dropped: int = 0
    events_detected: int = 0
    events_dropped: int = 0
    events_sent: int = 0
    events_filtered: int = 0
    events_failed: int = 0
    inference_errors: int = 0
    camera_errors: int = 0
    start_time: float = field(default_factory=time.time)
    camera_count: int = 0
    total_inference_time: float = 0.0
    inference_count: int = 0

    def get_fps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.frames_processed / elapsed if elapsed > 0 else 0

    def get_avg_inference_time(self) -> float:
        """평균 추론 시간 (ms)."""
        if self.inference_count == 0:
            return 0.0
        return (self.total_inference_time / self.inference_count) * 1000

    def snapshot(self) -> Dict:
        """원시 통계 스냅샷 반환 (파생값 제외)."""
        return asdict(self)

    @staticmethod
    def with_derived_stats(stats: Dict, now: Optional[float] = None) -> Dict:
        """원시 통계에 파생값(fps/uptime/avg_inference_ms) 추가."""
        if now is None:
            now = time.time()
        start_time = stats.get("start_time", now)
        elapsed = max(0.0, now - start_time)
        frames_processed = stats.get("frames_processed", 0)
        inference_count = stats.get("inference_count", 0)
        total_inference_time = stats.get("total_inference_time", 0.0)
        stats["fps"] = round(frames_processed / elapsed, 2) if elapsed > 0 else 0
        stats["uptime_seconds"] = round(elapsed, 2)
        stats["avg_inference_ms"] = (
            round((total_inference_time / inference_count) * 1000, 2)
            if inference_count > 0
            else 0.0
        )
        return stats

    def to_dict(self) -> Dict:
        return self.with_derived_stats(self.snapshot())


# ===========================================================================
# _EventDebouncer  (내부용)
# ===========================================================================


class _EventDebouncer:
    """이벤트 중복 전송 방지·로컬 백업·만료 정리.

    VideoProcessor 에서만 사용한다.
    """

    def __init__(self, config: AppConfig, increment_stat) -> None:
        self._config = config
        self._increment_stat = increment_stat
        self._last_events: Dict[Tuple[str, str, int], float] = {}
        self._lock = Lock()
        # 낙상 지속 감지용 상태 추적 (camera_id, object_id) 기준
        self._fall_first_seen: Dict[Tuple[str, int], float] = {}  # 낙상 최초 감지 시각
        self._fall_last_seen: Dict[Tuple[str, int], float] = {}   # 낙상 마지막 감지 시각
        self._fall_alerted: Dict[Tuple[str, int], float] = {}     # 낙상 알림 마지막 전송 시각
        # 헬멧 미착용(head) 상태 추적 (camera_id, object_id) 기준
        self._head_last_seen: Dict[Tuple[str, int], float] = {}   # head 마지막 감지 시각
        self._head_alerted: Dict[Tuple[str, int], float] = {}     # head 마지막 전송 시각

    def should_send(self, camera_id: str, event_type: str, object_id: int) -> bool:
        """중복 전송을 방지하기 위해 이벤트를 보내야 하는지 반환한다."""
        if not self._config.events.debounce_enabled:
            return True

        # 낙상: 10초 이상 지속 감지되어야 전송 (매 프레임 전송 방지)
        if event_type == "fall_detected":
            return self._should_send_fall(camera_id, object_id)

        # 헬멧 미착용: 상태 변화 감지 + 최소 재전송 간격 적용
        if event_type == "head":
            return self._should_send_head(camera_id, object_id)

        key = (camera_id, event_type, object_id)
        now = time.time()
        with self._lock:
            last_time = self._last_events.get(key, 0)
            if now - last_time >= self._config.events.debounce_seconds:
                self._last_events[key] = now
                return True
            self._increment_stat("events_filtered")
            return False

    def _should_send_head(self, camera_id: str, object_id: int) -> bool:
        """헬멧 미착용(head) 이벤트 전송 여부 판단.

        - gap_reset_seconds 이상 미감지 후 재등장 → 상태 변화로 보고 즉시 전송
        - 연속 감지 중에는 resend_cooldown 간격으로만 재전송
        """
        cfg = self._config.events
        key = (camera_id, object_id)
        now = time.time()
        with self._lock:
            last_seen = self._head_last_seen.get(key, 0)
            last_alert = self._head_alerted.get(key, 0)
            is_state_change = (now - last_seen) > cfg.head_gap_reset_seconds
            self._head_last_seen[key] = now

            if is_state_change or (now - last_alert) >= cfg.head_resend_cooldown:
                self._head_alerted[key] = now
                if is_state_change:
                    logger.info(
                        "[%s] 헬멧 미착용 재등장 감지 → 즉시 전송 (object_id=%s)",
                        camera_id, object_id,
                    )
                return True

            self._increment_stat("events_filtered")
            return False

    def _should_send_fall(self, camera_id: str, object_id: int) -> bool:
        """낙상이 sustained_seconds 이상 지속될 때만 True 반환.

        - gap_reset_seconds 이상 낙상이 끊기면 지속 타이머 초기화
        - 전송 후 resend_cooldown 동안 재전송 억제
        """
        cfg = self._config.events
        key = (camera_id, object_id)
        now = time.time()
        with self._lock:
            last_seen = self._fall_last_seen.get(key, 0)
            # 낙상이 gap_reset_seconds 이상 끊겼으면 타이머 리셋
            if now - last_seen > cfg.fall_gap_reset_seconds:
                self._fall_first_seen[key] = now
            self._fall_last_seen[key] = now

            duration = now - self._fall_first_seen.get(key, now)
            if duration < cfg.fall_sustained_seconds:
                # 아직 지속 시간 미달
                self._increment_stat("events_filtered")
                return False

            # 지속 시간 충족 — 재전송 쿨다운 확인
            last_alert = self._fall_alerted.get(key, 0)
            if now - last_alert < cfg.fall_resend_cooldown:
                self._increment_stat("events_filtered")
                return False

            self._fall_alerted[key] = now
            logger.info(
                "[%s] 낙상 지속 %.1f초 확인 → 이벤트 전송 (object_id=%s)",
                camera_id, duration, object_id,
            )
            return True

    def cleanup(self, max_age_hours: Optional[int] = None) -> int:
        """보존 기간이 지난 이벤트 키를 제거하고 제거 수를 반환한다."""
        if max_age_hours is None:
            max_age_hours = self._config.events.event_retention_hours
        cutoff = time.time() - max_age_hours * 3600
        # 낙상/헬멧 추적 딕셔너리 정리: 동일 보존 기준 적용
        with self._lock:
            before = len(self._last_events)
            self._last_events     = {k: v for k, v in self._last_events.items()     if v > cutoff}
            self._fall_first_seen = {k: v for k, v in self._fall_first_seen.items() if v > cutoff}
            self._fall_last_seen  = {k: v for k, v in self._fall_last_seen.items()  if v > cutoff}
            self._fall_alerted    = {k: v for k, v in self._fall_alerted.items()    if v > cutoff}
            self._head_last_seen  = {k: v for k, v in self._head_last_seen.items()  if v > cutoff}
            self._head_alerted    = {k: v for k, v in self._head_alerted.items()    if v > cutoff}
            return before - len(self._last_events)

    def save_locally(self, event_data: Dict) -> None:
        """큐 포화 시 이벤트를 로컬 JSON 파일로 백업한다.

        파일명에 나노초 타임스탬프를 사용하여 같은 초에 여러 이벤트가
        도착해도 덮어쓰기가 발생하지 않도록 한다.
        """
        try:
            backup_dir = os.path.join(os.getcwd(), "event_backup")
            os.makedirs(backup_dir, exist_ok=True)
            ts_ns = time.time_ns()          # 나노초 — 충돌 확률 사실상 0
            cam_id = event_data.get('camera_id', 'unknown')
            filename = f"event_{ts_ns}_{cam_id}.json"
            filepath = os.path.join(backup_dir, filename)
            with open(filepath, "w", encoding="utf-8") as fp:
                json.dump(event_data, fp, ensure_ascii=False, indent=2)
            logger.debug("이벤트 로컬 저장: %s", filepath)
        except Exception as exc:
            logger.error("로컬 저장 실패: %s", exc)

from ._adaptive_governor import _AdaptiveGovernor  # noqa: F401 — 하위 호환 재내보내기


# ===========================================================================
# VideoProcessor  (공개 API)
# ===========================================================================


class VideoProcessor(BaseProcessor):
    """AI 추론을 사용한 다중 카메라 비디오 처리 파이프라인.

    내부 헬퍼:
        _debouncer (_EventDebouncer) - 이벤트 디바운싱
        _display   (_DisplayGrid)    - 그리드 디스플레이
        _cams      (_CameraRegistry) - 카메라 생명주기

    모델 공유:
        _model_pool  -  (helmet_path, person_path, pose_path) 튜플 키로
                        로드된 YOLO 모델 객체를 캐시한다.
                        동일 모델 경로를 사용하는 카메라들은 GPU 메모리를
                        공유하여 중복 로드를 방지한다.
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._analyzers: Dict[str, AIAnalyzer] = {}
        # 카메라별 AI 모델 플래그 (ai_models 설정 딕셔너리에서 파싱)
        self._camera_ai_flags: Dict[str, Dict[str, bool]] = {}

        # ── 모델 공유 풀 ─────────────────────────────────────────────
        # key: (helmet_path, person_path, pose_path) 정규화 튜플
        # value: {"helmet": model|None, "person": model|None, "pose": model|None}
        # 같은 경로 조합의 카메라들은 동일 모델 객체를 참조 → GPU 메모리 절약
        self._model_pool: Dict[tuple, Dict[str, object]] = {}
        self._model_pool_lock = Lock()

        # ── 라이프사이클 ─────────────────────────────────────────────
        self.running    = False
        self.stop_event = Event()

        # ── 내부 헬퍼 ────────────────────────────────────────────────
        self._debouncer = _EventDebouncer(config, self._increment_stat)
        self._display   = _DisplayGrid(get_fps=lambda: self.stats.get_fps())
        self._cams      = _CameraRegistry(
            config,
            stop_event=self.stop_event,
            is_running=lambda: self.running,
        )

        # ── MQTT 퍼블리셔 ─────────────────────────────────────────────
        self.event_publisher = MqttEventPublisher(
            broker=config.mqtt.broker,
            port=config.mqtt.port,
            topic_prefix=config.mqtt.topic_prefix,
            client_id_prefix=config.mqtt.client_id_prefix,
            qos=config.mqtt.qos,
            retain=config.mqtt.retain,
        )

        # ── 이벤트 큐 ────────────────────────────────────────────────
        self.event_queue = Queue(maxsize=config.events.queue_max_size * 3)

        # ── 통계 ─────────────────────────────────────────────────────
        self.stats       = ProcessorStats()
        self._stats_lock = Lock()

        # ── 추적 / 누적 필터 ─────────────────────────────────────────
        self.track_manager = TrackManager(
            track_timeout=0.5,
            track_iou_threshold=0.5,
            min_track_frames=config.processing.min_track_frames,
        )
        self.violation_filter = CumulativeViolationFilter(
            history_max_size=config.processing.detection_history_size,
            violation_threshold=config.processing.violation_threshold,
            enabled=config.processing.cumulative_detection_enabled,
        )
        self._history_timeout = max(10.0, self.track_manager.track_timeout * 10)

        # ── Zone / Dataset ────────────────────────────────────────────
        self.zone_manager: Optional[ZoneManager] = None
        if config.zone_detection:
            try:
                self.zone_manager = ZoneManager(config.zones_config)
                logger.info("구역 감지 활성화됨")
            except Exception as exc:
                logger.warning("구역 로딩 실패: %s", exc)

        self.dataset_collector: Optional[DatasetCollector] = None
        if config.collect_dataset:
            os.makedirs(config.dataset_dir, exist_ok=True)
            try:
                self.dataset_collector = DatasetCollector(
                    output_dir=config.dataset_dir, format="yolo"
                )
                logger.info("데이터셋 수집 활성화됨")
            except Exception as exc:
                logger.warning("데이터셋 수집기 초기화 실패: %s", exc)

        self.cleanup_interval = config.events.cleanup_interval

        # ── 구역 점유 상태 (카메라별 object_id 집합) ──────────────────
        # 플리커링 방지: zone_entered/dwelling 시 추가, zone_exited 시 제거
        self._zone_in_objects: Dict[str, set] = {}
        self._display_event_mapper = DisplayEventMapper(self._zone_in_objects)

        # ── 탐지 스냅샷 (카메라별 최신 탐지 결과 + 타임스탬프) ──────────
        self._latest_snapshots: Dict[str, dict] = {}
        self._snapshot_lock = Lock()

        # ── 추론 파이프라인 (God Class 분리) ─────────────────────────
        self._pipeline = _InferencePipeline(
            analyzers=self._analyzers,
            camera_ai_flags=self._camera_ai_flags,
            track_manager=self.track_manager,
            violation_filter=self.violation_filter,
            debouncer=self._debouncer,
            event_queue=self.event_queue,
            zone_manager=self.zone_manager,
            dataset_collector=self.dataset_collector,
            display=self._display,
            snapshot_store=self._latest_snapshots,
            snapshot_lock=self._snapshot_lock,
            zone_in_objects=self._zone_in_objects,
            increment_stat=self._increment_stat,
            add_inference_metrics=self._add_inference_metrics,
            display_enabled=config.display,
        )

        # ── 워커 스레드 핸들 ─────────────────────────────────────────
        self.sender_thread:        Optional[Thread] = None
        self.cleanup_thread:       Optional[Thread] = None
        self.display_thread:       Optional[Thread] = None
        self._camera_retry_thread: Optional[Thread] = None
        self._governor_thread:     Optional[Thread] = None

        # ── 동적 성능 조율기 ─────────────────────────────────────────
        # device를 전달하여 CPU/GPU에 맞는 임계값 및 imgsz 단계를 자동 설정
        self._governor = _AdaptiveGovernor(config, self.stats, device=config.detection.device)

    # ------------------------------------------------------------------
    # 하위 호환 프로퍼티
    # ------------------------------------------------------------------

    def set_zone_drawer(self, drawer: ZoneDrawer) -> None:
        """ZoneDrawer 를 디스플레이 그리드에 연결한다.

        processor.start() 호출 *전* 또는 *후* 어느 시점이든 호출 가능하다.
        """
        self._display.set_drawer(drawer)

    # ------------------------------------------------------------------

    @property
    def cameras(self) -> Dict[str, RTSPCamera]:
        return self._cams.cameras

    @property
    def camera_threads(self) -> Dict[str, Thread]:
        return self._cams.camera_threads

    @property
    def inference_threads(self) -> Dict[str, Thread]:
        return self._cams.inference_threads

    @property
    def frame_queues(self) -> Dict[str, Queue]:
        return self._cams.frame_queues

    # ------------------------------------------------------------------
    # 통계 헬퍼
    # ------------------------------------------------------------------

    def _increment_stat(self, field_name: str, delta: int = 1) -> int:
        """통계 카운터를 스레드 안전하게 증가한다."""
        with self._stats_lock:
            current = getattr(self.stats, field_name)
            new_val = current + delta
            setattr(self.stats, field_name, new_val)
            return new_val

    def _add_inference_metrics(self, inference_time: float) -> None:
        """추론 성능 통계를 스레드 안전하게 갱신한다."""
        with self._stats_lock:
            self.stats.total_inference_time += inference_time
            self.stats.inference_count += 1

    def _set_camera_count(self, count: int) -> None:
        with self._stats_lock:
            self.stats.camera_count = count

    # ------------------------------------------------------------------
    # 모델 팩토리
    # ------------------------------------------------------------------

    def _build_analyzer(self, model_paths: Optional[Dict[str, str]] = None) -> AIAnalyzer:
        """AIAnalyzer 인스턴스를 생성한다.

        model_paths가 제공되면 해당 모델만 카메라별 로드,
        없는 항목은 전역 설정(config.models)로 폴백한다.

        model_paths 키: "helmet" | "person" | "pose"

        동일한 모델 경로 조합은 _model_pool에서 캐시된 모델 객체를 재사용하여
        GPU 메모리 중복 로드를 방지한다.
        """
        mp = model_paths or {}
        helmet_path = mp.get("helmet") or self.config.models.helmet_model or ""
        person_path = mp.get("person") or self.config.models.person_model or ""
        pose_path   = mp.get("pose")   or self.config.models.pose_model   or ""
        if pose_path:
            person_path = ""

        # 정규화 키: 절대 경로 기준으로 동일 파일 여부 판단
        def _norm(p: str) -> str:
            return str(Path(p).resolve()) if p else ""

        pool_key = (_norm(helmet_path), _norm(person_path), _norm(pose_path))

        with self._model_pool_lock:
            cached = self._model_pool.get(pool_key)

        if cached is not None:
            # 동일 경로 조합 → 캐시된 모델 객체 재사용 (GPU 메모리 공유)
            logger.info(
                "모델 풀 캐시 히트 — 동일 모델 공유 (helmet=%s, pose=%s)",
                Path(helmet_path).name if helmet_path else "없음",
                Path(pose_path).name   if pose_path   else "없음",
            )
            analyzer = AIAnalyzer(
                helmet_model_path=helmet_path or None,
                person_model_path=person_path or None,
                pose_model_path=pose_path or None,
                confidence_threshold=self.config.detection.pose_confidence,
                device=self.config.detection.device,
                fall_height_ratio=self.config.detection.fall_height_ratio,
                appearance_backend=self.config.appearance.backend,
                appearance_model_path=self.config.appearance.model_path,
                appearance_label_map_path=self.config.appearance.label_map_path,
                appearance_runtime=self.config.appearance.runtime,
                appearance_input_size=self.config.appearance.input_size,
                appearance_score_threshold=self.config.appearance.score_threshold,
                appearance_bbox_expand_ratio=self.config.appearance.bbox_expand_ratio,
            )
            # 새 모델 로드 없이 캐시된 모델 객체를 직접 주입
            analyzer.helmet_model = cached["helmet"]
            analyzer.person_model = cached["person"]
            analyzer.pose_model   = cached["pose"]
        else:
            # 최초 로드 → 모델 풀에 저장
            analyzer = AIAnalyzer(
                helmet_model_path=helmet_path or None,
                person_model_path=person_path or None,
                pose_model_path=pose_path or None,
                confidence_threshold=self.config.detection.pose_confidence,
                device=self.config.detection.device,
                fall_height_ratio=self.config.detection.fall_height_ratio,
                appearance_backend=self.config.appearance.backend,
                appearance_model_path=self.config.appearance.model_path,
                appearance_label_map_path=self.config.appearance.label_map_path,
                appearance_runtime=self.config.appearance.runtime,
                appearance_input_size=self.config.appearance.input_size,
                appearance_score_threshold=self.config.appearance.score_threshold,
                appearance_bbox_expand_ratio=self.config.appearance.bbox_expand_ratio,
            )
            with self._model_pool_lock:
                self._model_pool[pool_key] = {
                    "helmet": analyzer.helmet_model,
                    "person": analyzer.person_model,
                    "pose":   analyzer.pose_model,
                }
            logger.info(
                "모델 풀 신규 등록 (helmet=%s, pose=%s)",
                Path(helmet_path).name if helmet_path else "없음",
                Path(pose_path).name   if pose_path   else "없음",
            )

        analyzer.helmet_threshold = self.config.detection.helmet_confidence
        analyzer.person_threshold = self.config.detection.person_confidence
        analyzer.pose_threshold   = self.config.detection.pose_confidence

        # TRT .engine 사용 시 governor imgsz 자동 조정 잠금
        # (engine은 컴파일 시 imgsz 고정 → 런타임에 변경 불가)
        if any(
            p and str(p).endswith(".engine")
            for p in (analyzer.pose_model_path, analyzer.helmet_model_path, analyzer.person_model_path)
        ):
            self._governor.lock_imgsz()

        return analyzer

    @staticmethod
    def _normalize_model_flags(flags: Mapping[str, object]) -> Dict[str, bool]:
        """모델 on/off 플래그를 정규화한다.

        helmet 모델은 사람 ROI가 필요하므로 pose 모델이 자동으로 함께 활성화된다.
        appearance 모델은 사람 감지가 필요하므로 pose 또는 person이 함께 활성화된다.
        """
        use_pose = bool(flags.get("use_pose", flags.get("pose", False)))
        use_helmet = bool(flags.get("use_helmet", flags.get("helmet", False)))
        use_person = bool(flags.get("use_person", flags.get("person", False)))
        use_face = bool(flags.get("use_face", flags.get("face", False)))
        use_appearance = bool(flags.get("use_appearance", flags.get("appearance", False)))

        if use_helmet or use_face:
            use_pose = True

        # appearance는 사람 bbox가 필요 — pose나 person 중 하나는 활성화
        if use_appearance and not use_pose and not use_person:
            use_pose = True

        return {
            "use_helmet": use_helmet,
            "use_pose": use_pose,
            "use_person": use_person,
            "use_face": use_face,
            "use_appearance": use_appearance,
        }

    @classmethod
    def _flags_to_detection_modes(cls, flags: Mapping[str, object]) -> List[str]:
        """모델 플래그를 cameras.json 호환 detections 리스트로 변환한다."""
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

    @classmethod
    def _parse_detections(
        cls,
        detections: Optional[Union[List[str], Mapping[str, object]]],
    ) -> Dict[str, bool]:
        """cameras.json 의 detections/model_settings 값을 run_inference 플래그로 변환한다.

        None 또는 비어있으면 전체 모델 활성화 (하위 호환 유지).

        지원 값:
            "helmet"    →  안전모 미착용 감지
            "fall"      →  낙상 감지 (pose 모델)
            "intrusion" →  무단침입 / 좌리 감지
            "person"    →  사람 감지만
            "face"      →  얼굴 인식 (사람 ROI 기반)
        """
        if isinstance(detections, Mapping):
            return cls._normalize_model_flags(detections)

        if not detections:
            return {"use_helmet": True, "use_pose": True, "use_person": False, "use_face": False, "use_appearance": False}

        modes = {d.lower() for d in detections}
        use_pose   = bool(modes & {"fall", "intrusion", "person"}) or bool(modes & {"helmet", "face"})
        use_helmet = "helmet" in modes
        use_person = False
        use_face = "face" in modes
        use_appearance = "appearance" in modes

        # appearance는 사람 bbox가 필요
        if use_appearance and not use_pose:
            use_pose = True

        return {
            "use_helmet": use_helmet, "use_pose": use_pose,
            "use_person": use_person, "use_face": use_face,
            "use_appearance": use_appearance,
        }

    def get_camera_model_settings(self, camera_id: str) -> Optional[Dict[str, bool]]:
        """카메라의 현재 모델 on/off 상태를 반환한다."""
        flags = self._camera_ai_flags.get(camera_id)
        if flags is None:
            return None
        return {
            "use_helmet": bool(flags.get("use_helmet", False)),
            "use_pose": bool(flags.get("use_pose", False)),
            "use_person": bool(flags.get("use_person", False)),
            "use_face": bool(flags.get("use_face", False)),
            "use_appearance": bool(flags.get("use_appearance", False)),
        }

    def _get_face_recognizer(self):
        """실행 중인 분석기가 있으면 그 face_recognizer를, 없으면 독립 인스턴스를 반환한다."""
        analyzer = next(iter(self._analyzers.values()), None)
        if analyzer is not None:
            return analyzer.face_recognizer
        from ..utils.face_recognition import FaceRecognitionEngine
        return FaceRecognitionEngine()

    def list_registered_faces(self) -> List[Dict[str, str]]:
        """등록 얼굴 목록을 반환한다."""
        return self._get_face_recognizer().list_faces()

    def register_face(
        self,
        name: str,
        phone: str,
        image_base64: str,
        filename: Optional[str] = None,
        department: Optional[str] = None,
        position: Optional[str] = None,
        employee_id: Optional[str] = None,
        hired_at: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict[str, str]:
        """새 얼굴을 등록하고 모든 분석기의 얼굴 갤러리를 갱신한다."""
        kwargs = dict(
            name=name,
            phone=phone,
            image_base64=image_base64,
            filename=filename,
            department=department,
            position=position,
            employee_id=employee_id,
            hired_at=hired_at,
            note=note,
        )
        entry = self._get_face_recognizer().register_face(**kwargs)
        self.reload_face_gallery()
        return entry

    def delete_face(self, face_id: str) -> bool:
        """등록 얼굴을 삭제하고 모든 분석기의 얼굴 갤러리를 갱신한다."""
        deleted = self._get_face_recognizer().delete_face(face_id)

        if deleted:
            self.reload_face_gallery()
        return deleted

    def reload_face_gallery(self) -> None:
        """모든 분석기의 등록 얼굴 갤러리를 다시 읽는다."""
        for analyzer in self._analyzers.values():
            try:
                analyzer.face_recognizer.reload_gallery()
            except Exception as exc:
                logger.warning("얼굴 갤러리 리로드 실패: %s", exc)

    def update_camera_model_settings(
        self,
        camera_id: str,
        model_settings: Mapping[str, object],
        cameras_config_path: Optional[str] = None,
    ) -> Optional[Dict[str, bool]]:
        """카메라의 모델 on/off 상태를 갱신하고 필요 시 cameras.json 에 저장한다."""
        if camera_id not in self._camera_ai_flags:
            return None

        normalized = self._normalize_model_flags(model_settings)
        self._camera_ai_flags[camera_id] = normalized

        if cameras_config_path:
            self._save_camera_model_settings(camera_id, normalized, cameras_config_path)

        logger.info("[%s] 모델 설정 업데이트: %s", camera_id, normalized)
        return dict(normalized)

    def _save_camera_model_settings(
        self,
        camera_id: str,
        model_settings: Mapping[str, object],
        cameras_config_path: str,
    ) -> None:
        """cameras.json 에 모델 설정과 detections 목록을 함께 저장한다."""
        normalized = self._normalize_model_flags(model_settings)
        detections = self._flags_to_detection_modes(normalized)

        with open(cameras_config_path, "r", encoding="utf-8") as f:
            cameras = json.load(f)

        updated = False
        for cam in cameras:
            if cam.get("id") != camera_id:
                continue
            cam["model_settings"] = normalized
            cam["detections"] = detections
            updated = True
            break

        if not updated:
            raise KeyError(f"camera_id '{camera_id}' not found in cameras config")

        with open(cameras_config_path, "w", encoding="utf-8") as f:
            json.dump(cameras, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 카메라 관리 (공개 API)
    # ------------------------------------------------------------------

    def add_camera(
        self,
        camera_id: str,
        source: Union[str, int],
        *,
        detections: Optional[List[str]] = None,
        model_paths: Optional[Dict[str, str]] = None,
        zones_data: Optional[List[Dict]] = None,
    ) -> bool:
        """처리 파이프라인에 카메라를 추가한다."""
        if camera_id in self._cams.cameras:
            logger.warning("[%s] 이미 등록된 카메라입니다", camera_id)
            return False

        camera = RTSPCamera(camera_id, source, self.config)
        if not camera.connect():
            logger.error("카메라 연결 실패: %s", camera_id)
            return False

        self._cams.register(camera_id, camera)
        self._set_camera_count(self._cams.count)
        logger.info("카메라 추가됨: %s", camera_id)

        try:
            self._analyzers[camera_id] = self._build_analyzer(model_paths)
            if model_paths:
                logger.info("[%s] 카메라별 모델 경로: %s", camera_id, model_paths)
        except Exception as exc:
            logger.error("[%s] AIAnalyzer 초기화 실패: %s", camera_id, exc)
            self.remove_camera(camera_id)
            return False

        self._camera_ai_flags[camera_id] = self._parse_detections(detections)
        logger.info("[%s] 감지 항목: %s", camera_id, self._camera_ai_flags[camera_id])

        if zones_data:
            if self.zone_manager is None:
                try:
                    from ..utils.zone_detection import ZoneManager
                    self.zone_manager = ZoneManager(self.config.zones_config)
                    logger.info("[%s] zone_manager on-demand 초기화", camera_id)
                except Exception as exc:
                    logger.warning("[%s] zone_manager 초기화 실패: %s", camera_id, exc)
            if self.zone_manager is not None:
                try:
                    self.zone_manager.load_zones(camera_id, zones_data)
                    # _InferencePipeline에 최신 zone_manager 동기화 (on-demand 생성 시 반영)
                    self._pipeline._zone_manager = self.zone_manager
                except Exception as exc:
                    logger.warning("[%s] 구역 로딩 실패: %s", camera_id, exc)
        elif self.zone_manager:
            try:
                self.zone_manager.load_zones(camera_id, None)
                self._pipeline._zone_manager = self.zone_manager
            except Exception as exc:
                logger.warning("[%s] 구역 로딩 실패: %s", camera_id, exc)

        return True

    def remove_camera(self, camera_id: str) -> None:
        """처리 파이프라인에서 카메라를 제거한다."""
        if camera_id not in self._cams.cameras:
            return
        logger.info("카메라 제거 중: %s", camera_id)
        self._cams.unregister(camera_id)
        self.track_manager.remove_camera(camera_id)
        self.violation_filter.purge(camera_id)
        self._analyzers.pop(camera_id, None)
        self._camera_ai_flags.pop(camera_id, None)
        self._zone_in_objects.pop(camera_id, None)
        self._set_camera_count(self._cams.count)
        logger.info("카메라 제거됨: %s", camera_id)

    def update_zones(
        self,
        camera_id: str,
        zones_data: List[Dict],
        cameras_config_path: Optional[str] = None,
    ) -> bool:
        """카메라의 위험 구역을 업데이트하고 설정 파일에 저장한다.

        매개변수:
            camera_id: 카메라 ID
            zones_data: 구역 정의 리스트 [{'id': ..., 'name': ..., 'polygon': [...]}]
            cameras_config_path: cameras.json 경로 (없으면 zones_config.json에 저장)

        반환값:
            성공하면 True
        """
        if not self.zone_manager:
            logger.warning("[%s] zone_manager가 비활성화되어 있습니다", camera_id)
            return False
        try:
            self.zone_manager.save_zones(camera_id, zones_data, cameras_config_path)
            # 삭제된 구역의 잔상(zone_in_objects) 즉시 제거
            self._zone_in_objects.pop(camera_id, None)
            return True
        except Exception as exc:
            logger.error("[%s] 구역 업데이트 실패: %s", camera_id, exc)
            return False

    def enqueue_camera_retry(
        self, camera_id: str, source: Any, delay_seconds: float = 30.0
    ) -> None:
        """연결 실패한 카메라를 백그라운드 재시도 큐에 등록한다."""
        self._cams.enqueue_retry(camera_id, source, delay_seconds)

    # ------------------------------------------------------------------
    # 재연결 워커
    # ------------------------------------------------------------------

    def _camera_retry_worker(self) -> None:
        """연결 실패 카메라를 백그라운드에서 주기적으로 재시도한다."""
        while self.running and not self.stop_event.is_set():
            for camera_id, source, _ in self._cams.poll_ready_retries():
                if camera_id in self._cams.cameras:
                    continue
                logger.info("[%s] 백그라운드 재연결 시도 중...", camera_id)
                if self.add_camera(camera_id, source):
                    logger.info("[%s] 백그라운드 재연결 성공", camera_id)
                    if self.running:
                        self._cams.start_threads(
                            camera_id, self._process_camera, self._process_inference
                        )
                else:
                    self._cams.enqueue_retry(
                        camera_id, source, delay_seconds=min(300, 60)
                    )
            time.sleep(5.0)

    # ------------------------------------------------------------------
    # 카메라 스레드 / 추론 스레드
    # ------------------------------------------------------------------

    def _process_camera(self, camera_id: str, camera: RTSPCamera) -> None:
        """프레임 획득 루프 — AI 추론은 별도 스레드에서 처리한다."""
        while self.running and not self.stop_event.is_set():
            stop_flag = self._cams.stop_flag(camera_id)
            if stop_flag and stop_flag.is_set():
                break
            start = time.monotonic()
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                time.sleep(self.config.processing.camera_reconnect_delay)
                continue
            fq = self._cams.frame_queues.get(camera_id)
            if fq is not None:
                while fq.full():
                    try:
                        fq.get_nowait()
                        self._increment_stat("frames_dropped")
                    except Empty:
                        break
                fq.put_nowait(frame.copy())
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, 1.0 / self.config.detection.target_fps - elapsed))

    def _record_detection_snapshot(
        self, camera_id: str, events: List[DetectionEvent]
    ) -> None:
        """웹/API에서 조회할 최신 탐지 스냅샷을 저장한다."""
        with self._snapshot_lock:
            self._latest_snapshots[camera_id] = {
                "timestamp": time.time(),
                "detections": [e.to_dict() for e in events],
            }

    def _save_zone_event_snapshots(
        self, camera_id: str, zone_events: List[ZoneEvent]
    ) -> None:
        """의미 있는 구역 이벤트에 대해서만 스냅샷을 저장한다."""
        if not zone_events:
            return

        snapshot_dir = getattr(self, "snapshot_dir", "snapshots")
        snapshot_event_types = {"zone_entered", "zone_object_detected", "crowd_warning"}
        for zone_event in zone_events:
            if zone_event.event_type.value not in snapshot_event_types:
                continue
            self.save_event_snapshot(
                camera_id,
                event_type=zone_event.event_type.value,
                zone_id=getattr(zone_event, "zone_id", ""),
                snapshot_dir=snapshot_dir,
            )

    def _build_display_events(
        self,
        camera_id: str,
        events_for_display: List[DetectionEvent],
        zone_events: List[ZoneEvent],
        removed_ids: List[int],
    ) -> List[DetectionEvent]:
        """구역 상태를 반영해 디스플레이용 이벤트 타입을 보정한다."""
        return self._display_event_mapper.build(
            camera_id,
            events_for_display,
            zone_events,
            removed_ids,
            self.zone_manager,
        )

    def _process_inference(self, camera_id: str) -> None:
        """AI 추론 스레드 — 프레임 큐에서 가져와 _InferencePipeline 1사이클 실행."""
        consecutive_errors = 0
        max_errors = self.config.processing.consecutive_failure_threshold
        # frame_skip: N프레임마다 1회 추론, 나머지는 캐시된 이벤트로 디스플레이만 갱신
        _frame_counter = 0
        _cached_display_evts: List = []

        while self.running and not self.stop_event.is_set():
            stop_flag = self._cams.stop_flag(camera_id)
            if stop_flag and stop_flag.is_set():
                break
            fq = self._cams.frame_queues.get(camera_id)
            if fq is None:
                time.sleep(0.1)
                continue
            try:
                frame = fq.get(timeout=1.0)
            except Empty:
                continue
            try:
                self._increment_stat("frames_processed")
                _frame_counter += 1
                _skip = self.config.processing.frame_skip
                # 스킵 프레임: 추론 없이 캐시된 이벤트로 디스플레이만 갱신 (부드러운 화면)
                if _skip > 1 and _frame_counter % _skip != 0:
                    self._display.update_frame(camera_id, frame, _cached_display_evts)
                    continue
                events = self._pipeline._infer(camera_id, frame)
                events_for_display = events.copy()
                events_for_dataset = events.copy()
                self._record_detection_snapshot(camera_id, events)

                events, removed_ids = self.track_manager.update(camera_id, events)
                if removed_ids:
                    self.violation_filter.purge(camera_id, removed_ids)
                events = self.violation_filter.filter(camera_id, events)

                self._pipeline._collect(frame, events_for_dataset, camera_id)
                zone_events, frame = self._pipeline._check_zones(camera_id, events, frame)
                self._pipeline._enqueue(camera_id, events, zone_events)
                self._save_zone_event_snapshots(camera_id, zone_events)
                display_evts = self._build_display_events(
                    camera_id, events_for_display, zone_events, removed_ids
                )
                # 웹 스트리밍을 위해 항상 프레임 저장 (display 모드 여부 무관)
                # 새 이벤트가 있으면 캐시 갱신, 없으면 만료된 트랙만 제거하여 유지
                if display_evts:
                    _cached_display_evts = list(display_evts)
                elif removed_ids:
                    _cached_display_evts = [
                        e for e in _cached_display_evts
                        if getattr(e, "object_id", None) not in removed_ids
                    ]
                self._display.update_frame(camera_id, frame, _cached_display_evts)
                consecutive_errors = 0
            except Exception as exc:
                logger.error("[%s] 추론 루프 오류: %s", camera_id, exc, exc_info=True)
                self._increment_stat("inference_errors")
                consecutive_errors += 1

                if consecutive_errors >= max_errors:
                    logger.error(
                        "[%s] 연속 추론 오류 %d회 — 카메라를 재연결 큐에 등록하고 스레드를 종료합니다",
                        camera_id, consecutive_errors,
                    )
                    cam = self._cams.cameras.get(camera_id)
                    source = getattr(cam, "source", None)
                    if source is not None:
                        self._cams.enqueue_retry(camera_id, source, delay_seconds=30.0)
                    self._cams.unregister(camera_id)
                    break

                backoff = min(2 ** consecutive_errors, 30)
                logger.warning(
                    "[%s] 추론 오류 %d/%d회 — %.0f초 대기 후 재시도",
                    camera_id, consecutive_errors, max_errors, backoff,
                )
                self.stop_event.wait(timeout=backoff)

    # ------------------------------------------------------------------
    # 공유 워커 스레드
    # ------------------------------------------------------------------

    def _send_events_worker(self) -> None:
        """이벤트 MQTT 발행 워커."""
        consecutive_failures = 0
        while self.running and not self.stop_event.is_set():
            try:
                event_data = self.event_queue.get(timeout=1.0)
                try:
                    if self.event_publisher.publish_event(event_data):
                        self._increment_stat("events_sent")
                        consecutive_failures = 0
                    else:
                        self._increment_stat("events_failed")
                        consecutive_failures += 1
                        logger.warning("이벤트 전송 실패: %s", event_data)
                        if consecutive_failures >= self.config.processing.consecutive_failure_threshold:
                            logger.error(
                                "연속 전송 실패 %d회 - 서버 상태 확인 필요",
                                consecutive_failures,
                            )
                except Exception as exc:
                    logger.error("전송 오류: %s", exc, exc_info=True)
                    self._increment_stat("events_failed")
                    consecutive_failures += 1
            except Empty:
                pass
            except Exception as exc:
                logger.error("워커 오류: %s", exc, exc_info=True)

    def _cleanup_worker(self) -> None:
        """주기적 메모리 정리 워커."""
        while self.running and not self.stop_event.is_set():
            try:
                if self.stop_event.wait(timeout=self.cleanup_interval):
                    break
                logger.info("메모리 정리 시작...")
                removed = self._debouncer.cleanup()
                if removed > 0:
                    logger.info("  - last_events: %d개 정리됨", removed)
                removed_history = self.violation_filter.cleanup(self._history_timeout)
                if removed_history > 0:
                    logger.info("  - detection_history: %d개 정리됨", removed_history)
                qsize = self.event_queue.qsize()
                qmax  = self.event_queue.maxsize
                if qsize > qmax * self.config.processing.queue_warning_threshold:
                    logger.warning("이벤트 큐 포화 경고: %d/%d", qsize, qmax)
                logger.info("메모리 정리 완료")
            except Exception as exc:
                logger.error("정리 워커 오류: %s", exc)

    # ------------------------------------------------------------------
    # 라이프사이클
    # ------------------------------------------------------------------

    def start(self) -> None:
        """비디오 프로세서를 시작한다."""
        if self.running:
            logger.warning("이미 실행 중입니다")
            return
        if not self._cams.cameras:
            logger.error("등록된 카메라가 없습니다")
            return

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # 헤드리스 환경에서는 GUI 함수 무시
        time.sleep(0.1)

        self.running = True
        self.stop_event.clear()
        with self._stats_lock:
            self.stats.start_time = time.time()

        for camera_id in list(self._cams.cameras):
            self._cams.ensure_stop_flag(camera_id)
            self._cams.start_threads(
                camera_id, self._process_camera, self._process_inference
            )

        self.sender_thread = Thread(
            target=self._send_events_worker, daemon=True, name="EventSender"
        )
        self.sender_thread.start()

        self.cleanup_thread = Thread(
            target=self._cleanup_worker, daemon=True, name="MemoryCleanup"
        )
        self.cleanup_thread.start()

        self._camera_retry_thread = Thread(
            target=self._camera_retry_worker, daemon=True, name="CameraRetry"
        )
        self._camera_retry_thread.start()

        self._governor_thread = Thread(
            target=self._governor.run,
            args=(self.stop_event,),
            daemon=True,
            name="AdaptiveGovernor",
        )
        self._governor_thread.start()

        # display_thread는 메인 스레드에서 직접 start_display_loop()로 실행해야 함
        # (Windows에서 cv2.imshow는 메인 스레드 전용)

        logger.info(
            "프로세서 시작 (%d대 카메라, 분리된 추론 스레드)", self._cams.count
        )

    def stop(self) -> None:
        """비디오 프로세서를 안전하게 중지한다."""
        logger.info("프로세서 중지 중...")
        self.running = False
        self.stop_event.set()

        self._cams.set_all_stop_flags()

        timeout = self.config.processing.thread_join_timeout
        for thread_map in (self._cams.camera_threads, self._cams.inference_threads):
            for camera_id, t in thread_map.items():
                if t.is_alive():
                    t.join(timeout=timeout)
                    if t.is_alive():
                        logger.warning("[%s] 스레드 종료 시간 초과", camera_id)

        for t in (
            self.sender_thread,
            self.cleanup_thread,
            self._camera_retry_thread,
            self._governor_thread,
            self.display_thread,
        ):
            if t and t.is_alive():
                t.join(timeout=timeout)

        self.release_all_cameras()

        self.event_publisher.disconnect()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # 헤드리스 환경에서는 GUI 함수 무시
        logger.info("프로세서 종료 완료")

    def release_all_cameras(self) -> None:
        """등록된 모든 카메라 리소스를 해제한다."""
        for cam in self._cams.cameras.values():
            try:
                cam.release()
            except Exception:
                pass

    def start_display_loop(self) -> None:
        """메인 스레드에서 디스플레이 루프를 실행한다."""
        self._display.run_worker(self.stop_event, lambda: self.running)

    # ------------------------------------------------------------------
    # 통계
    # ------------------------------------------------------------------

    def get_camera_frame(
        self, camera_id: str, *, annotated: bool = False
    ) -> Optional[Any]:
        """특정 카메라의 최신 프레임(numpy ndarray)을 반환한다.

        추론 스레드가 아직 프레임을 등록하지 않았거나 카메라 ID가 없으면 None 반환.
        반환된 배열은 copy() 되어 있으므로 호출 측에서 안전하게 읽을 수 있다.

        Args:
            annotated: True 이면 탐지 박스·라벨이 그려진 프레임을 반환한다.
        """
        with self._display._lock:
            entry = self._display._frames.get(camera_id)
        if entry is None:
            return None
        frame, events = entry
        if frame is None:
            return None
        out = frame.copy()
        if annotated:
            if events:
                out = draw_events(out, events)
            # 디버그: 항상 프레임 카운터 표시로 코드 경로 확인
            import cv2 as _cv2
            _cv2.putText(
                out,
                f"EVT:{len(events)}",
                (10, 30),
                _cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                _cv2.LINE_AA,
            )
        return out

    def save_event_snapshot(
        self,
        camera_id: str,
        event_type: str = "event",
        snapshot_dir: str = "snapshots",
        zone_id: str = "",
    ) -> Optional[str]:
        """현재 프레임을 스냅샷으로 저장하고 파일 경로를 반환한다.

        저장 경로: ``<snapshot_dir>/<camera_id>/<YYYYMMDD_HHMMSS_mmm_[zone_id_]<event_type>.jpg>``
        opencv(cv2)가 없으면 None 을 반환한다.
        """
        try:
            import cv2  # 런타임 의존성 — cv2 없는 환경도 graceful 처리
        except ImportError:
            return None

        frame = self.get_camera_frame(camera_id)
        if frame is None:
            return None

        from pathlib import Path as _Path
        from datetime import datetime as _dt

        safe_type = re.sub(r"[^\w\-]", "_", event_type)
        now = _dt.now()
        ts = now.strftime("%Y%m%d_%H%M%S_") + now.strftime("%f")[:3]
        safe_zone = re.sub(r"[^\w\-]", "_", zone_id) if zone_id else ""
        out_dir = _Path(snapshot_dir) / camera_id
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{ts}_{safe_zone}_{safe_type}.jpg" if safe_zone else f"{ts}_{safe_type}.jpg"
        dest = out_dir / filename
        cv2.imwrite(str(dest), frame)
        logger.info("[Snapshot] saved %s", dest)
        return str(dest)

    def get_stats(self) -> Dict:
        with self._stats_lock:
            snapshot = self.stats.snapshot()
        return ProcessorStats.with_derived_stats(snapshot)

    def get_camera_status(self) -> Dict[str, dict]:
        """카메라별 연결 상태·재시도 횟수·마지막 프레임 시간을 반환한다.

        반환 예시::

            {
                "camera-1": {
                    "status": "online",
                    "connected": True,
                    "reconnect_attempts": 0,
                    "last_frame_time": 1741600000.5,
                    "last_frame_age_sec": 0.3,
                },
                "camera-2": {
                    "status": "reconnecting",
                    "connected": False,
                    "reconnect_attempts": 2,
                    "last_frame_time": 1741599800.0,
                    "last_frame_age_sec": 200.5,
                },
            }
        """
        now = time.time()
        result: Dict[str, dict] = {}
        for cam_id, cam in self._cams.cameras.items():
            if cam.connected:
                status = "online"
            elif cam.reconnect_attempts > 0:
                status = "reconnecting"
            else:
                status = "offline"
            last_ft = cam.last_frame_time
            result[cam_id] = {
                "status": status,
                "connected": cam.connected,
                "reconnect_attempts": cam.reconnect_attempts,
                "last_frame_time": last_ft if last_ft else None,
                "last_frame_age_sec": round(now - last_ft, 1) if last_ft else None,
            }
        for cam_id in self._cams.pending_camera_ids():
            if cam_id not in result:
                result[cam_id] = {
                    "status": "reconnecting",
                    "connected": False,
                    "reconnect_attempts": -1,
                    "last_frame_time": None,
                    "last_frame_age_sec": None,
                }
        return result

    def get_detection_snapshot(self) -> Dict[str, dict]:
        """카메라별 최신 탐지 스냅샷을 반환한다.

        반환 예시::

            {
                "camera-1": {
                    "timestamp": 1741600000.8,
                    "detections": [
                        {"type": "person", "bbox": {...}, "confidence": 0.87, ...},
                    ],
                },
            }
        """
        with self._snapshot_lock:
            return dict(self._latest_snapshots)

    def print_stats(self) -> None:
        s = self.get_stats()
        logger.info(
            "\n%s\n처리 통계\n%s\n"
            "프레임: %d 처리 | %d 드롭 | FPS: %s\n"
            "이벤트: 감지 %d | 전송 %d | 필터링 %d | 드롭 %d | 실패 %d\n"
            "오류: 추론 %d | 카메라 %d\n"
            "성능: 평균 추론 시간 %.1fms\n"
            "카메라: %d대 | 가동 시간: %ss\n%s",
            "=" * 70, "=" * 70,
            s["frames_processed"], s["frames_dropped"], s["fps"],
            s["events_detected"], s["events_sent"], s["events_filtered"],
            s["events_dropped"], s["events_failed"],
            s["inference_errors"], s["camera_errors"],
            s["avg_inference_ms"],
            s["camera_count"], s["uptime_seconds"],
            "=" * 70,
        )
