"""
processor.py - 실시간 CCTV 객체 감지 프로세서
다중 카메라 처리, RTSP 재연결, 이벤트 필터링 및 서버 전송

클래스 구성:
  ProcessorStats   - 처리 통계 DTO
  _EventDebouncer  - 이벤트 디바운싱 + 로컬 백업  (VideoProcessor 내부용)
  _DisplayGrid     - 다중 카메라 그리드 디스플레이 (VideoProcessor 내부용)
  _CameraRegistry  - 카메라·스레드·재시도 큐 관리  (VideoProcessor 내부용)
  VideoProcessor   - 파이프라인 오케스트레이터 (공개 API)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict, replace
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..config import AppConfig
from ..protocols.mqtt import MqttEventPublisher
from ..utils.camera_input import RTSPCamera
from ..utils.dataset_collector import DatasetCollector
from ..utils.visualizer import draw_events
from ..utils.zone_detection import ZoneEvent, ZoneManager
from ..utils.zone_drawer import GridLayout, ZoneDrawer
from .ai_analysis import AIAnalyzer
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
        # 낙상/헬멧 추적 딕셔너리 정리 (1시간 이상 갱신 없는 항목 제거)
        old_cutoff = time.time() - 3600
        with self._lock:
            before = len(self._last_events)
            self._last_events     = {k: v for k, v in self._last_events.items()     if v > cutoff}
            self._fall_first_seen = {k: v for k, v in self._fall_first_seen.items() if v > old_cutoff}
            self._fall_last_seen  = {k: v for k, v in self._fall_last_seen.items()  if v > old_cutoff}
            self._fall_alerted    = {k: v for k, v in self._fall_alerted.items()    if v > old_cutoff}
            self._head_last_seen  = {k: v for k, v in self._head_last_seen.items()  if v > old_cutoff}
            self._head_alerted    = {k: v for k, v in self._head_alerted.items()    if v > old_cutoff}
            return before - len(self._last_events)

    def save_locally(self, event_data: Dict) -> None:
        """큐 포화 시 이벤트를 로컬 JSON 파일로 백업한다."""
        try:
            backup_dir = os.path.join(os.getcwd(), "event_backup")
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"event_{timestamp}_{event_data.get('camera_id', 'unknown')}.json"
            filepath = os.path.join(backup_dir, filename)
            with open(filepath, "w", encoding="utf-8") as fp:
                json.dump(event_data, fp, ensure_ascii=False, indent=2)
            logger.debug("이벤트 로컬 저장: %s", filepath)
        except Exception as exc:
            logger.error("로컬 저장 실패: %s", exc)


# ===========================================================================
# _DisplayGrid  (내부용)
# ===========================================================================


class _DisplayGrid:
    """다중 카메라 통합 그리드 디스플레이.

    VideoProcessor 에서만 사용한다.
    """

    WIDTH   = 1280
    HEIGHT  = 720
    MAX_FPS = 20

    def __init__(self, get_fps) -> None:
        self._get_fps = get_fps
        self._frames: Dict[str, Any] = {}
        self._lock = Lock()
        self.window_name = "CCTV Multi-Camera View"
        self._drawer: Optional[ZoneDrawer] = None

    def set_drawer(self, drawer: ZoneDrawer) -> None:
        """ZoneDrawer 를 등록한다. run_worker 시작 전에 호출해야 한다."""
        self._drawer = drawer

    def update_frame(
        self, camera_id: str, frame: Any, events: List[DetectionEvent]
    ) -> None:
        """추론 스레드에서 최신 프레임·이벤트를 등록한다."""
        if frame is None:
            return
        with self._lock:
            self._frames[camera_id] = (frame, list(events))

    def build_grid(self) -> Optional[Any]:
        """전체 카메라 프레임으로 그리드 이미지를 생성한다."""
        with self._lock:
            if not self._frames:
                return None
            raw_items = [
                (cam_id, frame.copy(), list(evts))
                for cam_id, (frame, evts) in self._frames.items()
                if frame is not None
            ]
        n = len(raw_items)
        if n == 0:
            return None
        cols = max(1, int(n ** 0.5) + (1 if n > 1 else 0))
        rows = (n + cols - 1) // cols
        tw = self.WIDTH  // cols
        th = self.HEIGHT // rows

        # drawer 에 레이아웃 정보 전달
        if self._drawer is not None:
            layout = GridLayout(
                camera_ids=[cam_id for cam_id, _, _ in raw_items],
                cols=cols,
                rows=rows,
                tile_w=tw,
                tile_h=th,
                orig_sizes={
                    cam_id: (frame.shape[1], frame.shape[0])
                    for cam_id, frame, _ in raw_items
                },
            )
            self._drawer.set_layout(layout)

        resized: List[Any] = []
        for cam_id, frame, evts in raw_items:
            annotated = draw_events(frame, evts)
            cv2.putText(
                annotated, f"[{cam_id}] {len(evts)}",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
            )
            resized.append(cv2.resize(annotated, (tw, th)))
        black = np.zeros((th, tw, 3), dtype=np.uint8)
        grid_rows = []
        for r in range(rows):
            row = [
                resized[r * cols + c] if r * cols + c < n else black
                for c in range(cols)
            ]
            grid_rows.append(cv2.hconcat(row))
        grid = cv2.vconcat(grid_rows)
        cv2.putText(
            grid, f"FPS: {self._get_fps():.1f} | Cams: {n}",
            (8, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )
        # ZoneDrawer 오버레이
        if self._drawer is not None:
            grid = self._drawer.overlay(grid)
        return grid

    def run_worker(self, stop_event: Event, is_running) -> None:
        """디스플레이 워커 루프 — 전용 스레드에서 실행된다."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.WIDTH, self.HEIGHT)
        if self._drawer is not None:
            cv2.setMouseCallback(self.window_name, self._drawer.mouse_callback)
        interval = 1.0 / self.MAX_FPS
        last_render = 0.0
        while is_running() and not stop_event.is_set():
            try:
                now = time.monotonic()
                elapsed = now - last_render
                if elapsed < interval:
                    wait_ms = max(1, int((interval - elapsed) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == 0xFF:
                        continue
                    if self._drawer is not None and self._drawer.handle_key(key):
                        continue
                    if key == ord("q"):
                        logger.info("'q' 입력 감지 - 중지합니다")
                        stop_event.set()
                        break
                    continue
                grid = self.build_grid()
                last_render = time.monotonic()
                if grid is not None:
                    cv2.imshow(self.window_name, grid)
                    grid = None
                key = cv2.waitKey(1) & 0xFF
                if key != 0xFF:
                    if self._drawer is not None and self._drawer.handle_key(key):
                        pass
                    elif key == ord("q"):
                        logger.info("'q' 입력 감지 - 중지합니다")
                        stop_event.set()
                        break
            except Exception as exc:
                logger.error("디스플레이 워커 오류: %s", exc)
                time.sleep(0.1)


# ===========================================================================
# _CameraRegistry  (내부용)
# ===========================================================================


class _CameraRegistry:
    """카메라 인스턴스·스레드·재시도 큐를 관리하는 내부 레지스트리.

    VideoProcessor 에서만 사용한다.
    """

    def __init__(
        self, config: AppConfig, stop_event: Event, is_running
    ) -> None:
        self._config = config
        self._stop_event = stop_event
        self._is_running = is_running

        self.cameras:           Dict[str, RTSPCamera] = {}
        self.camera_threads:    Dict[str, Thread]     = {}
        self.inference_threads: Dict[str, Thread]     = {}
        self.frame_queues:      Dict[str, Queue]      = {}
        self._stop_flags:       Dict[str, Event]      = {}

        self._pending:      List[Tuple[str, Any, float]] = []
        self._pending_lock  = Lock()

    @property
    def count(self) -> int:
        return len(self.cameras)

    def register(self, camera_id: str, camera: RTSPCamera) -> None:
        """이미 연결된 카메라를 레지스트리에 등록한다."""
        self.cameras[camera_id]     = camera
        self.frame_queues[camera_id] = Queue(maxsize=1)
        self._stop_flags[camera_id]  = Event()

    def unregister(self, camera_id: str) -> None:
        """카메라와 관련 스레드를 레지스트리에서 제거한다."""
        timeout = self._config.processing.thread_join_timeout
        flag = self._stop_flags.pop(camera_id, None)
        if flag:
            flag.set()
        for thread_map in (self.camera_threads, self.inference_threads):
            t = thread_map.pop(camera_id, None)
            if t and t.is_alive():
                t.join(timeout=timeout)
                if t.is_alive():
                    logger.warning("[%s] 스레드 종료 시간 초과", camera_id)
        cam = self.cameras.pop(camera_id, None)
        if cam:
            cam.release()
        self.frame_queues.pop(camera_id, None)

    def stop_flag(self, camera_id: str) -> Optional[Event]:
        return self._stop_flags.get(camera_id)

    def ensure_stop_flag(self, camera_id: str) -> Event:
        """카메라 정지 플래그를 생성하거나 초기화하여 반환한다."""
        if camera_id not in self._stop_flags:
            self._stop_flags[camera_id] = Event()
        else:
            self._stop_flags[camera_id].clear()
        return self._stop_flags[camera_id]

    def start_threads(self, camera_id: str, cam_target, inf_target) -> None:
        """카메라·추론 스레드를 시작하고 레지스트리에 등록한다."""
        flag = self._stop_flags.get(camera_id)
        if flag:
            flag.clear()
        cam_t = Thread(
            target=cam_target,
            args=(camera_id, self.cameras[camera_id]),
            daemon=True,
            name=f"Camera-{camera_id}",
        )
        self.camera_threads[camera_id] = cam_t
        cam_t.start()
        inf_t = Thread(
            target=inf_target,
            args=(camera_id,),
            daemon=True,
            name=f"Inference-{camera_id}",
        )
        self.inference_threads[camera_id] = inf_t
        inf_t.start()

    def enqueue_retry(
        self, camera_id: str, source: Any, delay_seconds: float = 30.0
    ) -> None:
        """연결 실패한 카메라를 재시도 큐에 등록한다."""
        next_ts = time.time() + delay_seconds
        with self._pending_lock:
            self._pending = [
                (cid, src, ts) for cid, src, ts in self._pending if cid != camera_id
            ]
            self._pending.append((camera_id, source, next_ts))
        logger.info("[%s] 재연결 예약: %.0f초 후", camera_id, delay_seconds)

    def poll_ready_retries(self) -> List[Tuple[str, Any, float]]:
        """만료된 재시도 항목을 꺼내 반환한다 (큐에서 제거)."""
        now = time.time()
        ready: List[Tuple[str, Any, float]] = []
        remaining: List[Tuple[str, Any, float]] = []
        with self._pending_lock:
            for item in self._pending:
                (ready if now >= item[2] else remaining).append(item)
            self._pending = remaining
        return ready

    def pending_camera_ids(self) -> List[str]:
        """재연결 대기 중인 카메라 ID 목록을 반환한다 (읽기 전용)."""
        with self._pending_lock:
            return [cid for cid, _, _ in self._pending]


# ===========================================================================
# VideoProcessor  (공개 API)
# ===========================================================================


class VideoProcessor:
    """AI 추론을 사용한 다중 카메라 비디오 처리 파이프라인.

    내부 헬퍼:
        _debouncer (_EventDebouncer) - 이벤트 디바운싱
        _display   (_DisplayGrid)    - 그리드 디스플레이
        _cams      (_CameraRegistry) - 카메라 생명주기
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._analyzers: Dict[str, AIAnalyzer] = {}
        # 카메라별 AI 모델 플래그 (ai_models 설정 딕셔너리에서 파싱)
        self._camera_ai_flags: Dict[str, Dict[str, bool]] = {}

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

        # ── 탐지 스냅샷 (카메라별 최신 탐지 결과 + 타임스탬프) ──────────
        self._latest_snapshots: Dict[str, dict] = {}
        self._snapshot_lock = Lock()

        # ── 워커 스레드 핸들 ─────────────────────────────────────────
        self.sender_thread:        Optional[Thread] = None
        self.cleanup_thread:       Optional[Thread] = None
        self.display_thread:       Optional[Thread] = None
        self._camera_retry_thread: Optional[Thread] = None

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
        """
        mp = model_paths or {}
        analyzer = AIAnalyzer(
            helmet_model_path=mp.get("helmet") or self.config.models.helmet_model,
            person_model_path=mp.get("person") or self.config.models.person_model,
            pose_model_path=mp.get("pose")   or self.config.models.pose_model,
            confidence_threshold=self.config.detection.pose_confidence,
            device=self.config.detection.device,
            fall_height_ratio=self.config.detection.fall_height_ratio,
        )
        analyzer.helmet_threshold = self.config.detection.helmet_confidence
        analyzer.person_threshold = self.config.detection.person_confidence
        analyzer.pose_threshold   = self.config.detection.pose_confidence
        return analyzer

    @staticmethod
    def _parse_detections(detections: Optional[List[str]]) -> Dict[str, bool]:
        """cameras.json 의 detections 리스트를 run_inference 플래그 딕셔너리로 변환한다.

        None 또는 비어있으면 전체 모델 활성화 (하위 호환 유지).

        지원 값:
            "helmet"    →  안전모 미착용 감지
            "fall"      →  낙상 감지 (pose 모델)
            "intrusion" →  무단침입 / 좌리 감지
            "person"    →  사람 감지만
        """
        if not detections:
            return {"use_helmet": True, "use_pose": True, "use_person": True}

        modes = {d.lower() for d in detections}
        use_pose   = "fall" in modes
        use_helmet = "helmet" in modes
        use_person = bool(modes & {"intrusion", "person"}) or use_pose or use_helmet
        return {"use_helmet": use_helmet, "use_pose": use_pose, "use_person": use_person}

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
                except Exception as exc:
                    logger.warning("[%s] 구역 로딩 실패: %s", camera_id, exc)
        elif self.zone_manager:
            try:
                self.zone_manager.load_zones(camera_id, None)
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
    # AI 추론
    # ------------------------------------------------------------------

    def _run_ai_inference(
        self, camera_id: str, frame: Any
    ) -> List[DetectionEvent]:
        """카메라별 AI 분석기를 사용해 프레임을 추론한다."""
        analyzer = self._analyzers.get(camera_id)
        if analyzer is None:
            logger.error("[%s] 분석기 인스턴스를 찾을 수 없습니다", camera_id)
            self._increment_stat("inference_errors")
            return []
        start = time.time()
        flags = self._camera_ai_flags.get(
            camera_id, {"use_helmet": True, "use_pose": True, "use_person": True}
        )
        try:
            events = analyzer.run_inference(frame, **flags)
        except Exception as exc:
            logger.error("[%s] AI 추론 실패: %s", camera_id, exc, exc_info=True)
            self._increment_stat("inference_errors")
            return []
        finally:
            self._add_inference_metrics(time.time() - start)
        return events

    # ------------------------------------------------------------------
    # 데이터셋 수집 / 구역 탐지 / 이벤트 큐
    # ------------------------------------------------------------------

    def _collect_dataset(
        self, frame: Any, events: List[DetectionEvent], camera_id: str
    ) -> None:
        if not self.dataset_collector:
            return
        try:
            self.dataset_collector.save_frame(frame, events, camera_id=camera_id)
        except IOError as exc:
            logger.error("[%s] 데이터셋 파일 저장 실패: %s", camera_id, exc)
        except Exception as exc:
            logger.warning("[%s] 데이터셋 저장 오류: %s", camera_id, exc)

    def _check_danger_zones(
        self, camera_id: str, events: List[DetectionEvent], frame: Any
    ) -> Tuple[List[ZoneEvent], Any]:
        """위험 구역 침입 감지."""
        zone_events: List[ZoneEvent] = []
        if not self.zone_manager:
            return zone_events, frame
        try:
            zone_events = self.zone_manager.check_zones(camera_id, events)
            # 시각화는 ZoneDrawer.overlay()에서 위임하므로 draw_zones 호출 제거
        except Exception as exc:
            logger.warning("[%s] 구역 감지 오류: %s", camera_id, exc)
        return zone_events, frame

    def _queue_events(
        self,
        camera_id: str,
        events: List[DetectionEvent],
        zone_events: List[ZoneEvent],
    ) -> None:
        """디바운싱과 함께 이벤트를 큐에 추가한다 (비블로킹)."""
        for event in events:
            event_id = event.object_id if event.object_id is not None else 0
            if event.object_id is not None:
                frame_count = self.track_manager.get_frame_count(
                    camera_id, event.object_id
                )
                if frame_count < self.track_manager.min_track_frames:
                    continue
            if self._debouncer.should_send(camera_id, event.event_type.value, event_id):
                event_data = event.to_dict()
                event_data["camera_id"] = camera_id
                try:
                    self.event_queue.put_nowait(event_data)
                    self._increment_stat("events_detected")
                except Full:
                    self._increment_stat("events_dropped")
                    self._debouncer.save_locally(event_data)
                    logger.warning("[%s] 이벤트 큐 가득 참: 로컬 저장", camera_id)

        for zone_event in zone_events:
            evt_dict = zone_event.to_dict()
            if "type" not in evt_dict:
                evt_dict["type"] = evt_dict.get("event_type")
            try:
                self.event_queue.put_nowait(evt_dict)
                self._increment_stat("events_detected")
            except Full:
                self._increment_stat("events_dropped")
                self._debouncer.save_locally(evt_dict)
                logger.warning("[%s] 구역 이벤트 큐 가득 참: 로컬 저장", camera_id)

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

    def _process_inference(self, camera_id: str) -> None:
        """AI 추론 스레드 — 프레임 큐에서 가져와 전체 파이프라인 실행."""
        consecutive_errors = 0
        max_errors = self.config.processing.consecutive_failure_threshold

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
                events = self._run_ai_inference(camera_id, frame)
                events_for_display = events.copy()
                events_for_dataset = events.copy()
                # 최신 탐지 스냅샷 저장 (트래킹·필터링 전 원본 이벤트)
                with self._snapshot_lock:
                    self._latest_snapshots[camera_id] = {
                        "timestamp": time.time(),
                        "detections": [e.to_dict() for e in events],
                    }
                events, removed_ids = self.track_manager.update(camera_id, events)
                if removed_ids:
                    self.violation_filter.purge(camera_id, removed_ids)
                events = self.violation_filter.filter(camera_id, events)
                self._collect_dataset(frame, events_for_dataset, camera_id)
                zone_events, frame = self._check_danger_zones(camera_id, events, frame)
                self._queue_events(camera_id, events, zone_events)

                # ── 구역 점유 상태 영속 업데이트 ────────────────────────────
                zone_set = self._zone_in_objects.setdefault(camera_id, set())

                # 활성 존이 없으면 zone_set을 즉시 클리어
                # (존 삭제 시 zone_exited 이벤트가 발생하지 않아 zone_set이 잔류하는 문제 방지)
                has_active_zones = bool(
                    self.zone_manager
                    and self.zone_manager.zones.get(camera_id)
                )
                if not has_active_zones:
                    zone_set.clear()
                else:
                    for ze in zone_events:
                        if ze.event_type.value in ("zone_entered", "zone_dwelling"):
                            zone_set.add(ze.object_id)
                        elif ze.event_type.value == "zone_exited":
                            zone_set.discard(ze.object_id)
                # 트랙에서 사라진 객체는 zone 집합에서도 제거
                for rid in removed_ids:
                    zone_set.discard(rid)

                if self.config.display:
                    # person 타입만 DANGER_ZONE으로 표시 (head/helmet은 그대로 유지)
                    # zone_set을 사용하므로 zone_exited 전까지 flickering 없음
                    display_evts = [
                        replace(ev, event_type=EventType.DANGER_ZONE)
                        if (
                            ev.event_type == EventType.PERSON
                            and getattr(ev, 'object_id', None) in zone_set
                        )
                        else ev
                        for ev in events_for_display
                    ]
                    self._display.update_frame(camera_id, frame, display_evts)
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

                # 연속 오류 횟수에 비례한 백오프 (최대 30초)
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

        cv2.destroyAllWindows()
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

        if self.config.display:
            self.display_thread = Thread(
                target=self._display.run_worker,
                args=(self.stop_event, lambda: self.running),
                daemon=True,
                name="UnifiedDisplay",
            )
            self.display_thread.start()

        logger.info(
            "프로세서 시작 (%d대 카메라, 분리된 추론 스레드)", self._cams.count
        )

    def stop(self) -> None:
        """비디오 프로세서를 안전하게 중지한다."""
        logger.info("프로세서 중지 중...")
        self.running = False
        self.stop_event.set()

        for flag in self._cams._stop_flags.values():
            flag.set()

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
            self.display_thread,
        ):
            if t and t.is_alive():
                t.join(timeout=timeout)

        for cam in self._cams.cameras.values():
            cam.release()

        self.event_publisher.disconnect()
        cv2.destroyAllWindows()
        logger.info("프로세서 종료 완료")

    # ------------------------------------------------------------------
    # 통계
    # ------------------------------------------------------------------

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
