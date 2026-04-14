"""AI 추론, 구역 감지, 이벤트 큐 적재를 담당하는 내부 파이프라인."""

from __future__ import annotations

import logging
import time
from queue import Full, Queue
from typing import Any, Dict, List, TYPE_CHECKING, Tuple

from ..utils.zone_detection import ZoneEvent
from .events import DetectionEvent

if TYPE_CHECKING:
    from threading import Lock

    from .ai_analysis import AIAnalyzer
    from .event_filters import CumulativeViolationFilter, TrackManager
    from .processor import _EventDebouncer
    from ._display_grid import _DisplayGrid

logger = logging.getLogger(__name__)


class _InferencePipeline:
    """AI 추론, 구역 감지, 이벤트 큐 처리를 담당한다."""

    def __init__(
        self,
        analyzers: Dict[str, "AIAnalyzer"],
        camera_ai_flags: Dict[str, Dict[str, bool]],
        track_manager: "TrackManager",
        violation_filter: "CumulativeViolationFilter",
        debouncer: "_EventDebouncer",
        event_queue: "Queue",
        zone_manager,
        dataset_collector,
        display: "_DisplayGrid",
        snapshot_store: Dict[str, dict],
        snapshot_lock: "Lock",
        zone_in_objects: Dict[str, set],
        increment_stat,
        add_inference_metrics,
        display_enabled: bool,
        queue_warning_threshold: float = 0.8,
    ) -> None:
        self._analyzers = analyzers
        self._camera_ai_flags = camera_ai_flags
        self._track_manager = track_manager
        self._violation_filter = violation_filter
        self._debouncer = debouncer
        self._event_queue = event_queue
        self._zone_manager = zone_manager
        self._dataset_collector = dataset_collector
        self._display = display
        self._snapshots = snapshot_store
        self._snapshot_lock = snapshot_lock
        self._zone_in_objects = zone_in_objects
        self._increment_stat = increment_stat
        self._add_inference_metrics = add_inference_metrics
        self.display_enabled = display_enabled
        self.queue_warning_threshold = queue_warning_threshold

    def _infer(self, camera_id: str, frame: Any) -> List[DetectionEvent]:
        """AIAnalyzer를 호출하고 추론 시간을 기록한다."""
        analyzer = self._analyzers.get(camera_id)
        if analyzer is None:
            logger.error("[%s] 분석기 인스턴스를 찾을 수 없습니다", camera_id)
            self._increment_stat("inference_errors")
            return []

        flags = self._camera_ai_flags.get(
            camera_id,
            {
                "use_helmet": True,
                "use_pose": True,
                "use_person": False,
                "use_face": False,
                "use_appearance": False,
            },
        )
        started_at = time.time()
        try:
            return analyzer.run_inference(frame, camera_id=camera_id, **flags)
        except Exception as exc:
            logger.error("[%s] AI 추론 실패: %s", camera_id, exc, exc_info=True)
            self._increment_stat("inference_errors")
            return []
        finally:
            self._add_inference_metrics(time.time() - started_at)

    def _collect(
        self, frame: Any, events: List[DetectionEvent], camera_id: str
    ) -> None:
        """데이터셋 프레임을 저장한다."""
        if not self._dataset_collector:
            return
        try:
            self._dataset_collector.save_frame(frame, events, camera_id=camera_id)
        except IOError as exc:
            logger.error("[%s] 데이터셋 파일 저장 실패: %s", camera_id, exc)
        except Exception as exc:
            logger.warning("[%s] 데이터셋 저장 오류: %s", camera_id, exc)

    def _check_zones(
        self, camera_id: str, events: List[DetectionEvent], frame: Any
    ) -> Tuple[List[ZoneEvent], Any]:
        """ZoneManager를 호출해 구역 이벤트를 생성한다."""
        if not self._zone_manager:
            return [], frame
        try:
            return self._zone_manager.check_zones(camera_id, events), frame
        except Exception as exc:
            logger.warning("[%s] 구역 감지 오류: %s", camera_id, exc)
            return [], frame

    def _enqueue(
        self,
        camera_id: str,
        events: List[DetectionEvent],
        zone_events: List[ZoneEvent],
    ) -> None:
        """디바운싱을 적용해 이벤트 큐에 비블로킹 적재한다."""

        def _put(event_dict: dict) -> None:
            try:
                self._event_queue.put_nowait(event_dict)
                self._increment_stat("events_detected")
            except Full:
                self._increment_stat("events_dropped")
                self._debouncer.save_locally(event_dict)
                logger.warning("[%s] 이벤트 큐 가득 참: 로컬 저장", camera_id)

        for event in events:
            event_id = event.object_id if event.object_id is not None else 0
            if event.object_id is not None:
                frame_count = self._track_manager.get_frame_count(camera_id, event.object_id)
                if frame_count < self._track_manager.min_track_frames:
                    continue
            if self._debouncer.should_send(camera_id, event.event_type.value, event_id):
                event_data = event.to_dict()
                event_data["camera_id"] = camera_id
                _put(event_data)

        for zone_event in zone_events:
            event_dict = zone_event.to_dict()
            if "type" not in event_dict:
                event_dict["type"] = event_dict.get("event_type")
            _put(event_dict)
