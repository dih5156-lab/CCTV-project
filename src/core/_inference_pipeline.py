"""AI 추론, 구역 감지, 이벤트 큐 적재를 담당하는 내부 파이프라인."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Set, Tuple

from ..utils.zone_detection import ZoneEvent
from .events import DetectionEvent

if TYPE_CHECKING:
    from .ai.analyzer import AIAnalyzer
    from .event_debouncer import EventDebouncer
    from .event_dispatcher import EventDispatcher
    from .event_filters import CumulativeViolationFilter, TrackManager

logger = logging.getLogger(__name__)


@dataclass
class InferenceCycleResult:
    """한 프레임 처리 결과."""

    frame: Any
    events_for_display: List[DetectionEvent]
    zone_events: List[ZoneEvent]
    removed_ids: Set[int]


class _InferencePipeline:
    """AI 추론, 구역 감지, 이벤트 큐 처리를 담당한다."""

    def __init__(
        self,
        analyzers: Dict[str, "AIAnalyzer"],
        camera_ai_flags: Dict[str, Dict[str, bool]],
        track_manager: "TrackManager",
        violation_filter: "CumulativeViolationFilter",
        debouncer: "EventDebouncer",
        event_dispatcher: "EventDispatcher",
        zone_manager,
        on_raw_detections: Callable[[str, Any, List[DetectionEvent]], None],
        increment_stat,
        add_inference_metrics,
    ) -> None:
        self._analyzers = analyzers
        self._camera_ai_flags = camera_ai_flags
        self._track_manager = track_manager
        self._violation_filter = violation_filter
        self._debouncer = debouncer
        self._event_dispatcher = event_dispatcher
        self._zone_manager = zone_manager
        self._on_raw_detections = on_raw_detections
        self._increment_stat = increment_stat
        self._add_inference_metrics = add_inference_metrics

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

    def process_frame(self, camera_id: str, frame: Any) -> InferenceCycleResult:
        """한 프레임의 추론부터 이벤트 큐 적재까지 처리한다."""
        events = self._infer(camera_id, frame)
        events_for_display = events.copy()

        self._collect(frame, events, camera_id)

        events, removed_ids = self._track_manager.update(camera_id, events)
        if removed_ids:
            self._violation_filter.purge(camera_id, removed_ids)
        events = self._violation_filter.filter(camera_id, events)

        zone_events, frame = self._check_zones(camera_id, events, frame)
        self._enqueue(camera_id, events, zone_events)

        return InferenceCycleResult(
            frame=frame,
            events_for_display=events_for_display,
            zone_events=zone_events,
            removed_ids=removed_ids,
        )

    def _collect(
        self,
        frame: Any,
        events: List[DetectionEvent],
        camera_id: str,
    ) -> None:
        """추론 직후 부수 효과(스냅샷/데이터셋 수집)를 호출한다."""
        self._on_raw_detections(camera_id, frame, events)

    def _enqueue(
        self,
        camera_id: str,
        events: List[DetectionEvent],
        zone_events: List[ZoneEvent],
    ) -> None:
        """디바운싱을 적용해 이벤트 큐에 비블로킹 적재한다."""

        for event in events:
            event_id = event.object_id if event.object_id is not None else 0
            if event.object_id is not None:
                frame_count = self._track_manager.get_frame_count(camera_id, event.object_id)
                if frame_count < self._track_manager.min_track_frames:
                    continue
            if self._debouncer.should_send(camera_id, event.event_type.value, event_id):
                event_data = event.to_dict()
                event_data["camera_id"] = camera_id
                self._event_dispatcher.enqueue(camera_id, event_data)

        for zone_event in zone_events:
            event_dict = zone_event.to_dict()
            if "type" not in event_dict:
                event_dict["type"] = event_dict.get("event_type")
            self._event_dispatcher.enqueue(camera_id, event_dict)
