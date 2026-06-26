"""Background face and appearance context worker for DeepStream."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event
from typing import Any, Callable, Dict, Iterable, List

from ._event_context import events_to_nearby_objects
from .events import DetectionEvent, EventType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceContextTask:
    camera_name: str
    person_events: List[DetectionEvent]
    frame: Any
    flags: Dict[str, bool]
    context_events: List[DetectionEvent]


class DeepStreamContextWorker:
    """Queues and runs face/appearance post-processing off the GStreamer loop."""

    def __init__(
        self,
        *,
        queue: Queue,
        feature_flags_for_camera: Callable[[str], Dict[str, bool]],
        remember_context_events: Callable[[str, List[DetectionEvent]], None],
        collect_context_events: Callable[[str, List[DetectionEvent]], List[DetectionEvent]],
        get_camera_frame: Callable[[str], Any],
        run_face_recognition: Callable[[Any, List[DetectionEvent], str], List[DetectionEvent]],
        log_appearance_capability_hints: Callable[[str, Dict[str, bool]], None],
        refresh_appearance_conditions: Callable[[], None],
        appearance_pipeline: Any,
        enqueue_event: Callable[[DetectionEvent, str], bool],
    ) -> None:
        self.queue = queue
        self.feature_flags_for_camera = feature_flags_for_camera
        self.remember_context_events = remember_context_events
        self.collect_context_events = collect_context_events
        self.get_camera_frame = get_camera_frame
        self.run_face_recognition = run_face_recognition
        self.log_appearance_capability_hints = log_appearance_capability_hints
        self.refresh_appearance_conditions = refresh_appearance_conditions
        self.appearance_pipeline = appearance_pipeline
        self.enqueue_event = enqueue_event

    def submit(self, camera_name: str, filtered_events: Iterable[DetectionEvent]) -> None:
        events = list(filtered_events)
        flags = self.feature_flags_for_camera(camera_name)
        if not (flags.get("use_face") or flags.get("use_appearance")):
            return

        self.remember_context_events(camera_name, events)
        person_events = [event for event in events if event.event_type == EventType.PERSON]
        if not person_events:
            return

        context_events = self.collect_context_events(camera_name, events)
        frame = self.get_camera_frame(camera_name)
        if frame is None:
            return

        try:
            self.queue.put_nowait(
                FaceContextTask(
                    camera_name=camera_name,
                    person_events=person_events,
                    frame=frame,
                    flags=flags,
                    context_events=context_events,
                )
            )
        except Full:
            logger.debug("[%s] 얼굴 인식 워커 큐 가득 참 - 프레임 건너뜀", camera_name)

    def run_loop(self, stop_event: Event) -> None:
        logger.info("얼굴 인식 비동기 워커 시작")
        while not stop_event.is_set():
            try:
                task = self.queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self.process(task)
            except Exception as exc:
                logger.warning("[%s] 얼굴/외형 컨텍스트 후처리 실패: %s", task.camera_name, exc)

    def process(self, task: FaceContextTask) -> None:
        face_events = (
            self.run_face_recognition(
                task.frame,
                task.person_events,
                task.camera_name,
            )
            if task.flags.get("use_face")
            else []
        )
        appearance_events = self._run_appearance(task, face_events)
        for event in face_events + appearance_events:
            self.enqueue_event(event, task.camera_name)

    def _run_appearance(
        self,
        task: FaceContextTask,
        face_events: List[DetectionEvent],
    ) -> List[DetectionEvent]:
        if not task.flags.get("use_appearance"):
            return []

        self.log_appearance_capability_hints(task.camera_name, task.flags)
        self.refresh_appearance_conditions()
        appearance_events = self.appearance_pipeline.run(
            task.frame,
            task.person_events,
            face_events,
            camera_id=task.camera_name,
            use_appearance=True,
            nearby_objects=events_to_nearby_objects(task.context_events),
        )
        for event in appearance_events:
            metadata = dict(event.metadata or {})
            metadata.setdefault("backend", "deepstream_context")
            metadata.setdefault("camera_id", task.camera_name)
            event.metadata = metadata
        return appearance_events
