"""Background face and appearance context worker for DeepStream."""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from queue import Empty, Full, Queue
from threading import Event, RLock
from typing import Any, Callable, Dict, Iterable, List

from ._event_context import events_to_nearby_objects
from ._preview_frame_store import FRAME_IDENTITY_MARKER, read_frame_identity
from .events import DetectionEvent, EventType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceContextTask:
    camera_name: str
    person_events: List[DetectionEvent]
    frame: Any
    flags: Dict[str, bool]
    context_events: List[DetectionEvent]
    generation: int = 0
    frame_identity: dict | None = None


class DeepStreamContextWorker:
    """Queues and runs face/appearance post-processing off the GStreamer loop."""

    def __init__(
        self,
        *,
        queue: Queue,
        feature_flags_for_camera: Callable[[str], Dict[str, bool]],
        remember_context_events: Callable[[str, List[DetectionEvent]], None],
        collect_context_events: Callable[[str, List[DetectionEvent]], List[DetectionEvent]],
        run_face_recognition: Callable[[Any, List[DetectionEvent], str], List[DetectionEvent]],
        log_appearance_capability_hints: Callable[[str, Dict[str, bool]], None],
        refresh_appearance_conditions: Callable[[], None],
        appearance_pipeline: Any,
        enqueue_event: Callable[[DetectionEvent, str], bool],
        record_stat: Callable[[str], Any] = lambda name: None,
        min_interval_sec: float = 0.0,
    ) -> None:
        self.queue = queue
        self.feature_flags_for_camera = feature_flags_for_camera
        self.remember_context_events = remember_context_events
        self.collect_context_events = collect_context_events
        self.run_face_recognition = run_face_recognition
        self.log_appearance_capability_hints = log_appearance_capability_hints
        self.refresh_appearance_conditions = refresh_appearance_conditions
        self.appearance_pipeline = appearance_pipeline
        self.enqueue_event = enqueue_event
        self.record_stat = record_stat
        if not math.isfinite(min_interval_sec) or min_interval_sec < 0:
            raise ValueError("context min_interval_sec must be finite and nonnegative")
        self.min_interval_sec = min_interval_sec
        self._last_admitted: dict[str, dict[int, float]] = {}
        self._inflight: set[tuple[str, int, int]] = set()
        self._generations: Dict[str, int] = {}
        self._state_lock = RLock()
        self._frame_token = 0
        self._pending: dict[str, OrderedDict] = {}
        self.max_pending_frames = 32
        self.max_pending_age_sec = 2.0
        # 이미지 없이 최근 판정 근거만 보관한다. 오래된 근거는 추정 대신 unknown으로 집계한다.
        self.max_match_history_frames = 256
        self._match_history: dict[str, OrderedDict[tuple, str]] = {}

    def stamp_frame(self, frame_meta: Any, camera_name: str) -> None:
        """메타 probe에서만 호출: 재연결에도 재사용하지 않는 토큰을 붙인다."""
        with self._state_lock:
            fields = getattr(frame_meta, "misc_frame_info", None)
            if fields is None or (int(fields[0]) != FRAME_IDENTITY_MARKER and any(fields[:3])):
                self.record_stat("context_identity_rejected")
                return
            self._frame_token += 1
            fields[:3] = (FRAME_IDENTITY_MARKER,
                          self._generations.get(camera_name, 0), self._frame_token)

    def _remember_match_state(self, camera_name: str, key: tuple, state: str) -> None:
        history = self._match_history.setdefault(camera_name, OrderedDict())
        history[key] = state
        history.move_to_end(key)
        while len(history) > self.max_match_history_frames:
            history.popitem(last=False)

    def finish_frame(self, frame_meta: Any, camera_name: str) -> None:
        """텐서/객체 메타를 모두 처리한 프레임의 무대상 여부만 확정한다."""
        with self._state_lock:
            identity = read_frame_identity(frame_meta, camera_name)
            key = self._identity_key(camera_name, identity)
            if key is None or identity["generation"] != self._generations.get(camera_name, 0):
                return
            # 이미 등록/소비된 대상은 빈 두 번째 메타 경로가 덮어쓰지 못한다.
            if (key not in self._match_history.get(camera_name, {})
                    and key not in self._pending.get(camera_name, {})):
                self._remember_match_state(camera_name, key, "no_target")

    @staticmethod
    def _identity_key(camera_name: str, identity: dict | None) -> tuple | None:
        if not identity or identity.get("camera_id") != camera_name:
            return None
        fields = ("source_id", "frame_num", "source_pts_ns", "generation", "frame_token")
        values = tuple(identity.get(name) for name in fields)
        if any(type(value) is not int or value < 0 for value in values):
            return None
        if values[2] >= 2**64 - 1 or values[4] == 0:
            return None
        return values

    def _expire_pending(self, camera_name: str, now: float) -> OrderedDict:
        pending = self._pending.setdefault(camera_name, OrderedDict())
        while pending and now - next(iter(pending.values()))[0] > self.max_pending_age_sec:
            key, _ = pending.popitem(last=False)
            self._remember_match_state(camera_name, key, "expired")
            self.record_stat("context_match_expired")
        return pending

    @contextmanager
    def reset_camera(self, camera_name: str):
        """이전 작업을 무효화하고 같은 경계에서 카메라 상태를 정리한다."""
        with self._state_lock:
            self._generations[camera_name] = self._generations.get(camera_name, 0) + 1
            self._pending.pop(camera_name, None)
            self._match_history.pop(camera_name, None)
            self._last_admitted.pop(camera_name, None)
            self._inflight = {key for key in self._inflight if key[0] != camera_name}
            self.appearance_pipeline.clear_camera(camera_name)
            yield

    @contextmanager
    def commit_guard(self, task: FaceContextTask):
        """추론 밖의 결과 반영과 입력 초기화를 직렬화한다."""
        with self._state_lock:
            yield task.generation == self._generations.get(task.camera_name, 0)

    def submit(self, camera_name: str, filtered_events: Iterable[DetectionEvent]) -> None:
        with self._state_lock:
            self._submit_current(camera_name, filtered_events)

    def _submit_current(self, camera_name: str, filtered_events: Iterable[DetectionEvent]) -> None:
        events = list(filtered_events)
        flags = self.feature_flags_for_camera(camera_name)
        if not (flags.get("use_face") or flags.get("use_appearance")):
            return
        if not events:
            return
        identity = (events[0].metadata or {}).get("frame_identity")
        key = self._identity_key(camera_name, identity)
        if (key is None or identity["generation"] != self._generations.get(camera_name, 0)
                or any(self._identity_key(camera_name, (event.metadata or {}).get("frame_identity"))
                       != key for event in events)):
            self.record_stat("context_identity_rejected")
            return

        self.remember_context_events(camera_name, events)
        person_events = [event for event in events if event.event_type == EventType.PERSON]
        if not person_events:
            return
        context_events = self.collect_context_events(camera_name, events)

        now = time.monotonic()
        pending = self._expire_pending(camera_name, now)
        if key in pending:
            # tensor/object 메타가 같은 사람을 두 번 전달해도 한 번만 처리한다.
            return
        while len(pending) >= self.max_pending_frames:
            removed_key, _ = pending.popitem(last=False)
            self._remember_match_state(camera_name, removed_key, "evicted")
            self.record_stat("context_match_evicted")
        pending[key] = (now, FaceContextTask(
            camera_name=camera_name, person_events=person_events, frame=None,
            flags=flags, context_events=context_events,
            generation=identity["generation"], frame_identity=dict(identity)))
        self._remember_match_state(camera_name, key, "registered")

    def is_current_frame(self, camera_name: str, identity: dict) -> bool:
        """재연결 이전 프레임을 현재 영상으로 제공하지 않도록 세대를 확인한다."""
        with self._state_lock:
            return (self._identity_key(camera_name, identity) is not None
                    and identity["generation"] == self._generations.get(camera_name, 0))

    def accept_frame(self, camera_name: str, frame: Any, identity: dict) -> bool:
        """일치하는 영상만 큐에 전달한다. 누락/늦은 영상은 기다리지 않는다."""
        with self._state_lock:
            key = self._identity_key(camera_name, identity)
            if key is None or identity["generation"] != self._generations.get(camera_name, 0):
                self.record_stat("context_identity_rejected")
                return False
            pending = self._expire_pending(camera_name, time.monotonic())
            matched = pending.pop(key, None)
            if matched is None:
                self.record_stat("context_match_missed")
                # 총계와 사유별 부분집합을 같이 유지한다. 미완료/중복/기록 유실은 무대상으로 추정하지 않는다.
                reason = self._match_history.get(camera_name, {}).get(key)
                if reason not in {"no_target", "expired", "evicted"}:
                    reason = "unknown"
                self.record_stat(f"context_match_missed_{reason}")
                return False
            self._remember_match_state(camera_name, key, "consumed")
            return self._enqueue_matched(replace(matched[1], frame=frame))

    def _enqueue_matched(self, task: FaceContextTask) -> bool:
        # 프레임 전체를 폐기하는 대신 같은 추적 인물의 반복 분석만 주기로 제한한다.
        # 새 인물·추적 ID 없는 인물은 즉시 대상에 넣고, 위험 탐지/이벤트 큐는 건드리지 않는다.
        now = time.monotonic()
        admitted = self._last_admitted.setdefault(task.camera_name, {})
        for object_id, timestamp in list(admitted.items()):
            if now - timestamp > max(5.0, self.min_interval_sec * 2):
                del admitted[object_id]
        selected = []
        selected_ids = []
        for person in task.person_events:
            object_id = person.object_id
            tracked = type(object_id) is int and 0 <= object_id < 2**64 - 1
            # 실행이 0.5초를 넘으면 같은 인물의 오래된 작업이 줄지어 쌓이지 않게 한다.
            if (self.min_interval_sec > 0 and tracked
                    and (task.camera_name, task.generation, object_id) in self._inflight):
                self.record_stat("context_people_busy")
                continue
            previous = admitted.get(object_id) if tracked else None
            if previous is not None and now - previous < self.min_interval_sec:
                self.record_stat("context_people_sampled")
                continue
            selected.append(person)
            if tracked:
                selected_ids.append(object_id)
        if not selected:
            self.record_stat("context_frames_sampled")
            return False
        task = replace(task, person_events=selected)
        try:
            self.queue.put_nowait(task)
            # 큐 포화로 실패한 인물은 시각을 갱신하지 않아 다음 유효 프레임에서 재시도한다.
            for object_id in selected_ids:
                admitted[object_id] = now
                if self.min_interval_sec > 0:
                    self._inflight.add((task.camera_name, task.generation, object_id))
            self.record_stat("context_frames_matched")
            return True
        except Full:
            self.record_stat("context_queue_dropped")
            logger.debug("[%s] 얼굴 인식 워커 큐 가득 참 - 프레임 건너뜀", task.camera_name)
            return False

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
                self.record_stat("context_tasks_failed")
                logger.warning("[%s] 얼굴/외형 컨텍스트 후처리 실패: %s", task.camera_name, exc)

    def process(self, task: FaceContextTask) -> None:
        try:
            self._process_current(task)
        finally:
            # 실패·재연결 무효화에도 예약을 풀되 이전 세대가 새 작업의 예약을 지우지 않는다.
            with self._state_lock:
                for person in task.person_events:
                    self._inflight.discard((task.camera_name, task.generation, person.object_id))

    def _process_current(self, task: FaceContextTask) -> None:
        with self.commit_guard(task) as current:
            if not current:
                return
        face_events = (
            self.run_face_recognition(
                task.frame,
                task.person_events,
                task.camera_name,
                commit_guard=lambda: self.commit_guard(task),
            )
            if task.flags.get("use_face")
            else []
        )
        with self.commit_guard(task) as current:
            if not current:
                return
        appearance_events = self._run_appearance(task, face_events)
        with self.commit_guard(task) as current:
            if current:
                for event in face_events + appearance_events:
                    self.enqueue_event(event, task.camera_name)
                self.record_stat("context_tasks_completed")

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
            commit_guard=lambda: self.commit_guard(task),
        )
        for event in appearance_events:
            metadata = dict(event.metadata or {})
            metadata.setdefault("backend", "deepstream_context")
            metadata.setdefault("camera_id", task.camera_name)
            event.metadata = metadata
        return appearance_events
