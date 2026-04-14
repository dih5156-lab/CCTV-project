"""카메라 인스턴스, 스레드, 재시도 큐 관리 레지스트리."""

from __future__ import annotations

import logging
import time
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig
from ..utils.camera_input import RTSPCamera

logger = logging.getLogger(__name__)


class _CameraRegistry:
    """카메라와 관련된 런타임 상태를 한곳에서 관리한다."""

    def __init__(
        self, config: AppConfig, stop_event: Event, is_running
    ) -> None:
        self._config = config
        self._stop_event = stop_event
        self._is_running = is_running

        self.cameras: Dict[str, RTSPCamera] = {}
        self.camera_threads: Dict[str, Thread] = {}
        self.inference_threads: Dict[str, Thread] = {}
        self.frame_queues: Dict[str, Queue] = {}
        self._stop_flags: Dict[str, Event] = {}

        self._pending: List[Tuple[str, Any, float]] = []
        self._pending_lock = Lock()

    @property
    def count(self) -> int:
        return len(self.cameras)

    def register(self, camera_id: str, camera: RTSPCamera) -> None:
        """이미 연결된 카메라를 레지스트리에 등록한다."""
        self.cameras[camera_id] = camera
        self.frame_queues[camera_id] = Queue(maxsize=1)
        self._stop_flags[camera_id] = Event()

    def unregister(self, camera_id: str) -> None:
        """카메라와 연결된 스레드 및 리소스를 정리한다."""
        timeout = self._config.processing.thread_join_timeout
        flag = self._stop_flags.pop(camera_id, None)
        if flag:
            flag.set()

        for thread_map in (self.camera_threads, self.inference_threads):
            thread = thread_map.pop(camera_id, None)
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning("[%s] 스레드 종료 시간 초과", camera_id)

        camera = self.cameras.pop(camera_id, None)
        if camera:
            camera.release()
        self.frame_queues.pop(camera_id, None)

    def stop_flag(self, camera_id: str) -> Optional[Event]:
        return self._stop_flags.get(camera_id)

    def ensure_stop_flag(self, camera_id: str) -> Event:
        """정지 플래그를 생성하거나 기존 플래그를 초기화한다."""
        if camera_id not in self._stop_flags:
            self._stop_flags[camera_id] = Event()
        else:
            self._stop_flags[camera_id].clear()
        return self._stop_flags[camera_id]

    def start_threads(self, camera_id: str, cam_target, inf_target) -> None:
        """카메라 스레드와 추론 스레드를 시작한다."""
        flag = self._stop_flags.get(camera_id)
        if flag:
            flag.clear()

        camera_thread = Thread(
            target=cam_target,
            args=(camera_id, self.cameras[camera_id]),
            daemon=True,
            name=f"Camera-{camera_id}",
        )
        self.camera_threads[camera_id] = camera_thread
        camera_thread.start()

        inference_thread = Thread(
            target=inf_target,
            args=(camera_id,),
            daemon=True,
            name=f"Inference-{camera_id}",
        )
        self.inference_threads[camera_id] = inference_thread
        inference_thread.start()

    def enqueue_retry(
        self, camera_id: str, source: Any, delay_seconds: float = 30.0
    ) -> None:
        """카메라 재연결을 예약한다."""
        next_ts = time.time() + delay_seconds
        with self._pending_lock:
            self._pending = [
                (cid, src, ts) for cid, src, ts in self._pending if cid != camera_id
            ]
            self._pending.append((camera_id, source, next_ts))
        logger.info("[%s] 재연결 예약: %.0f초 후", camera_id, delay_seconds)

    def poll_ready_retries(self) -> List[Tuple[str, Any, float]]:
        """실행 시간이 된 재시도 항목을 반환하고 내부 큐에서 제거한다."""
        now = time.time()
        ready: List[Tuple[str, Any, float]] = []
        remaining: List[Tuple[str, Any, float]] = []
        with self._pending_lock:
            for item in self._pending:
                (ready if now >= item[2] else remaining).append(item)
            self._pending = remaining
        return ready

    def pending_camera_ids(self) -> List[str]:
        """재연결 대기 중인 카메라 ID 목록을 반환한다."""
        with self._pending_lock:
            return [camera_id for camera_id, _, _ in self._pending]

    def set_all_stop_flags(self) -> None:
        """등록된 모든 카메라의 정지 플래그를 설정한다."""
        for flag in self._stop_flags.values():
            flag.set()
