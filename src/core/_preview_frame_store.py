"""DeepStream preview frame 상태 저장소."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Optional


class PreviewFrameStore:
    """카메라별 최신 preview frame과 샘플링 시간을 관리한다."""

    def __init__(self, max_fps: float) -> None:
        self.max_fps = max_fps
        self.min_interval_sec = 1.0 / max_fps if max_fps > 0 else 0.0
        self.last_frame_at: Optional[float] = None
        self.last_sample_at = 0.0
        self._frames: dict[str, Any] = {}
        self._lock = Lock()

    def should_accept_sample(self, now_monotonic: float) -> bool:
        """preview FPS 제한에 걸리지 않으면 True를 반환한다."""
        return not (
            self.min_interval_sec > 0
            and now_monotonic - self.last_sample_at < self.min_interval_sec
        )

    def put_frame(
        self,
        camera_id: str,
        frame: Any,
        *,
        now_monotonic: float,
        wall_time: Optional[float] = None,
    ) -> None:
        """카메라별 최신 frame과 샘플링 시간을 갱신한다."""
        with self._lock:
            self._frames[camera_id] = frame
            self.last_frame_at = time.time() if wall_time is None else wall_time
            self.last_sample_at = now_monotonic

    def get_frame(
        self,
        camera_id: str,
        *,
        fallback_camera_id: Optional[str] = None,
        copy_frame: bool = True,
    ) -> Optional[Any]:
        """요청 카메라 frame을 반환하고, 없으면 fallback 카메라 frame을 반환한다."""
        with self._lock:
            frame = self._frames.get(camera_id)
            if frame is None and fallback_camera_id:
                frame = self._frames.get(fallback_camera_id)
        if frame is None:
            return None
        return frame.copy() if copy_frame else frame
