"""DeepStream preview frame 상태 저장소."""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


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


def process_preview_sample(
    *,
    sink: Any,
    preview_store: PreviewFrameStore,
    preview_camera_id: Optional[str],
    cameras: Mapping[str, Any],
    gst_module: Any,
) -> Any:
    """DeepStream appsink preview 샘플을 읽어 최신 프레임 저장소를 갱신한다."""
    now_monotonic = time.monotonic()
    sample = sink.emit("pull-sample")
    if sample is None:
        return gst_module.FlowReturn.OK

    if not preview_store.should_accept_sample(now_monotonic):
        return gst_module.FlowReturn.OK

    buffer = sample.get_buffer()
    caps = sample.get_caps()
    if buffer is None or caps is None or caps.get_size() == 0:
        return gst_module.FlowReturn.OK

    structure = caps.get_structure(0)
    width = int(structure.get_value("width") or 0)
    height = int(structure.get_value("height") or 0)
    pixel_format = str(structure.get_value("format") or "")
    if width <= 0 or height <= 0:
        return gst_module.FlowReturn.OK

    success, map_info = buffer.map(gst_module.MapFlags.READ)
    if not success:
        return gst_module.FlowReturn.OK

    try:
        import numpy as np

        data = np.frombuffer(map_info.data, dtype=np.uint8)
        if pixel_format == "BGRx":
            expected_size = width * height * 4
            if data.size < expected_size:
                return gst_module.FlowReturn.OK
            frame = np.ascontiguousarray(
                data[:expected_size].reshape((height, width, 4))[:, :, :3]
            )
        elif pixel_format == "BGR":
            expected_size = width * height * 3
            if data.size < expected_size:
                return gst_module.FlowReturn.OK
            frame = data[:expected_size].reshape((height, width, 3)).copy()
        else:
            logger.debug("지원하지 않는 DeepStream preview pixel format: %s", pixel_format)
            return gst_module.FlowReturn.OK

        camera_id = preview_camera_id or next(iter(cameras.keys()), "camera_1")
        preview_store.put_frame(
            camera_id,
            frame,
            now_monotonic=now_monotonic,
        )
    except Exception as exc:
        logger.debug("DeepStream preview sample 처리 실패: %s", exc)
    finally:
        buffer.unmap(map_info)

    return gst_module.FlowReturn.OK
