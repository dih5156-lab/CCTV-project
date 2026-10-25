"""DeepStream preview frame 상태 저장소."""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

# Application-owned NvDsFrameMeta.misc_frame_info[0:3]; reserved fields stay untouched.
FRAME_IDENTITY_MARKER = 0x43435456


def read_frame_identity(frame: Any, camera_id: str) -> Optional[dict]:
    """변환 전후에 보존된 앱 토큰과 원본 프레임 식별자를 읽는다."""
    fields = getattr(frame, "misc_frame_info", None)
    if fields is None or int(fields[0]) != FRAME_IDENTITY_MARKER:
        return None
    pts = int(frame.buf_pts)
    if int(frame.frame_num) < 0 or not 0 <= pts < 2**64 - 1 or int(fields[2]) <= 0:
        return None
    return dict(camera_id=camera_id, source_id=int(frame.source_id),
                frame_num=int(frame.frame_num), source_pts_ns=pts,
                generation=int(fields[1]), frame_token=int(fields[2]))


class PreviewFrameStore:
    """카메라별 최신 preview frame과 샘플링 시간을 관리한다."""

    def __init__(self, max_fps: float) -> None:
        self.max_fps = max_fps
        self.min_interval_sec = 1.0 / max_fps if max_fps > 0 else 0.0
        self.last_frame_at: Optional[float] = None
        self.last_sample_at = 0.0
        self._frames: dict[str, Any] = {}
        self._frame_info: dict[str, dict] = {}
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
        identity: Optional[dict] = None,
    ) -> None:
        """카메라별 최신 frame과 샘플링 시간을 갱신한다."""
        with self._lock:
            self._frames[camera_id] = frame
            self.last_frame_at = time.time() if wall_time is None else wall_time
            self.last_sample_at = now_monotonic
            self._frame_info[camera_id] = {
                **(identity or {}),
                "received_monotonic": now_monotonic,
                "received_at": self.last_frame_at,
            }

    def get_frame_info(self, camera_id: str) -> Optional[dict]:
        with self._lock:
            info = self._frame_info.get(camera_id)
            return dict(info) if info is not None else None

    def get_frame_with_info(self, camera_id: str):
        """갱신 중에도 영상과 식별 정보가 같은 프레임을 가리키도록 함께 읽는다."""
        with self._lock:
            frame = self._frames.get(camera_id)
            info = self._frame_info.get(camera_id)
            if frame is None or info is None:
                return None
            return frame.copy(), dict(info)

    def clear_camera(self, camera_id: str) -> None:
        with self._lock:
            self._frames.pop(camera_id, None)
            self._frame_info.pop(camera_id, None)
            self.last_frame_at = max(
                (info["received_at"] for info in self._frame_info.values()), default=None
            )
            self.last_sample_at = 0.0

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


def read_single_preview_identity(
    buffer: Any, *, pyds_module: Any, pad_to_camera: Mapping[int, str]
) -> Optional[dict]:
    """CPU 단일 surface와 대응 가능한 native frame 식별자만 반환한다."""
    batch = pyds_module.gst_buffer_get_nvds_batch_meta(hash(buffer))
    if batch is None or int(batch.num_frames_in_batch) != 1:
        return None
    node = batch.frame_meta_list
    if node is None:
        return None
    frame = pyds_module.NvDsFrameMeta.cast(node.data)
    source_id = int(frame.source_id)
    camera_id = pad_to_camera.get(source_id)
    if camera_id is None or int(frame.batch_id) != 0 or int(frame.frame_num) < 0:
        return None
    source_pts, buffer_pts = int(frame.buf_pts), int(buffer.pts)
    return {
        "camera_id": camera_id, "source_id": source_id,
        "frame_num": int(frame.frame_num),
        "source_pts_ns": source_pts if 0 <= source_pts < 2**64 - 1 else None,
        "buffer_pts_ns": buffer_pts if 0 <= buffer_pts < 2**64 - 1 else None,
        **(read_frame_identity(frame, camera_id) or {}),
    }


def process_preview_sample(
    *,
    sink: Any,
    preview_store: PreviewFrameStore,
    preview_camera_id: Optional[str],
    cameras: Mapping[str, Any],
    gst_module: Any,
    read_frame_identity: Optional[Callable[[Any], Optional[dict]]] = None,
    on_frame_stored: Optional[Callable[[str, Any, dict], None]] = None,
    record_stat: Optional[Callable[[str], Any]] = None,
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

    try:
        identity = read_frame_identity(buffer) if read_frame_identity else None
    except Exception as exc:
        logger.debug("preview frame 식별 실패: %s", exc)
        identity = None
    if identity is None or identity.get("camera_id") not in cameras:
        if record_stat:
            record_stat("preview_identity_rejected")
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

        camera_id = identity["camera_id"]
        preview_store.put_frame(
            camera_id,
            frame,
            now_monotonic=now_monotonic,
            identity=identity,
        )
        if record_stat:
            record_stat("preview_frames_stored")
        if on_frame_stored:
            on_frame_stored(camera_id, frame, identity)
    except Exception as exc:
        logger.debug("DeepStream preview sample 처리 실패: %s", exc)
    finally:
        buffer.unmap(map_info)

    return gst_module.FlowReturn.OK
