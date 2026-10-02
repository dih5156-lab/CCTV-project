"""Primary GIE 출력의 카메라별 프레임/추론 관측 카운터."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Callable, Iterable, Mapping


class PrimaryFrameMetrics:
    """주 추론기의 tensor 출력 근거를 집계한다. bInferDone은 사용하지 않는다."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = Lock()
        self._cameras: dict[str, dict[str, Any]] = {}

    def reset(self, camera_ids: Iterable[str]) -> None:
        """파이프라인 재생성 시 현재 등록된 카메라로 계측을 초기화한다."""
        with self._lock:
            self._cameras = {
                camera_id: {
                    "frames_seen": 0,
                    "frames_inferred": 0,
                    "frames_skipped": 0,
                    "frames_inference_unknown": 0,
                    "last_frame_at": None,
                }
                for camera_id in camera_ids
            }

    def record_batch(
        self,
        batch_meta: Any,
        *,
        pyds_module: Any,
        pad_to_camera: Mapping[int, str],
        primary_element: Any = None,
    ) -> None:
        primary_id = self._observable_primary_id(primary_element)
        now = self._clock()
        node = batch_meta.frame_meta_list
        with self._lock:
            while node is not None:
                try:
                    frame = pyds_module.NvDsFrameMeta.cast(node.data)
                    camera_id = pad_to_camera.get(frame.source_id)
                    counters = (
                        self._cameras.get(camera_id) if camera_id is not None else None
                    )
                    if counters is not None:
                        counters["frames_seen"] += 1
                        counters["last_frame_at"] = now
                        infer_done = self._tensor_inference_done(
                            frame, pyds_module, primary_id
                        )
                        if infer_done is None:
                            counters["frames_inference_unknown"] += 1
                        elif infer_done:
                            counters["frames_inferred"] += 1
                        else:
                            counters["frames_skipped"] += 1
                    node = node.next
                except StopIteration:
                    break

    @staticmethod
    def _observable_primary_id(primary_element: Any) -> int | None:
        try:
            if primary_element.get_property("output-tensor-meta") is not True:
                return None
            component_id = primary_element.get_property("unique-id")
            if type(component_id) is int and component_id >= 0:
                return component_id
        except Exception:
            pass
        return None

    @staticmethod
    def _tensor_inference_done(
        frame: Any, pyds_module: Any, primary_id: int | None
    ) -> bool | None:
        if primary_id is None:
            return None
        try:
            node = frame.frame_user_meta_list
            while node is not None:
                meta = pyds_module.NvDsUserMeta.cast(node.data)
                if (
                    meta.base_meta.meta_type
                    == pyds_module.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                ):
                    tensor = pyds_module.NvDsInferTensorMeta.cast(meta.user_meta_data)
                    if type(tensor.unique_id) is not int or tensor.unique_id < 0:
                        return None
                    if tensor.unique_id == primary_id:
                        return True
                node = node.next
        except Exception:
            # 잘못된 메타를 '미추론'으로 세거나 다른 카메라 집계를 중단하지 않는다.
            return None
        return False

    def snapshot(self) -> dict[str, dict[str, Any]]:
        now = self._clock()
        with self._lock:
            result = {}
            for camera_id, counters in self._cameras.items():
                values = dict(counters)
                last_frame_at = values.pop("last_frame_at")
                values["last_frame_age_seconds"] = (
                    max(0.0, now - last_frame_at) if last_frame_at is not None else None
                )
                result[camera_id] = values
            return result
