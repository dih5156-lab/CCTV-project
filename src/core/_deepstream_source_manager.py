"""DeepStream camera source attach/detach helper."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Union

from ._deepstream_source_state import (
    clear_source_state,
    remove_pad_mapping,
    set_source_state,
)

logger = logging.getLogger(__name__)


def detach_camera_source_from_pipeline(
    *,
    camera_id: str,
    info: Dict[str, Any],
    pad_to_camera: Dict[int, str],
    gst_module: Any,
    pipeline: Optional[Any],
    streammux: Optional[Any],
) -> Optional[int]:
    """카메라 source element와 streammux request pad를 파이프라인에서 제거한다."""
    previous_pad_id = info.get("pad_id")
    src = info.get("src_element")
    sinkpad = info.get("sinkpad")

    if src is not None:
        try:
            src.set_state(gst_module.State.NULL)
        except Exception as exc:
            logger.debug("[%s] source NULL 전환 실패: %s", camera_id, exc)
        if pipeline is not None:
            try:
                pipeline.remove(src)
            except Exception as exc:
                logger.debug("[%s] source pipeline 제거 실패: %s", camera_id, exc)

    if sinkpad is not None and streammux is not None and hasattr(streammux, "release_request_pad"):
        try:
            streammux.release_request_pad(sinkpad)
        except Exception as exc:
            logger.debug("[%s] streammux request pad 해제 실패: %s", camera_id, exc)

    remove_pad_mapping(pad_to_camera, previous_pad_id)
    clear_source_state(info)
    return previous_pad_id


def attach_camera_source_to_pipeline(
    *,
    camera_id: str,
    info: Dict[str, Any],
    pad_to_camera: Dict[int, str],
    gst_module: Any,
    pipeline: Any,
    streammux: Any,
    pad_id: Optional[int],
    make_element: Callable[[str, str], Any],
    normalize_uri: Callable[[Union[str, int]], str],
    on_source_pad_added: Callable[[Any, Any, Any], None],
    next_pad_id: Callable[[], int],
    detach_existing: bool = False,
) -> bool:
    """카메라 source를 nvstreammux에 연결한다."""
    try:
        source_uri = normalize_uri(info["source"])
    except ValueError as exc:
        logger.warning("[%s] DeepStream 소스 제외: %s", camera_id, exc)
        return False

    if detach_existing:
        previous_pad_id = detach_camera_source_from_pipeline(
            camera_id=camera_id,
            info=info,
            pad_to_camera=pad_to_camera,
            gst_module=gst_module,
            pipeline=pipeline,
            streammux=streammux,
        )
        if pad_id is None:
            pad_id = previous_pad_id

    if pad_id is None:
        pad_id = next_pad_id()

    src = make_element("nvurisrcbin", f"src-{camera_id}")
    src.set_property("uri", source_uri)
    try:
        src.set_property("latency", int(os.environ.get("DS_RTSP_LATENCY_MS", "200")))
    except TypeError:
        logger.debug("nvurisrcbin latency property 미지원")

    sinkpad = streammux.get_request_pad(f"sink_{pad_id}")
    if sinkpad is None:
        logger.warning("[%s] nvstreammux sink_%s pad 요청 실패", camera_id, pad_id)
        return False

    try:
        pipeline.add(src)
        src.connect("pad-added", on_source_pad_added, sinkpad)
        static_srcpad = src.get_static_pad("src")
        if static_srcpad is not None:
            on_source_pad_added(src, static_srcpad, sinkpad)
        if hasattr(src, "sync_state_with_parent"):
            src.sync_state_with_parent()
    except Exception as exc:
        logger.warning("[%s] source attach 중 오류: %s", camera_id, exc)
        try:
            streammux.release_request_pad(sinkpad)
        except Exception:
            pass
        try:
            pipeline.remove(src)
        except Exception:
            pass
        return False

    set_source_state(info, src_element=src, sinkpad=sinkpad, pad_id=pad_id)
    pad_to_camera[pad_id] = camera_id
    logger.info("[%s] DeepStream source attach 완료: pad_id=%s uri=%s", camera_id, pad_id, source_uri)
    return True
