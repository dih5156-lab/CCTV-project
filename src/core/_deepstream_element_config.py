"""DeepStream GStreamer element 설정/링크 헬퍼."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


def set_optional_property(element: Any, name: str, value: Any) -> None:
    try:
        element.set_property(name, value)
    except TypeError:
        logger.debug("%s property 미지원: %s", element.get_name(), name)


def configure_streammux(streammux: Any, n_cams: int) -> None:
    streammux.set_property("batch-size", n_cams)
    streammux.set_property("width", int(os.environ.get("DS_STREAM_WIDTH", "1920")))
    streammux.set_property("height", int(os.environ.get("DS_STREAM_HEIGHT", "1080")))
    streammux.set_property(
        "batched-push-timeout",
        int(os.environ.get("DS_BATCH_TIMEOUT_US", "33333")),
    )
    streammux.set_property("live-source", 1)
    streammux.set_property("enable-padding", 1)
    try:
        streammux.set_property(
            "nvbuf-memory-type",
            int(os.environ.get("DS_NVBUF_MEMORY_TYPE", "0")),
        )
    except TypeError:
        logger.debug("nvstreammux nvbuf-memory-type property 미지원")


def configure_infer_elements(
    *,
    nvinfer: Optional[Any],
    helmet_infer: Optional[Any],
    pphuman_infer: Optional[Any],
    n_cams: int,
    infer_config: Path,
    helmet_infer_config: Path,
    pphuman_infer_config: Path,
    env_int: Callable[[str, int], int],
    set_property_optional: Callable[[Any, str, Any], None],
) -> None:
    if nvinfer is not None:
        nvinfer.set_property("config-file-path", str(infer_config))
        nvinfer.set_property("batch-size", n_cams)
        set_property_optional(nvinfer, "interval", env_int("DS_PRIMARY_INTERVAL", 0))
    if pphuman_infer is not None:
        pphuman_infer.set_property("config-file-path", str(pphuman_infer_config))
        pphuman_infer.set_property("batch-size", n_cams)
        set_property_optional(
            pphuman_infer,
            "interval",
            env_int("DS_PPHUMAN_INTERVAL", 4),
        )
        set_property_optional(
            pphuman_infer,
            "secondary-reinfer-interval",
            env_int("DS_PPHUMAN_REINFER_INTERVAL", 15),
        )
    if helmet_infer is not None:
        helmet_infer.set_property("config-file-path", str(helmet_infer_config))
        helmet_infer.set_property("batch-size", n_cams)
        set_property_optional(helmet_infer, "interval", env_int("DS_HELMET_INTERVAL", 1))


def configure_tracker(
    *,
    tracker: Any,
    tracker_lib: str,
    tracker_config: Path,
) -> None:
    if Path(tracker_lib).exists():
        tracker.set_property("ll-lib-file", tracker_lib)
    tracker.set_property("ll-config-file", str(tracker_config))
    tracker.set_property("tracker-width", int(os.environ.get("DS_TRACKER_WIDTH", "640")))
    tracker.set_property("tracker-height", int(os.environ.get("DS_TRACKER_HEIGHT", "384")))
    tracker.set_property("gpu-id", 0)
    try:
        tracker.set_property("enable-past-frame", 1)
    except TypeError:
        logger.debug("nvtracker enable-past-frame property 미지원")


def configure_output_queue(output_queue: Any) -> None:
    output_queue.set_property("leaky", 2)
    output_queue.set_property("max-size-buffers", 2)
    output_queue.set_property("max-size-bytes", 0)
    output_queue.set_property("max-size-time", 0)


def link_or_raise(first: Any, second: Any, message: Optional[str] = None) -> None:
    if not first.link(second):
        if message is None:
            message = f"{first.get_name()} -> {second.get_name()} link 실패"
        raise RuntimeError(message)


def link_preview_branch(
    *,
    osd: Any,
    tee: Any,
    output_queue: Any,
    preview_elements: List[Any],
    link: Callable[[Any, Any, Optional[str]], None],
) -> Any:
    link(osd, tee, "nvdsosd -> preview-tee link 실패")
    link(tee, output_queue, "preview-tee -> output-queue link 실패")
    if preview_elements:
        link(
            tee,
            preview_elements[0],
            "preview-tee -> preview-queue link 실패",
        )
        preview_previous = preview_elements[0]
        for element in preview_elements[1:]:
            link(preview_previous, element, None)
            preview_previous = element
    return output_queue
