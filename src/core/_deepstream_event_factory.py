"""DeepStream detection dict를 DetectionEvent로 변환한다."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from .events import DetectionEvent, EventType


def _add_frame_size_metadata(
    metadata: Dict[str, Any],
    *,
    frame_width: Optional[int],
    frame_height: Optional[int],
) -> None:
    if frame_width:
        metadata["frame_width"] = frame_width
    if frame_height:
        metadata["frame_height"] = frame_height


def object_meta_to_event(
    obj_meta: Any,
    *,
    camera_name: str,
    source_id: int,
    frame_num: int,
    timestamp_factory: Callable[[], float],
    event_type_for_label: Callable[[str], EventType],
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> Optional[DetectionEvent]:
    """DeepStream NvDsObjectMeta 형태의 객체를 DetectionEvent로 변환한다."""
    label = getattr(obj_meta, "obj_label", "") or ""
    event_type = event_type_for_label(str(label))
    if event_type == EventType.OTHER:
        return None

    rect = obj_meta.rect_params
    object_id = int(obj_meta.object_id)
    if object_id < 0:
        object_id = None

    metadata = {
        "backend": "deepstream",
        "camera_id": camera_name,
        "source_id": source_id,
        "frame_num": frame_num,
    }
    _add_frame_size_metadata(
        metadata,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    return DetectionEvent(
        event_type=event_type,
        x=int(rect.left),
        y=int(rect.top),
        width=int(rect.width),
        height=int(rect.height),
        confidence=float(obj_meta.confidence),
        timestamp=timestamp_factory(),
        object_id=object_id,
        class_idx=int(obj_meta.class_id),
        class_name=str(label),
        metadata=metadata,
    )


def detections_to_events(
    detections: Iterable[Dict[str, Any]],
    *,
    camera_name: str,
    source_id: int,
    frame_num: int,
    timestamp_factory: Callable[[], float],
    event_type_for_label: Callable[[str], EventType],
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> List[DetectionEvent]:
    """YOLO 후처리 detection dict 목록을 DeepStream 이벤트로 변환한다."""
    events: List[DetectionEvent] = []
    for detection in detections:
        event_type = event_type_for_label(str(detection["label"]))
        if event_type == EventType.OTHER:
            continue

        x, y, width, height = detection["box"]
        timestamp = timestamp_factory()
        base_metadata = {
            "backend": "deepstream_tensor",
            "camera_id": camera_name,
            "source_id": source_id,
            "frame_num": frame_num,
            "gie_id": int(detection.get("gie_id", 0)),
            "model_task": str(detection.get("task", "")),
        }
        resolved_frame_width = detection.get("frame_width", frame_width)
        resolved_frame_height = detection.get("frame_height", frame_height)
        _add_frame_size_metadata(
            base_metadata,
            frame_width=resolved_frame_width,
            frame_height=resolved_frame_height,
        )
        events.append(
            DetectionEvent(
                event_type=event_type,
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=float(detection["confidence"]),
                timestamp=timestamp,
                object_id=None,
                class_idx=int(detection["class_id"]),
                class_name=detection["label"],
                keypoints=detection.get("keypoints"),
                metadata=dict(base_metadata),
            )
        )
        if detection.get("is_fall"):
            metadata = dict(base_metadata)
            metadata["derived_from"] = "pose"
            events.append(
                DetectionEvent(
                    event_type=EventType.FALL_DETECTED,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=float(detection["confidence"]),
                    timestamp=timestamp_factory(),
                    object_id=None,
                    class_idx=int(detection["class_id"]),
                    class_name=detection["label"],
                    keypoints=detection.get("keypoints"),
                    metadata=metadata,
                )
            )
    return events
