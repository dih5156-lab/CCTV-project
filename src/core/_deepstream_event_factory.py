"""DeepStream detection dict를 DetectionEvent로 변환한다."""

from __future__ import annotations

import time
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
    object_id: Optional[int] = int(obj_meta.object_id)
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
        if detection.get("fall_near_miss") is not None:
            base_metadata["fall_near_miss"] = detection.get("fall_near_miss")
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
            if detection.get("fall_score") is not None:
                metadata["fall_score"] = detection.get("fall_score")
            if detection.get("fall_reasons") is not None:
                metadata["fall_reasons"] = detection.get("fall_reasons")
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


def filter_detections_for_camera(
    detections: List[Dict[str, Any]],
    *,
    camera_name: str,
    feature_flags_for_camera: Callable[[str], Dict[str, bool]],
    event_type_for_label: Callable[[str], EventType],
) -> List[Dict[str, Any]]:
    """카메라별 모델 on/off 설정에 맞지 않는 tensor detection을 제거한다."""
    flags = feature_flags_for_camera(camera_name)
    filtered: List[Dict[str, Any]] = []
    for detection in detections:
        event_type = event_type_for_label(str(detection.get("label", "")))
        if event_type == EventType.FALL_DETECTED and not flags.get("use_pose"):
            continue
        if event_type == EventType.PERSON and not (
            flags.get("use_pose") or flags.get("use_person")
        ):
            continue
        if event_type in {EventType.HELMET, EventType.HEAD} and not flags.get("use_helmet"):
            continue
        filtered.append(detection)
    return filtered


def filter_events_for_camera(
    events: List[DetectionEvent],
    *,
    camera_name: str,
    feature_flags_for_camera: Callable[[str], Dict[str, bool]],
) -> List[DetectionEvent]:
    """카메라별 모델 on/off 설정에 맞지 않는 DetectionEvent를 제거한다."""
    flags = feature_flags_for_camera(camera_name)
    filtered: List[DetectionEvent] = []
    for event in events:
        if event.event_type == EventType.FALL_DETECTED and not flags.get("use_pose"):
            continue
        if event.event_type == EventType.PERSON and not (
            flags.get("use_pose") or flags.get("use_person")
        ):
            continue
        if event.event_type in {EventType.HELMET, EventType.HEAD} and not flags.get("use_helmet"):
            continue
        filtered.append(event)
    return filtered


def emit_tensor_events(
    *,
    batch_meta: Any,
    frame_meta: Any,
    camera_name: str,
    pyds_module: Any,
    detections_from_tensor: Callable[[Any, Any], List[Dict[str, Any]]],
    add_osd_overlays: Callable[[Any, Any, List[Dict[str, Any]]], None],
    apply_existing_event_pipeline: Callable[[str, List[DetectionEvent]], None],
    feature_flags_for_camera: Callable[[str], Dict[str, bool]],
    event_type_for_label: Callable[[str], EventType],
) -> int:
    """frame_user_meta_list의 tensor 결과를 이벤트로 변환하고 파이프라인에 전달한다."""
    detected = 0
    l_user = frame_meta.frame_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds_module.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break

        if user_meta.base_meta.meta_type == pyds_module.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds_module.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            detections = filter_detections_for_camera(
                detections_from_tensor(tensor_meta, frame_meta),
                camera_name=camera_name,
                feature_flags_for_camera=feature_flags_for_camera,
                event_type_for_label=event_type_for_label,
            )
            add_osd_overlays(batch_meta, frame_meta, detections)
            events = detections_to_events(
                detections,
                camera_name=camera_name,
                source_id=int(frame_meta.source_id),
                frame_num=int(frame_meta.frame_num),
                timestamp_factory=time.time,
                frame_width=int(getattr(frame_meta, "source_frame_width", 0) or 0),
                frame_height=int(getattr(frame_meta, "source_frame_height", 0) or 0),
                event_type_for_label=event_type_for_label,
            )
            detected += sum(
                1 for event in events if event.event_type != EventType.FALL_DETECTED
            )
            apply_existing_event_pipeline(camera_name, events)

        try:
            l_user = l_user.next
        except StopIteration:
            break
    return detected


def object_meta_events_from_frame(
    *,
    frame_meta: Any,
    camera_name: str,
    pyds_module: Any,
    pphuman_sgie_enabled: bool,
    feature_flags_for_camera: Callable[[str], Dict[str, bool]],
    decode_pphuman_for_obj: Callable[[Any], Dict[str, Any]],
    event_type_for_label: Callable[[str], EventType],
) -> List[DetectionEvent]:
    """object_meta_list를 DetectionEvent 목록으로 변환하고 카메라 플래그를 적용한다."""
    flags = feature_flags_for_camera(camera_name)
    attach_appearance = pphuman_sgie_enabled and flags.get("use_appearance", False)
    events: List[DetectionEvent] = []

    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds_module.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break

        event = object_meta_to_event(
            obj_meta,
            camera_name=camera_name,
            source_id=int(frame_meta.source_id),
            frame_num=int(frame_meta.frame_num),
            timestamp_factory=time.time,
            frame_width=int(getattr(frame_meta, "source_frame_width", 0) or 0),
            frame_height=int(getattr(frame_meta, "source_frame_height", 0) or 0),
            event_type_for_label=event_type_for_label,
        )
        if event is not None:
            if attach_appearance and int(obj_meta.class_id) == 0:
                pphuman_attrs = decode_pphuman_for_obj(obj_meta)
                if pphuman_attrs:
                    if event.metadata is None:
                        event.metadata = {}
                    event.metadata["appearance"] = pphuman_attrs
                    event.metadata["appearance_backend"] = "pphuman_sgie"
            events.append(event)

        try:
            l_obj = l_obj.next
        except StopIteration:
            break

    return filter_events_for_camera(
        events,
        camera_name=camera_name,
        feature_flags_for_camera=feature_flags_for_camera,
    )


def process_batch_frames(
    *,
    batch_meta: Any,
    pyds_module: Any,
    pad_to_camera: Dict[int, str],
    frames_processed: int,
    tensor_probe_warned: bool,
    cleanup_interval: int,
    cleanup_callback: Callable[[], None],
    emit_tensor_events_for_frame: Callable[[Any, Any, str], int],
    object_meta_events_for_frame: Callable[[Any, str], List[DetectionEvent]],
    apply_existing_event_pipeline: Callable[[str, List[DetectionEvent]], None],
    tensor_warn_log: Callable[[str], None],
) -> tuple[int, bool]:
    """batch frame 순회를 처리하고 갱신된 통계/경고 상태를 반환한다."""
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds_module.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        camera_name = pad_to_camera.get(frame_meta.source_id, "unknown")
        frames_processed += 1

        if cleanup_interval > 0 and frames_processed % cleanup_interval == 0:
            cleanup_callback()

        detected_from_tensor = emit_tensor_events_for_frame(
            batch_meta,
            frame_meta,
            camera_name,
        )
        apply_existing_event_pipeline(
            camera_name,
            object_meta_events_for_frame(frame_meta, camera_name),
        )

        if (
            detected_from_tensor == 0
            and frame_meta.frame_num % 300 == 0
            and not tensor_probe_warned
        ):
            tensor_warn_log(camera_name)
            tensor_probe_warned = True

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return frames_processed, tensor_probe_warned
