"""ROI, pose and events must describe the mux surface, including its padding."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from src.core import deepstream_processor as ds
from src.core._deepstream_event_factory import (
    emit_tensor_events,
    object_meta_events_from_frame,
)
from src.core._preview_frame_store import FRAME_IDENTITY_MARKER
from src.core.event_debouncer import EventDebouncer
from src.core.events import EventType


@pytest.fixture
def coordinate_case(monkeypatch):
    monkeypatch.setenv("DS_STREAM_WIDTH", "1920")
    monkeypatch.setenv("DS_STREAM_HEIGHT", "1080")
    monkeypatch.setenv("DS_YOLO_INPUT_SIZE", "640")
    proc = object.__new__(ds.DeepStreamProcessor)
    proc._pose_gie_id, proc._helmet_gie_id, proc._person_gie_id = 1, 2, 1
    proc._crop_pose_enabled = False
    proc._task_by_gie, proc._labels_by_gie = {}, {}
    proc._class_ids_by_gie, proc._confidence_by_gie = {}, {}
    proc._yolo_task, proc._yolo_labels = "pose", ["person"]
    proc._yolo_class_ids, proc._yolo_conf_threshold = {0}, 0.5
    proc._yolo_iou_threshold, proc._yolo_max_detections = 0.5, 10
    proc._yolo_postprocess_mode = "vectorized"
    proc._is_fall_pose = lambda *args: False
    proc._is_valid_person_pose = lambda *args: True
    proc._record_yolo_postprocess_timing = Mock()
    proc._filter_detections_for_camera = lambda detections, camera: detections
    # mux box (600,300,300,600), point (750,450), letterboxed to 640x640.
    output = np.zeros((56, 100), dtype=np.float32)
    output[:5, 0] = [250, 340, 100, 200, 0.9]
    output[5:, 0] = np.tile([250, 290, 0.9], 17)
    monkeypatch.setattr(ds, "select_yolo_output", lambda *args: object())
    monkeypatch.setattr(ds, "layer_to_numpy", lambda *args: output)
    added = []
    fake_pyds = NS(
        NVDSINFER_TENSOR_OUTPUT_META=100,
        NvDsUserMeta=NS(cast=lambda value: value),
        NvDsInferTensorMeta=NS(cast=lambda value: value),
        NvDsObjectMeta=NS(cast=lambda value: value),
        nvds_acquire_obj_meta_from_pool=lambda batch: NS(rect_params=NS()),
        nvds_add_obj_meta_to_frame=lambda frame, obj, parent: added.append(obj),
    )
    monkeypatch.setattr(ds, "pyds", fake_pyds)
    tensor = NS(unique_id=1)
    user = NS(base_meta=NS(meta_type=100), user_meta_data=tensor)
    frame = NS(source_frame_width=3840, source_frame_height=2160,
               source_id=0, frame_num=1, bInferDone=False,
               frame_user_meta_list=NS(data=user, next=None), obj_meta_list=None)
    return proc, frame, tensor, fake_pyds, added


@pytest.mark.parametrize("source_size", [(3840, 2160), (1920, 1080), (1280, 960), (1080, 1920)])
def test_tensor_roi_and_events_share_mux_coordinates(coordinate_case, source_size):
    proc, frame, tensor, pyds, added = coordinate_case
    frame.source_frame_width, frame.source_frame_height = source_size
    detections = proc._detections_from_tensor(tensor, frame)
    assert detections[0]["box"] == (600, 300, 300, 600)
    assert detections[0]["keypoints"][0][:2] == pytest.approx([750, 450])
    assert proc._inject_primary_person_object_meta(object(), frame, "cam") == 1
    rect = added[0].rect_params
    assert (rect.left, rect.top, rect.width, rect.height) == (600, 300, 300, 600)
    added[0].object_id = 37
    frame.obj_meta_list = NS(data=added[0], next=None)
    def flags(camera):
        return {"use_appearance": True, "use_fall": True, "use_helmet": True}
    tracker_events = object_meta_events_from_frame(
        frame_meta=frame, camera_name="cam", pyds_module=pyds,
        pphuman_sgie_enabled=False, feature_flags_for_camera=flags,
        decode_pphuman_for_obj=lambda obj: {}, event_type_for_label=proc._event_type_for_label)
    captured = []
    emit_tensor_events(
        batch_meta=object(), frame_meta=frame, camera_name="cam", pyds_module=pyds,
        detections_from_tensor=proc._detections_from_tensor, add_osd_overlays=lambda *args: None,
        apply_existing_event_pipeline=lambda camera, events: captured.extend(events),
        feature_flags_for_camera=flags, event_type_for_label=proc._event_type_for_label,
        object_meta_events_for_frame=lambda *args: tracker_events)
    for event in tracker_events + captured:
        assert (event.metadata["frame_width"], event.metadata["frame_height"]) == (1920, 1080)
        assert event.object_id == 37
        assert (event.x, event.y, event.width, event.height) == (600, 300, 300, 600)


@pytest.mark.parametrize("crop_enabled", [False, True])
@pytest.mark.parametrize("box,expected", [
    ((-20, -10, 100, 100), (0, 0, 80, 90)),
    ((1900, 1000, 100, 100), (1900, 1000, 20, 80)),
    ((1920, 100, 100, 100), None),
    ((100, 1080, 100, 100), None),
    ((100, 100, 0, 100), None),
])
def test_native_roi_clips_edges_and_rejects_empty(coordinate_case, crop_enabled, box, expected):
    proc, frame, tensor, pyds, added = coordinate_case
    proc._crop_pose_enabled = crop_enabled
    proc._detections_from_tensor = lambda *args: [dict(label="person", box=box, confidence=0.9)]
    assert proc._inject_primary_person_object_meta(object(), frame, "cam") == int(expected is not None)
    assert len(added) == int(expected is not None)
    if expected is not None:
        rect = added[0].rect_params
        assert (rect.left, rect.top, rect.width, rect.height) == expected


def test_tensor_respects_nondefault_mux_size(coordinate_case, monkeypatch):
    proc, frame, tensor, _, _ = coordinate_case
    monkeypatch.setenv("DS_STREAM_WIDTH", "1280")
    monkeypatch.setenv("DS_STREAM_HEIGHT", "960")
    detection = proc._detections_from_tensor(tensor, frame)[0]
    assert detection["box"] == (400, 320, 200, 400)
    assert detection["keypoints"][0][:2] == pytest.approx([500, 420])


@pytest.mark.parametrize("source_size", [(3840, 2160), (640, 480)])
def test_crop_pose_restores_points_and_metadata_to_mux(coordinate_case, source_size, monkeypatch):
    proc, frame, _, _, _ = coordinate_case
    monkeypatch.setenv("DS_POSE_INPUT_SIZE", "640")
    frame.source_frame_width, frame.source_frame_height = source_size
    proc._crop_pose_results = 0
    proc._add_osd_overlays = Mock()
    captured = []
    proc._filter_events_for_camera = lambda events, camera: events
    proc._apply_existing_event_pipeline = lambda camera, events: captured.extend(events)
    frame.obj_meta_list = NS(data=NS(
        class_id=0, object_id=37,
        rect_params=NS(left=1500, top=400, width=300, height=600),
        obj_user_meta_list=frame.frame_user_meta_list), next=None)
    assert proc._emit_crop_pose_events(object(), frame, "cam") == 1
    event = captured[0]
    assert (event.x, event.y, event.width, event.height) == (1500, 400, 300, 600)
    assert event.keypoints[0][:2] == pytest.approx([1584.375, 671.875])
    assert (event.metadata["frame_width"], event.metadata["frame_height"]) == (1920, 1080)
    assert event.object_id == 37


@pytest.mark.parametrize("path", ["tensor", "object", "crop"])
def test_all_deepstream_paths_deliver_pts_to_fall_debouncer(
    coordinate_case, monkeypatch, debounce_clock, path
):
    """모델 연산만 대체하고 실제 메타 변환·공통 판정 경로의 영상시계를 확인한다."""
    proc, frame, _, pyds, _ = coordinate_case
    proc._debouncer = EventDebouncer(NS(events=NS(
        debounce_enabled=True, fall_sustained_seconds=2.,
        fall_gap_reset_seconds=5., fall_resend_cooldown=60.,
    )), lambda *_: None)
    proc._crop_pose_results = 0
    proc._filter_events_for_camera = lambda events, camera: events
    proc._add_osd_overlays = lambda *_: None
    detections = [dict(box=(100, 100, 120, 70), confidence=.9, class_id=0,
                       label="person", object_id=37, is_fall=True)]
    monkeypatch.setattr(ds, "detections_from_yolo_output", lambda *args, **kwargs: detections)
    frame.obj_meta_list = NS(data=NS(
        class_id=0, object_id=37, obj_label="fall_detected", confidence=.9,
        rect_params=NS(left=100, top=100, width=120, height=70),
        obj_user_meta_list=frame.frame_user_meta_list), next=None)
    decisions = []

    def consume(camera, events):
        for event in events:
            if event.event_type == EventType.FALL_DETECTED:
                assert event.timestamp > 1_000_000_000
                decisions.append(proc._should_enqueue_event(event, camera))

    proc._apply_existing_event_pipeline = consume
    for seconds in range(3):
        debounce_clock.now = 1_000_000 + seconds * 4
        frame.frame_num, frame.buf_pts = seconds * 30, seconds * 1_000_000_000
        frame.misc_frame_info = [FRAME_IDENTITY_MARKER, 1, seconds + 1]
        if path == "tensor":
            emit_tensor_events(
                batch_meta=object(), frame_meta=frame, camera_name="cam", pyds_module=pyds,
                detections_from_tensor=lambda *_: detections, add_osd_overlays=lambda *_: None,
                apply_existing_event_pipeline=consume,
                feature_flags_for_camera=lambda _: {"use_pose": True},
                event_type_for_label=proc._event_type_for_label,
            )
        elif path == "object":
            consume("cam", object_meta_events_from_frame(
                frame_meta=frame, camera_name="cam", pyds_module=pyds,
                pphuman_sgie_enabled=False, feature_flags_for_camera=lambda _: {"use_pose": True},
                decode_pphuman_for_obj=lambda _: {}, event_type_for_label=proc._event_type_for_label,
            ))
        else:
            proc._emit_crop_pose_events(object(), frame, "cam")
    assert decisions == [False, False, True]
