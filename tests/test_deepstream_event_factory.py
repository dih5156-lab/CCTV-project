from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core._deepstream_event_factory import (
    detections_to_events,
    object_meta_to_event,
)
from src.core.events import EventType


@pytest.mark.parametrize("failing_phase", [None, "tensor", "object"])
def test_batch_finishes_metadata_only_after_both_event_paths(failing_phase):
    from src.core._deepstream_event_factory import process_batch_frames

    frame = SimpleNamespace(source_id=0, frame_num=1)
    batch = SimpleNamespace(frame_meta_list=SimpleNamespace(data=frame, next=None))
    phases = []

    def process(phase):
        phases.append(phase)
        if phase == failing_phase:
            raise RuntimeError(phase)
        return 0

    def run():
        return process_batch_frames(
            batch_meta=batch, pyds_module=SimpleNamespace(NvDsFrameMeta=SimpleNamespace(cast=lambda value: value)),
            pad_to_camera={0: "cam1"}, frames_processed=0, tensor_probe_warned=False,
            cleanup_interval=0, cleanup_callback=lambda: None,
            emit_tensor_events_for_frame=lambda *_: process("tensor"),
            object_meta_events_for_frame=lambda *_: [],
            apply_existing_event_pipeline=lambda *_: process("object"),
            tensor_warn_log=lambda *_: None,
            finish_frame=lambda *_: phases.append("finished"),
        )

    if failing_phase:
        with pytest.raises(RuntimeError, match=failing_phase):
            run()
        assert "finished" not in phases
    else:
        assert run() == (1, False)
        assert phases == ["tensor", "object", "finished"]


def _event_type_for_label(label: str) -> EventType:
    if label == "person":
        return EventType.PERSON
    return EventType.OTHER


def test_pose_fall_event_keeps_parent_tracker_id() -> None:
    events = detections_to_events(
        [
            {
                "box": (100, 200, 320, 640),
                "confidence": 0.91,
                "class_id": 0,
                "label": "person",
                "keypoints": [[120.0, 220.0, 0.9]],
                "is_fall": True,
                "fall_score": 4.2,
                "object_id": 37,
            }
        ],
        camera_name="camera_1",
        source_id=0,
        frame_num=123,
        timestamp_factory=lambda: 1.0,
        event_type_for_label=_event_type_for_label,
    )

    assert [event.event_type for event in events] == [
        EventType.PERSON,
        EventType.FALL_DETECTED,
    ]
    assert events[0].object_id == 37
    assert events[1].object_id == 37
    assert events[1].metadata["fall_score"] == 4.2


def test_untracked_deepstream_object_uses_synthetic_id_later() -> None:
    obj_meta = SimpleNamespace(
        obj_label="person",
        rect_params=SimpleNamespace(left=10, top=20, width=30, height=40),
        object_id=0xFFFFFFFFFFFFFFFF,
        confidence=0.8,
        class_id=0,
    )

    event = object_meta_to_event(
        obj_meta,
        camera_name="camera_1",
        source_id=0,
        frame_num=1,
        timestamp_factory=lambda: 1.0,
        event_type_for_label=_event_type_for_label,
    )

    assert event is not None
    assert event.object_id is None
