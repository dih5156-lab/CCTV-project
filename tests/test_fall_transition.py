"""실제 후면 포즈와 정상 착석으로 관측 이력·경계 조건을 검증한다."""

import json
from pathlib import Path

import pytest

from src.core.ai._fall_detector import FallDetector
from src.core.events import DetectionEvent, EventType


@pytest.fixture
def samples():
    return json.loads((Path(__file__).parent / "fixtures/fall_transition_20261015.json").read_text())


def pose(sample, pts, *, camera="cam", generation=1, object_id=1):
    x, y, width, height = sample["bbox"]
    return DetectionEvent(
        EventType.PERSON, x, y, width, height, sample["confidence"], 100.0,
        object_id=object_id, keypoints=sample["keypoints"],
        metadata={"frame_identity": {"camera_id": camera, "generation": generation,
                                     "source_pts_ns": round(pts * 1e9)}},
    )


def standing(model, samples, *, camera="cam", generation=1):
    for pts in (0.0, 0.1, 0.2):
        assert not model.observe_floor_transition(camera, pose(samples["rear_upright"], pts, camera=camera, generation=generation))


def test_rear_floor_requires_observed_standing_transition(samples):
    model = FallDetector(enable_folded_pose=True)
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 0))
    model.clear_camera_history("cam")
    standing(model, samples)
    assert model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.3))


def test_flat_floor_after_standing_is_not_suppressed_as_seated(samples):
    model = FallDetector(enable_folded_pose=True)
    assert not model.observe_floor_transition("cam", pose(samples["front_floor_960"], 0))
    model.clear_camera_history("cam")
    for pts in (0.0, 0.1, 0.2):
        assert not model.observe_floor_transition("cam", pose(samples["front_upright_960"], pts))
    assert model.observe_floor_transition("cam", pose(samples["front_floor_960"], 0.3))


@pytest.mark.parametrize("change", ["height", "vertical_leg", "upright_torso", "low_joint", "low_person"])
def test_flat_floor_transition_keeps_seated_and_visibility_guards(samples, change):
    model = FallDetector(enable_folded_pose=True)
    for pts in (0.0, 0.1, 0.2):
        model.observe_floor_transition("cam", pose(samples["front_upright_960"], pts))
    # 단일 평면 자세의 기존 보호 조건을 하강 비교 창(0.45초) 밖에서 검증한다.
    event = pose(samples["front_floor_960"], 0.8)
    if change == "height":
        event.height = 300
    elif change == "vertical_leg":
        event.keypoints[15][1] = event.keypoints[11][1] + 120
    elif change == "upright_torso":
        event.keypoints[5][0] = event.keypoints[11][0]
        event.keypoints[6][0] = event.keypoints[12][0]
    elif change == "low_joint":
        event.keypoints[13][2] = 0.49
    else:
        event.confidence = 0.49
    assert not model.observe_floor_transition("cam", event)


def test_normal_clip_starting_seated_cannot_supply_standing_evidence(samples):
    model = FallDetector(enable_folded_pose=True)
    for tick in range(50):
        assert not model.observe_floor_transition("cam", pose(samples["normal_seated_start"], tick / 10))


def test_transition_requires_explicit_folded_enable(samples):
    model = FallDetector()
    standing(model, samples)
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.3))


def test_single_or_duplicate_standing_frame_cannot_seed_transition(samples):
    model = FallDetector(enable_folded_pose=True)
    for _ in range(5):
        assert not model.observe_floor_transition("cam", pose(samples["rear_upright"], 0))
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.1))


@pytest.mark.parametrize("joint", [5, 6, 11, 12, 13, 14, 15, 16])
def test_transition_requires_both_observed_legs_and_upper_body(samples, joint):
    model = FallDetector(enable_folded_pose=True)
    standing(model, samples)
    samples["rear_floor"]["keypoints"][joint][2] = 0.49
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.3))


def test_transition_does_not_extend_missing_or_low_confidence_detection(samples):
    model = FallDetector(enable_folded_pose=True)
    standing(model, samples)
    event = pose(samples["rear_floor"], 0.3)
    event.confidence = 0.49
    assert not model.observe_floor_transition("cam", event)
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 2.0))


def test_transition_scopes_camera_track_generation_and_rewind(samples):
    model = FallDetector(enable_folded_pose=True)
    standing(model, samples)
    assert not model.observe_floor_transition("other", pose(samples["rear_floor"], 0.3, camera="other"))
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.3, object_id=2))
    assert model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.3))
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.4, generation=2))
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.5, generation=1))
    model.clear_camera_history("cam")
    standing(model, samples)
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.0))


def test_transition_reference_expires_without_new_standing(samples):
    model = FallDetector(enable_folded_pose=True)
    standing(model, samples)
    for tick in range(3, 90):
        result = model.observe_floor_transition("cam", pose(samples["rear_floor"], tick / 10))
    assert not result


@pytest.mark.parametrize("field,value", [("source_pts_ns", None), ("source_pts_ns", True), ("source_pts_ns", 2**64-1), ("camera_id", "wrong"), ("generation", -1)])
def test_transition_rejects_bad_frame_identity(samples, field, value):
    model = FallDetector(enable_folded_pose=True)
    standing(model, samples)
    event = pose(samples["rear_floor"], 0.3)
    event.metadata["frame_identity"][field] = value
    assert not model.observe_floor_transition("cam", event)


def test_transition_requires_source_identity_and_finite_pose(samples):
    model = FallDetector(enable_folded_pose=True)
    standing(model, samples)
    event = pose(samples["rear_floor"], 0.3)
    event.metadata = {}
    assert not model.observe_floor_transition("cam", event)
    event = pose(samples["rear_floor"], 0.4)
    event.keypoints[0][0] = float("nan")
    assert not model.observe_floor_transition("cam", event)


def test_large_floor_position_change_does_not_transfer_reference(samples):
    model = FallDetector(enable_folded_pose=True)
    for pts in (0.0, 0.1, 0.2):
        model.observe_floor_transition("cam", pose(samples["rear_distant_upright"], pts))
    assert not model.observe_floor_transition("cam", pose(samples["rear_floor"], 0.3))


def test_deepstream_transition_preserves_input_and_does_not_duplicate_static_fall(samples):
    from dataclasses import replace

    from src.core.deepstream_processor import DeepStreamProcessor

    processor = object.__new__(DeepStreamProcessor)
    processor._fall_detector = FallDetector(enable_folded_pose=True)
    processor._write_fall_near_miss_review_records = lambda *_: None
    for pts in (0.0, 0.1, 0.2):
        processor._observe_prepared_pose_events("cam", [pose(samples["rear_upright"], pts)])
    source = pose(samples["rear_floor"], 0.3)
    events = [source]
    processor._observe_prepared_pose_events("cam", events)
    assert len(events) == 2 and events[1].event_type == EventType.FALL_DETECTED
    assert events[1].confidence == source.confidence >= 0.5
    assert events[1].object_id == source.object_id
    assert events[1].metadata["frame_identity"] == source.metadata["frame_identity"]
    assert "fall_score" not in source.metadata
    assert events[1].metadata["fall_score"] == 3.0
    source = pose(samples["rear_floor"], 0.4)
    events = [source, replace(source, event_type=EventType.FALL_DETECTED)]
    processor._observe_prepared_pose_events("cam", events)
    assert len(events) == 2


def test_reference_count_is_bounded_per_camera(samples):
    model = FallDetector(enable_folded_pose=True)
    for track in range(300):
        model.observe_floor_transition("cam", pose(samples["rear_upright"], 0, object_id=track))
    assert len(model._standing_references) == 256
    model.clear_camera_history("cam")
    assert not model._standing_references and not model._transition_clocks


def test_initial_folded_floor_pose_has_no_standalone_fall_score(samples):
    import numpy as np

    model = FallDetector(enable_folded_pose=True)
    sample = samples["normal_folded_1080"]
    kpts = np.asarray(sample["keypoints"])
    assert model.folded_floor_pose_score(kpts, sample["bbox"][3]) is not None
    assert model._score_fall(kpts, *sample["bbox"][2:], bbox_y=sample["bbox"][1]).score < 3
    for tick in range(50):
        assert not model.observe_floor_transition("cam", pose(sample, tick / 10))


@pytest.mark.parametrize("occluded_leg", [None, "left", "right"])
def test_folded_floor_requires_standing_and_one_fully_observed_leg(samples, occluded_leg):
    model = FallDetector(enable_folded_pose=True)
    assert not model.observe_floor_transition("cam", pose(samples["side_folded_1080"], 0))
    model.clear_camera_history("cam")
    for pts in (0.0, 0.1, 0.2):
        model.observe_floor_transition("cam", pose(samples["side_upright_1080"], pts))
    event = pose(samples["side_folded_1080"], 0.4)
    if occluded_leg:
        for joint in ((13, 15) if occluded_leg == "left" else (14, 16)):
            event.keypoints[joint][2] = 0.49
    assert model.observe_floor_transition("cam", event)


@pytest.mark.parametrize("change", ["both_legs", "ankles", "upper_body", "person", "height"])
def test_folded_transition_keeps_current_observation_guards(samples, change):
    model = FallDetector(enable_folded_pose=True)
    for pts in (0.0, 0.1, 0.2):
        model.observe_floor_transition("cam", pose(samples["side_upright_1080"], pts))
    event = pose(samples["side_folded_1080"], 0.3)
    if change == "both_legs":
        event.keypoints[13][2] = event.keypoints[14][2] = 0.49
    elif change == "ankles":
        event.keypoints[15][2] = event.keypoints[16][2] = 0.49
    elif change == "upper_body":
        event.keypoints[5][2] = 0.49
    elif change == "person":
        event.confidence = 0.49
    else:
        event.height = samples["side_upright_1080"]["bbox"][3]
    assert not model.observe_floor_transition("cam", event)
