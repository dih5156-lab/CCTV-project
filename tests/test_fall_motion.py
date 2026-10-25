"""저장GPU 자세로 낙상 하강·일상 착석·숙임과 관측 경계를 검증한다."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.core.ai._fall_detector import FallDetector, _BodyObservation
from src.core.deepstream_processor import DeepStreamProcessor
from src.core.event_debouncer import EventDebouncer
from src.core.events import DetectionEvent, EventType


@pytest.fixture
def clips():
    return json.loads((Path(__file__).parent / "fixtures/fall_motion_20261016.json").read_text())


def event(row, *, camera="cam", generation=1, offset=0):
    x, y, width, height = row["bbox"]
    pts_ns = (row["pts_ns"] + round(offset * 1e9) if "pts_ns" in row
              else round((row["frame_index"] / 30 + offset) * 1e9))
    return DetectionEvent(
        EventType.PERSON, x, y, width, height, row["confidence"], 100,
        object_id=row["object_id"], keypoints=row["keypoints"],
        metadata={"frame_identity": {"camera_id": camera, "generation": generation,
                                     "source_pts_ns": pts_ns}},
    )


def test_descent_does_not_add_candidates_to_ordinary_seating(clips):
    model = FallDetector(enable_folded_pose=True)
    for row in clips["19"]["poses"]:
        assert not model.observe_floor_transition("cam", event(row))
        assert all(history.origin is None for history in model._descent_histories.values())


def test_rtsp_seating_joint_jitter_does_not_become_observed_descent(clips):
    model = FallDetector(enable_folded_pose=True)
    # 파일 재생에 없던 실제 RTSP 관절 흔들림과 원본 PTS를 함께 재현한다.
    for row in clips["rtsp_seating"]["poses"]:
        assert not model.observe_floor_transition("cam", event(row)), row["frame_index"]


@pytest.mark.parametrize("index,required_frame", [("development16", 252), ("14", 166), ("6", 114), ("72", 166)])
def test_seating_filter_retains_observed_falls_with_occlusion_or_short_legs(clips, index, required_frame):
    model = FallDetector(enable_folded_pose=True)
    clip = clips[index]
    candidates = [row["frame_index"] for row in clip["poses"]
                  if model.observe_floor_transition("cam", event(row))]
    # 발목 상승만 커진 14번의 162프레임 대신 몸통 하강을 재관측한 166프레임을 보존한다.
    # 다리 가림 뒤 재관측·직립 대비 높이 감소·수평 몸통을 각각 보존해야 한다.
    assert required_frame in candidates
    assert min(candidates) / 30 >= clip["onset"]


def test_slow_descent_does_not_extend_to_stationary_folded_pose(clips):
    model = FallDetector(enable_folded_pose=True)
    rows = clips["72"]["poses"]
    for row in rows:
        if row["frame_index"] >= 166:
            break
        model.observe_floor_transition("cam", event(row))
    observed = next(row for row in rows if row["frame_index"] == 166)
    assert model.observe_floor_transition("cam", event(observed))
    # 느린 하강의 현재 관측만 쓸 수 있고 같은 자세의 반복으로 경보 창을 채우면 안 된다.
    for tick in range(1, 11):
        result = model.observe_floor_transition("cam", event(observed, offset=tick / 15))
        if tick >= 7:
            assert not result


@pytest.mark.parametrize("change", ["camera", "track", "generation", "gap", "invalid", "recovery", "clear"])
def test_observed_drop_does_not_survive_person_or_stream_reset(clips, change):
    model = FallDetector(enable_folded_pose=True)
    rows = clips["development16"]["poses"]
    for row in rows:
        if row["frame_index"] >= 252:
            break
        model.observe_floor_transition("cam", event(row))
    target = event(next(row for row in rows if row["frame_index"] == 252))
    camera = "cam"
    if change == "camera":
        camera = target.metadata["frame_identity"]["camera_id"] = "other"
    elif change == "track":
        target.object_id += 1
    elif change == "generation":
        target.metadata["frame_identity"]["generation"] += 1
    elif change == "gap":
        target.metadata["frame_identity"]["source_pts_ns"] += 2_000_000_000
    elif change in ("invalid", "recovery"):
        reset = copy.deepcopy(target)
        reset.metadata["frame_identity"]["source_pts_ns"] -= 1
        if change == "invalid":
            reset.confidence = 0.49
        else:
            standing = event(rows[0])
            reset.keypoints = standing.keypoints
            reset.width, reset.height = standing.width, standing.height
        assert not model.observe_floor_transition(camera, reset)
    else:
        model.clear_camera_history(camera)
    assert not model.observe_floor_transition(camera, target)


@pytest.mark.parametrize("index", ["0", "22", "26", "46", "56", "74"])
def test_descent_retains_observed_floor_evidence_for_alarm_window(clips, index):
    model = FallDetector(enable_folded_pose=True)
    clip = clips[index]
    candidates = [row["frame_index"] / 30 for row in clip["poses"]
                  if model.observe_floor_transition("cam", event(row))]
    assert candidates
    assert candidates[0] >= clip["onset"]
    # 실제 후보 관측으로 기존2초 경보 창을 채워야 하며 벽시계는 쓰지 않는다.
    timely = [pts for pts in candidates if pts <= clip["onset"] + 3.3]
    assert timely[-1] - timely[0] >= 2.0


def test_squatting_from_bent_torso_does_not_start_descent_alarm(clips):
    model = FallDetector(enable_folded_pose=True)
    for row in clips["32"]["poses"]:
        if row["frame_index"] / 30 >= clips["32"]["onset"]:
            break
        assert not model.observe_floor_transition("cam", event(row))


def active_descent(clips, index="22"):
    """직립 또는 숙임에서 시작하는 실제 하강 표본을 사용한다."""
    model = FallDetector(enable_folded_pose=True)
    rows = clips[index]["poses"]
    for position, row in enumerate(rows):
        if model.observe_floor_transition("cam", event(row)):
            assert model._descent_histories[("cam", row["object_id"])].origin is not None
            return model, copy.deepcopy(rows[position + 1])
    raise AssertionError("하강 시작이 재현되지 않음")


@pytest.mark.parametrize("index", ["22", "74"])
@pytest.mark.parametrize("change", ["camera", "track", "generation", "rewind", "identity", "pts", "person", "shape", "nan", "upper", "legs", "gap"])
def test_descent_requires_same_valid_current_observation(clips, change, index):
    model, row = active_descent(clips, index)
    observed = event(row)
    identity = observed.metadata["frame_identity"]
    camera = "cam"
    if change == "camera":
        camera = identity["camera_id"] = "other"
    elif change == "track":
        observed.object_id = 2**63 + 1
    elif change == "generation":
        identity["generation"] = 2
    elif change == "rewind":
        identity["source_pts_ns"] = 0
    elif change == "identity":
        observed.metadata = {}
    elif change == "pts":
        identity["source_pts_ns"] = True
    elif change == "person":
        observed.confidence = 0.49
    elif change == "shape":
        observed.keypoints = []
    elif change == "nan":
        observed.keypoints[5][0] = float("nan")
    elif change == "upper":
        observed.keypoints[5][2] = 0.49
    elif change == "legs":
        observed.keypoints[15][2] = observed.keypoints[16][2] = 0.49
    elif change == "gap":
        identity["source_pts_ns"] += 2_000_000_000
    assert not model.observe_floor_transition(camera, observed)


def test_bent_cleaning_does_not_start_transition_before_actual_fall(clips):
    model = FallDetector(enable_folded_pose=True)
    for row in clips["74"]["poses"]:
        if row["frame_index"] / 30 >= clips["74"]["onset"]:
            break
        assert not model.observe_floor_transition("cam", event(row))


def test_descent_duplicate_pts_cannot_supply_alarm_evidence(clips):
    model, row = active_descent(clips)
    assert model.observe_floor_transition("cam", event(row))
    assert not model.observe_floor_transition("cam", event(row))


def test_descent_resets_on_recovery_and_expires_with_current_observations(clips):
    model, row = active_descent(clips)
    history = model._descent_histories[("cam", row["object_id"])]
    origin_pts = history.origin.pts
    for tick in range(1, 90):
        result = model.observe_floor_transition("cam", event(row, offset=tick / 10))
    assert row["frame_index"] / 30 + 8.9 - origin_pts > 8
    assert not result
    model, row = active_descent(clips)
    standing_row = copy.deepcopy(clips["0"]["poses"][0])
    standing_row["frame_index"] = row["frame_index"]
    assert not model.observe_floor_transition("cam", event(standing_row))
    assert model._descent_histories[("cam", row["object_id"])].origin is None


def test_descent_history_is_bounded_and_camera_clear_is_scoped(clips):
    model = FallDetector(enable_folded_pose=True)
    row = copy.deepcopy(clips["0"]["poses"][0])
    for camera in ("cam", "other"):
        for index in range(300):
            row["object_id"] = 2**63 + index
            model.observe_floor_transition(camera, event(row, camera=camera))
        assert sum(key[0] == camera for key in model._descent_histories) == 256
    model.clear_camera_history("cam")
    assert len(model._descent_histories) == 256
    assert all(key[0] == "other" for key in model._descent_histories)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_descent_is_invariant_to_image_scale_and_translation(clips, scale):
    models = [FallDetector(enable_folded_pose=True) for _ in range(2)]
    results = [[], []]
    for row in clips["22"]["poses"]:
        changed = copy.deepcopy(row)
        changed["bbox"] = [round(value * scale) + (100 if i < 2 else 0)
                           for i, value in enumerate(row["bbox"])]
        changed["keypoints"] = [[x * scale + 100, y * scale + 100, confidence]
                                for x, y, confidence in row["keypoints"]]
        for index, sample in enumerate((row, changed)):
            results[index].append(models[index].observe_floor_transition("cam", event(sample)))
    assert results[0] == results[1]


def test_descent_requires_common_foot_and_rejects_large_ground_shift():
    before = _BodyObservation(0, 400, 200, 80, (350, None))
    swapped = _BodyObservation(0.3, 200, 280, 180, (None, 350))
    moved = _BodyObservation(0.3, 200, 280, 180, (500, None))
    assert FallDetector._relative_body_drop(before, swapped) is None
    assert FallDetector._relative_body_drop(before, moved) is None
    matched = _BodyObservation(0.3, 200, 280, 180, (350, None))
    assert FallDetector._relative_body_drop(before, matched) == (0.2, 0.25)


def test_raised_feet_require_body_height_loss_before_counting_as_descent():
    before = _BodyObservation(0, 400, 200, 80, (350, 350))
    seated = _BodyObservation(0.3, 390, 200, 80, (270, 270))
    fallen = _BodyObservation(0.3, 250, 220, 180, (270, 270))
    assert FallDetector._relative_body_drop(before, seated) == (0.0, 0.0)
    assert FallDetector._relative_body_drop(before, fallen) == (0.25, 0.45)


@pytest.mark.parametrize("index", [str(i) for i in range(11)])
def test_rtsp_cadence_changes_do_not_turn_seating_into_alarm(index):
    # 실제 1080p·0.5 GPU 입력과 ID/PTS를 보존해 순간 후보의 재연결까지 검사한다.
    clips = json.loads((Path(__file__).parent / "fixtures/fall_cadence_20261019.json").read_text())
    processor = object.__new__(DeepStreamProcessor)
    processor._fall_detector = FallDetector(enable_folded_pose=True, angle_horizontal=55, angle_inverted=125)
    processor._write_fall_near_miss_review_records = lambda *_: None
    debouncer = EventDebouncer(SimpleNamespace(events=SimpleNamespace(
        debounce_enabled=True, fall_sustained_seconds=2.0, fall_gap_reset_seconds=5.0,
        fall_resend_cooldown=60.0, event_retention_hours=1)), lambda *_: None)
    for row in clips[index]["poses"]:
        observed = event(row)
        observed.event_type = EventType(row["event_type"])
        events = [observed]
        processor._observe_prepared_pose_events("cam", events)
        for candidate in events:
            if candidate.event_type == EventType.FALL_DETECTED:
                with patch("time.time", return_value=row["pts_ns"] / 1e9):
                    assert not debouncer.should_send("cam", "fall_detected", candidate.object_id, candidate)


@pytest.mark.parametrize("frame", [0, 60, 108])
def test_cleaning_with_supported_legs_is_not_a_static_fall(clips, frame):
    model = FallDetector(angle_horizontal=55, angle_inverted=125, bbox_aspect_ratio=1.35,
                         span_bbox_aspect_ratio=1.2, span_ratio=0.55,
                         min_keypoint_confidence=0.25, min_hip_confidence=0.25,
                         min_leg_confidence=0.35, suppress_sitting_like_pose=True)
    row = next(p for p in clips["74"]["poses"] if p["frame_index"] == frame)
    score = model._score_fall(np.asarray(row["keypoints"]), *row["bbox"][2:], bbox_y=row["bbox"][1])
    assert score.score < 3.0


@pytest.mark.parametrize("joint", [11, 12, 13, 14, 15, 16])
def test_supported_bend_suppression_requires_both_observed_legs(clips, joint):
    row = copy.deepcopy(clips["74"]["poses"][0])
    row["keypoints"][joint][2] = 0.49
    model = FallDetector()
    score = model._score_fall(np.asarray(row["keypoints"]), *row["bbox"][2:], bbox_y=row["bbox"][1])
    assert "supported_bend" not in score.reasons


def test_supported_bend_requires_pose_consistent_with_bbox(clips):
    row = copy.deepcopy(clips["74"]["poses"][0])
    model = FallDetector()
    assert "supported_bend" in model._score_fall(np.asarray(row["keypoints"]), *row["bbox"][2:], bbox_y=row["bbox"][1]).reasons
    assert "supported_bend" not in model._score_fall(np.asarray(row["keypoints"]), row["bbox"][2], 50, bbox_y=row["bbox"][1]).reasons


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_supported_bend_keeps_scale_and_translation_invariance(clips, scale):
    row = copy.deepcopy(clips["74"]["poses"][0])
    kpts = np.asarray(row["keypoints"])
    kpts[:, :2] = kpts[:, :2] * scale + 100
    score = FallDetector()._score_fall(kpts, round(row["bbox"][2] * scale),
                                     round(row["bbox"][3] * scale), bbox_y=round(row["bbox"][1] * scale) + 100)
    assert "supported_bend" in score.reasons and score.score < 3.0
