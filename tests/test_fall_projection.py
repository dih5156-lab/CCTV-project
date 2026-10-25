"""실제 GPU 포즈로 전방 투영 누락과 가림 착석 오탐을 재현한다."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.ai._fall_detector import FallDetector


@pytest.fixture
def samples():
    return json.loads(
        (Path(__file__).parent / "fixtures/fall_projection_20261013.json").read_text()
    )


def detector(*, folded=False):
    return FallDetector(
        angle_horizontal=55, angle_inverted=125, bbox_aspect_ratio=1.35,
        span_bbox_aspect_ratio=1.2, span_ratio=0.55,
        min_keypoint_confidence=0.25, min_hip_confidence=0.25,
        min_leg_confidence=0.35, suppress_sitting_like_pose=True,
        enable_folded_pose=folded,
    )


def score(sample, model):
    _, y, width, height = sample["bbox"]
    return model._score_fall(np.asarray(sample["keypoints"]), width, height, bbox_y=y)


def test_front_fall_with_inverted_torso_and_legs_above_head(samples):
    result = score(samples["front_foreshortened"], detector())
    assert result.score >= 3.0
    assert "torso_inverted" in result.reasons
    assert "leg_above_head" in result.reasons


def test_occluded_sitting_does_not_supply_folded_floor_evidence(samples):
    sample = samples["occluded_sitting"]
    model = detector(folded=True)
    assert model.folded_floor_pose_score(sample["keypoints"], sample["bbox"][3]) is None
    assert score(sample, model).score < 3.0


def test_inverted_torso_without_ankles_keeps_single_signal_below_threshold(samples):
    sample = samples["front_foreshortened"]
    sample["keypoints"][15][2] = 0.0
    sample["keypoints"][16][2] = 0.0
    result = score(sample, detector())
    assert "torso_inverted" not in result.reasons
    assert result.score < 3.0


def test_rear_folded_alone_cannot_reach_alarm_score(samples):
    sample = samples["rear_folded"]
    assert score(sample, detector()).score < 3.0
    result = score(sample, detector(folded=True))
    assert result.score < 3.0
    assert any(reason.startswith("folded_floor_pose:") for reason in result.reasons)


def test_folded_requires_observed_ankle(samples):
    sample = samples["rear_folded"]
    sample["keypoints"][15][2] = 0.0
    sample["keypoints"][16][2] = 0.0
    assert score(sample, detector(folded=True)).score < 3.0


def test_head_below_hips_completes_front_fall_evidence(samples):
    result = score(samples["head_below_hips"], detector())
    assert result.score >= 3.0
    assert "head_below_hips" in result.reasons
    assert "leg_above_head" not in result.reasons


@pytest.mark.parametrize("name", ["normal_forward_bend", "normal_seated"])
def test_head_below_hips_does_not_promote_upright_head(samples, name):
    result = score(samples[name], detector())
    assert result.score < 3.0
    assert "head_below_hips" not in result.reasons


@pytest.mark.parametrize("joint", [0, 11, 12, 15, 16])
def test_head_below_hips_requires_clear_head_hips_and_both_ankles(samples, joint):
    sample = samples["head_below_hips"]
    sample["keypoints"][joint][2] = 0.0
    result = score(sample, detector())
    assert "head_below_hips" not in result.reasons
    assert result.score < 3.0


def test_head_below_hips_does_not_double_count_legs_above_head(samples):
    result = score(samples["front_foreshortened"], detector())
    assert "leg_above_head" in result.reasons
    assert "head_below_hips" not in result.reasons


@pytest.mark.parametrize("joint", [0, 11, 12])
def test_head_below_hips_rejects_uncertain_landmarks(samples, joint):
    sample = samples["head_below_hips"]
    sample["keypoints"][joint][2] = 0.49
    assert "head_below_hips" not in score(sample, detector()).reasons


def test_head_below_hips_rejects_small_vertical_change(samples):
    sample = samples["head_below_hips"]
    hips_y = max(sample["keypoints"][joint][1] for joint in [11, 12])
    sample["keypoints"][0][1] = hips_y + sample["bbox"][3] * 0.19
    assert score(sample, detector()).score < 3.0


@pytest.mark.parametrize("name", ["front_foreshortened", "rear_folded", "occluded_sitting", "head_below_hips"])
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_projection_score_preserves_scale_and_translation(samples, name, scale):
    sample = samples[name]
    model = detector(folded=True)
    expected = score(sample, model).score
    x, y, width, height = sample["bbox"]
    sample["bbox"] = [x * scale + 100, y * scale + 80, width * scale, height * scale]
    for point in sample["keypoints"]:
        point[0] = point[0] * scale + 100
        point[1] = point[1] * scale + 80
    assert score(sample, model).score == expected
