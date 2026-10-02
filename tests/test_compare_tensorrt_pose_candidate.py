import importlib
from pathlib import Path

import numpy as np
import pytest


def comparison():
    assert Path("scripts/ops/compare_tensorrt_pose_candidate.py").is_file()
    return importlib.import_module("scripts.ops.compare_tensorrt_pose_candidate")


def test_shape_mismatch_is_rejected_before_numeric_comparison():
    with pytest.raises(ValueError, match="shape"):
        comparison().pose_tensor_difference(np.zeros((1, 56, 2)), np.zeros((2, 56, 2)))


def test_nonfinite_outputs_cannot_look_like_valid_differences():
    candidate = np.zeros((1, 56, 2))
    candidate[0, 4, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        comparison().pose_tensor_difference(np.zeros((1, 56, 2)), candidate)


def test_numeric_difference_reports_scores_separately_from_coordinates():
    reference = np.zeros((1, 56, 2))
    candidate = reference.copy()
    reference[0, 4, 0] = 0.5
    candidate[0, 4, 0] = 0.6
    candidate[0, 0, 0] = 2
    result = comparison().pose_tensor_difference(reference, candidate)
    assert result["max_abs"] == 2
    assert result["person_score_max_abs"] == pytest.approx(0.1)
    assert result["foreground_proposals"] == 1
    assert result["foreground_max_abs"] == 2
    assert result["numerically_close"] is False


def test_identical_empty_scene_is_numeric_match_not_quality_acceptance():
    tensor = np.zeros((1, 56, 2))
    result = comparison().pose_tensor_difference(tensor, tensor.copy())
    assert result["numerically_close"] is True
    assert result["foreground_proposals"] == 0
    assert result["foreground_max_abs"] is None


def test_letterbox_preserves_aspect_ratio_rgb_order_and_normalization():
    frame = np.full((2, 4, 3), [10, 20, 30], dtype=np.uint8)
    result = comparison().letterbox_pose(frame, size=8)
    assert result.shape == (1, 3, 8, 8)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, :, 3, 3], [30 / 255, 20 / 255, 10 / 255])
    np.testing.assert_allclose(result[0, :, 0, 0], [114 / 255] * 3)
