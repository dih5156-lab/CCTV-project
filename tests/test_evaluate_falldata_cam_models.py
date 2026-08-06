import numpy as np

from scripts.datasets.evaluate_falldata_cam_models import (
    _camera_number,
    _prediction_summary,
    _transform_sequence,
)


def test_camera_number_accepts_manifest_and_model_formats() -> None:
    assert _camera_number(3) == 3
    assert _camera_number("camera_1") == 1
    assert _camera_number("FNF_RF_SMOTE_CAM_8") == 8
    assert _camera_number("unknown") is None


def test_prediction_summary_uses_fall_as_class_zero() -> None:
    summary = _prediction_summary(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    )

    assert summary["tp"] == 1
    assert summary["fn"] == 1
    assert summary["fp"] == 1
    assert summary["tn"] == 1
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5


def test_tail_align_moves_source_frames_to_sequence_end() -> None:
    sequence = np.zeros((600, 2), dtype=np.float32)
    sequence[:2] = [[1.0, 2.0], [3.0, 4.0]]

    transformed = _transform_sequence(
        sequence,
        source_frames=2,
        mode="tail_align",
    )

    assert np.count_nonzero(transformed[:-2]) == 0
    np.testing.assert_array_equal(transformed[-2:], sequence[:2])


def test_stretch_repeats_source_across_full_sequence() -> None:
    sequence = np.zeros((600, 1), dtype=np.float32)
    sequence[:2, 0] = [1.0, 2.0]

    transformed = _transform_sequence(
        sequence,
        source_frames=2,
        mode="stretch",
    )

    assert transformed[0, 0] == 1.0
    assert transformed[-1, 0] == 2.0
    assert np.count_nonzero(transformed) == 600
