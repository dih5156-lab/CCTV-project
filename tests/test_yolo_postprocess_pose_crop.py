import numpy as np
import pytest

from src.core._yolo_postprocess import (
    attach_pose_results_to_tracks,
    expand_bbox_for_pose_crop,
    extract_pose_crops,
    restore_crop_keypoints,
)
from src.core.ai._fall_detector import FallDetector


def test_expand_bbox_for_pose_crop_adds_margin_and_clips_to_frame() -> None:
    assert expand_bbox_for_pose_crop(
        (10, 20, 50, 80), frame_width=100, frame_height=100, margin_ratio=0.25
    ) == (0, 5, 60, 95)


def test_expand_bbox_for_pose_crop_rejects_invalid_bbox() -> None:
    with pytest.raises(ValueError):
        expand_bbox_for_pose_crop((10, 10, 10, 20), frame_width=100, frame_height=100)


def test_restore_crop_keypoints_adds_crop_offset_and_clips() -> None:
    values = np.asarray([[2, 3, 0.9], [99, 98, 0.7]], dtype=np.float32)
    restored = restore_crop_keypoints(
        values, (10, 20, 50, 80), frame_width=100, frame_height=100
    )
    assert restored[0] == pytest.approx([12.0, 23.0, 0.9])
    assert restored[1] == pytest.approx([99.0, 99.0, 0.7])


def test_restore_crop_keypoints_rejects_invalid_crop() -> None:
    with pytest.raises(ValueError):
        restore_crop_keypoints([[1, 2, 0.5]], (5, 5, 5, 8))


def test_restore_crop_keypoints_returns_empty_for_missing_pose() -> None:
    assert restore_crop_keypoints(None, (1, 2, 10, 20)) == []
    assert restore_crop_keypoints(np.empty((0, 3)), (1, 2, 10, 20)) == []


def test_extract_pose_crops_returns_independent_crop_and_origin() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[10:20, 20:30] = 7
    crops = extract_pose_crops(frame, [(20, 10, 30, 20)], margin_ratio=0)
    crop, bbox = crops[0]
    assert bbox == (20, 10, 30, 20)
    assert crop.shape == (10, 10, 3)
    assert int(crop[0, 0, 0]) == 7
    crop[0, 0] = 99
    assert int(frame[10, 20, 0]) == 7


def test_extract_pose_crops_empty_input_returns_empty_list() -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    assert extract_pose_crops(frame, []) == []


def test_attach_pose_results_to_tracks_restores_coordinates_and_ids() -> None:
    attached = attach_pose_results_to_tracks(
        [[[1, 2, 0.8]], None],
        [(10, 20, 30, 40), (0, 0, 10, 10)],
        [42, 43],
    )
    assert attached[0]["track_id"] == 42
    assert attached[0]["keypoints"][0][:2] == [11.0, 22.0]
    assert attached[1]["keypoints"] == []


def test_attach_pose_results_to_tracks_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        attach_pose_results_to_tracks([[[1, 2, 0.8]]], [(0, 0, 10, 10)], [1, 2])


def test_crop_pose_cpu_contract_preserves_track_and_fall_input_coordinates() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    crops = extract_pose_crops(frame, [(40, 20, 100, 100)], margin_ratio=0)
    _, crop_bbox = crops[0]
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[0] = [30, 20, 0.9]
    keypoints[5] = [10, 30, 0.9]
    keypoints[6] = [50, 32, 0.9]
    keypoints[11] = [10, 35, 0.9]
    keypoints[12] = [50, 37, 0.9]
    attached = attach_pose_results_to_tracks([keypoints], [crop_bbox], [9])
    assert attached[0]["track_id"] == 9
    assert attached[0]["keypoints"][0][:2] == [70.0, 40.0]
    assert isinstance(FallDetector(score_threshold=2.0).detect_keypoints(
        attached[0]["keypoints"], bbox_width=60, bbox_height=80, bbox_y=20
    ), bool)
