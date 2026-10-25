"""YOLO tensor 후처리 유틸리티 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from src.core._yolo_postprocess import (
    detections_from_yolo_output,
    filter_yolo_candidates,
    filter_yolo_candidates_legacy,
    map_yolo_box_to_frame,
    nms_detections,
)


def test_map_yolo_box_to_frame_handles_letterbox_padding():
    box = [320, 320, 100, 100]

    mapped = map_yolo_box_to_frame(
        box,
        frame_width=640,
        frame_height=480,
        input_size=640,
    )

    assert mapped == (270, 190, 100, 100)


@pytest.mark.parametrize("input_size", [640, 960])
@pytest.mark.parametrize("frame_size", [(1920, 1080), (3840, 2160)])
def test_pose_resolution_preserves_coordinates_and_confidence(input_size, frame_size):
    # 입력 확대가 원본 좌표나 신뢰도0.5 필터를 바꾸지 않아야 한다.
    width, height = frame_size
    gain = input_size / width
    pad_y = (input_size - height * gain) / 2
    points = np.array([[width / 2 + j * 3, height / 2 + j * 2, 0.75] for j in range(17)])
    network_points = points.copy()
    network_points[:, :2] *= gain
    network_points[:, 1] += pad_y
    output = np.zeros((57, 56), dtype=np.float32)
    output[0, :5] = [input_size / 2, input_size / 2, input_size / 4, height / 3 * gain, 0.75]
    output[0, 5:] = network_points.reshape(-1)
    output[1] = output[0]
    output[1, 0] += input_size / 3
    output[1, 4] = 0.49

    detections = detections_from_yolo_output(
        output, task="pose", gie_id=1, labels=["person"],
        frame_width=width, frame_height=height, confidence_threshold=0.5,
        class_ids_filter={0}, input_size=input_size, iou_threshold=0.45,
        max_detections=10, fall_checker=lambda *_: False,
        person_pose_validator=lambda _: True,
    )

    assert len(detections) == 1
    assert detections[0]["box"] == (width * 3 // 8, height // 3, width // 4, height // 3)
    assert detections[0]["confidence"] == 0.75
    np.testing.assert_allclose(detections[0]["keypoints"], points, atol=1e-4)


def test_nms_detections_keeps_highest_confidence_overlap():
    detections = [
        {"box": (0, 0, 20, 20), "confidence": 0.9, "class_id": 0},
        {"box": (2, 2, 20, 20), "confidence": 0.8, "class_id": 0},
        {"box": (100, 100, 20, 20), "confidence": 0.7, "class_id": 0},
    ]

    kept = nms_detections(detections, iou_threshold=0.3, max_detections=10)

    assert [item["confidence"] for item in kept] == [0.9, 0.7]


def test_detections_from_yolo_output_decodes_detect_rows():
    output = np.array(
        [
            [320, 320, 100, 100, 0.1, 0.95],
            [100, 100, 50, 50, 0.3, 0.2],
            [110, 110, 50, 50, 0.2, 0.1],
            [120, 120, 50, 50, 0.2, 0.1],
            [130, 130, 50, 50, 0.2, 0.1],
            [140, 140, 50, 50, 0.2, 0.1],
        ],
        dtype=np.float32,
    )

    detections = detections_from_yolo_output(
        output,
        task="detect",
        gie_id=1,
        labels=["person", "helmet"],
        frame_width=640,
        frame_height=480,
        confidence_threshold=0.5,
        class_ids_filter={1},
        input_size=640,
        iou_threshold=0.45,
        max_detections=10,
        fall_checker=lambda keypoints, width, height, bbox_y: False,
        person_pose_validator=lambda keypoints: True,
    )

    assert len(detections) == 1
    assert detections[0]["label"] == "helmet"
    assert detections[0]["class_id"] == 1
    assert detections[0]["box"] == (270, 190, 100, 100)


def test_detections_from_yolo_output_passes_bbox_top_to_pose_checker():
    output = np.zeros((57, 56), dtype=np.float32)
    output[0, :5] = [320, 300, 100, 100, 0.9]
    observed_bbox_tops = []

    detections_from_yolo_output(
        output,
        task="pose",
        gie_id=1,
        labels=["person"],
        frame_width=640,
        frame_height=480,
        confidence_threshold=0.5,
        class_ids_filter={0},
        input_size=640,
        iou_threshold=0.45,
        max_detections=10,
        fall_checker=lambda keypoints, width, height, bbox_y: (
            observed_bbox_tops.append(bbox_y) or False
        ),
        person_pose_validator=lambda keypoints: True,
    )

    assert observed_bbox_tops == [170]


@pytest.mark.parametrize("invalid_highest_is_fall", [None, False, True])
def test_pose_scoring_skips_suppressed_duplicates_and_preserves_fall_exception(invalid_highest_is_fall):
    # 중복 제거 전 사람 검증에 실패해도 실제 낙상인 후보를 지우면 안 된다.
    output = np.zeros((57, 56), dtype=np.float32)
    for index, confidence in enumerate((0.9, 0.8, 0.7)):
        output[index, :5] = [320, 300, 100, 100, confidence]
        output[index, 5:8] = [320, 300, confidence]
    scored = []

    def check(keypoints, *_):
        confidence = keypoints[0][2]
        scored.append(confidence)
        return {"is_fall": bool(invalid_highest_is_fall and confidence > 0.85),
                "score": 3.0, "reasons": ["test_pose"], "near_miss": None}

    detections = detections_from_yolo_output(
        output, task="pose", gie_id=1, labels=["person"],
        frame_width=1920, frame_height=1080, confidence_threshold=0.5,
        class_ids_filter={0}, input_size=960, iou_threshold=0.45, max_detections=10,
        fall_checker=check,
        person_pose_validator=lambda points: invalid_highest_is_fall is None or points[0][2] < 0.85,
    )
    assert len(detections) == 1
    assert detections[0]["confidence"] == pytest.approx(0.8 if invalid_highest_is_fall is False else 0.9)
    assert detections[0]["is_fall"] is bool(invalid_highest_is_fall)
    assert detections[0]["fall_reasons"] == ["test_pose"]
    assert len(scored) == (2 if invalid_highest_is_fall is False else 1)
    assert "_fall_result" not in detections[0]


def test_filter_yolo_candidates_vectorizes_pose_confidence_filter():
    rows = np.zeros((8400, 56), dtype=np.float32)
    rows[17, 4] = 0.91
    rows[7000, 4] = 0.49

    candidates, class_ids, confidences = filter_yolo_candidates(
        rows,
        task="pose",
        confidence_threshold=0.5,
        class_ids_filter={0},
    )

    assert candidates.shape == (1, 56)
    assert class_ids.tolist() == [0]
    assert confidences.tolist() == [np.float32(0.91)]


def test_filter_yolo_candidates_vectorizes_detect_class_filter():
    rows = np.array(
        [
            [10, 10, 4, 4, 0.90, 0.10],
            [20, 20, 4, 4, 0.20, 0.80],
            [30, 30, 4, 4, np.nan, 0.95],
        ],
        dtype=np.float32,
    )

    candidates, class_ids, confidences = filter_yolo_candidates(
        rows,
        task="detect",
        confidence_threshold=0.5,
        class_ids_filter={1},
    )

    assert candidates[:, 0].tolist() == [20.0, 30.0]
    assert class_ids.tolist() == [1, 1]
    assert confidences.tolist() == [np.float32(0.8), np.float32(0.95)]


@pytest.mark.parametrize("class_ids_filter", [None, set(), {0}, {1}, {0, 1}])
@pytest.mark.parametrize("order", ["C", "F"])
def test_pose_class_filter_preserves_finite_threshold_and_input(class_ids_filter, order):
    # 실제 텐서의 전치 배치에서도 임계값 경계·손상 점수·원본 보존 계약을 확인한다.
    rows = np.zeros((7, 56), dtype=np.float32, order=order)
    rows[:, 0] = np.arange(7)
    rows[:, 4] = [0.49, 0.5, np.nan, np.inf, -np.inf, 0.9, 0.7]
    original = rows.copy()

    candidates, class_ids, confidences = filter_yolo_candidates(
        rows, task="pose", confidence_threshold=0.5, class_ids_filter=class_ids_filter,
    )

    expected_indices = [] if class_ids_filter == {1} else [1, 5, 6]
    np.testing.assert_array_equal(candidates, rows[expected_indices])
    np.testing.assert_array_equal(class_ids, np.zeros(len(expected_indices), dtype=np.int32))
    np.testing.assert_array_equal(confidences, rows[expected_indices, 4])
    np.testing.assert_array_equal(rows, original)
    assert candidates.dtype == confidences.dtype == np.float32
    assert class_ids.dtype == np.int32


def test_vectorized_candidate_filter_matches_legacy_for_finite_pose_rows():
    rng = np.random.default_rng(20260708)
    rows = rng.random((8400, 56), dtype=np.float32)

    vectorized = filter_yolo_candidates(
        rows,
        task="pose",
        confidence_threshold=0.55,
        class_ids_filter={0},
    )
    legacy = filter_yolo_candidates_legacy(
        rows,
        task="pose",
        confidence_threshold=0.55,
        class_ids_filter={0},
    )

    for vectorized_value, legacy_value in zip(vectorized, legacy):
        np.testing.assert_array_equal(vectorized_value, legacy_value)
