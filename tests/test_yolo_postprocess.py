"""YOLO tensor 후처리 유틸리티 테스트."""

from __future__ import annotations

import numpy as np

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
        fall_checker=lambda keypoints, width, height: False,
        person_pose_validator=lambda keypoints: True,
    )

    assert len(detections) == 1
    assert detections[0]["label"] == "helmet"
    assert detections[0]["class_id"] == 1
    assert detections[0]["box"] == (270, 190, 100, 100)


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
