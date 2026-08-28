from types import SimpleNamespace

import numpy as np

from scripts.datasets import train_yolo_pose_fall_rf
from scripts.datasets.train_yolo_pose_fall_rf import (
    _build_model_bundle,
    _dataset_summary,
    _extract_video_features,
    _group_holdout_indices,
    _hard_case_sample_weights,
    _pose_geometry,
    _sample_video_frames,
    _sampling_window_for_row,
    _scene_base,
    _select_rows,
    _select_tracked_pose_index,
    _summarize_frames,
)


def test_hard_case_weights_only_upweight_out_of_fold_errors():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    fall_probabilities = np.asarray([0.2, 0.8, 0.7, 0.3], dtype=np.float32)

    sample_weights = _hard_case_sample_weights(
        labels,
        fall_probabilities,
        hard_case_weight=3.0,
    )

    np.testing.assert_array_equal(sample_weights, np.asarray([1.0, 3.0, 1.0, 3.0]))


def test_model_bundle_records_feature_and_inference_compatibility():
    model = SimpleNamespace(n_features_in_=len(train_yolo_pose_fall_rf.FEATURE_NAMES))
    args = SimpleNamespace(
        max_frames=48,
        frame_stride=3,
        imgsz=640,
        confidence_threshold=0.35,
        candidate_window_frames=181,
        candidate_window_seconds=3.0,
        prediction_batch_size=4,
        min_pose_frames=3,
        decision_threshold=0.7,
    )

    bundle = _build_model_bundle(model, args)

    assert bundle["bundle_schema_version"] == 1
    assert bundle["model_kind"] == "yolo_pose_summary_rf"
    assert bundle["fall_class_label"] == 1
    assert bundle["inference_config"]["max_frames"] == 48
    assert bundle["inference_config"]["imgsz"] == 640
    assert bundle["inference_config"]["candidate_window_frames"] == 181
    assert bundle["inference_config"]["candidate_window_seconds"] == 3.0
    assert len(bundle["feature_names"]) == len(train_yolo_pose_fall_rf.FEATURE_NAMES)


def test_sample_video_frames_decodes_forward_without_random_seeks():
    class FakeCapture:
        def __init__(self):
            self.position = 0
            self.set_calls = 0

        def get(self, _property):
            return 10

        def set(self, _property, _value):
            self.set_calls += 1
            return True

        def grab(self):
            if self.position >= 10:
                return False
            self.position += 1
            return True

        def read(self):
            if self.position >= 10:
                return False, None
            frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, frame

    capture = FakeCapture()

    sampled = _sample_video_frames(capture, max_frames=4, frame_stride=2)

    assert capture.set_calls == 0
    assert [frame_index for frame_index, _ in sampled] == [1, 4, 7, 10]
    assert [int(frame[0, 0, 0]) for _, frame in sampled] == [0, 3, 6, 9]


def test_sample_video_frames_limits_sampling_to_labeled_fall_window():
    class FakeCapture:
        def __init__(self):
            self.position = 0

        def get(self, _property):
            return 10

        def grab(self):
            self.position += 1
            return self.position <= 10

        def read(self):
            if self.position >= 10:
                return False, None
            frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, frame

    sampled = _sample_video_frames(
        FakeCapture(),
        max_frames=3,
        frame_stride=1,
        start_frame=3,
        end_frame=7,
    )

    assert [frame_index for frame_index, _ in sampled] == [3, 5, 7]


def test_sampling_window_uses_margin_for_falls_and_full_video_for_non_falls():
    fall_window = _sampling_window_for_row(
        {
            "is_fall": True,
            "fall_start_frame": 233,
            "fall_end_frame": 293,
            "scene_length": 600,
        },
        margin_frames=120,
    )
    non_fall_window = _sampling_window_for_row(
        {
            "is_fall": False,
            "fall_start_frame": 0,
            "fall_end_frame": 0,
            "scene_length": 600,
        },
        margin_frames=120,
    )

    assert fall_window == (113, 413)
    assert non_fall_window == (None, None)


def test_scene_base_strips_camera_suffix():
    assert _scene_base("00074_H_A_BY_C6") == "00074_H_A_BY"
    assert _scene_base("field_camera_1_20260713") == "field_camera_1_20260713"


def test_group_holdout_keeps_camera_variants_together():
    scene_ids = [
        "fall_a_C1",
        "fall_a_C2",
        "fall_b_C1",
        "fall_b_C2",
        "normal_a_C1",
        "normal_a_C2",
        "normal_b_C1",
        "normal_b_C2",
    ]
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    train_indices, holdout_indices, split_info = _group_holdout_indices(
        scene_ids,
        labels,
        test_size=0.25,
        random_state=42,
    )

    train_groups = {_scene_base(scene_ids[index]) for index in train_indices}
    holdout_groups = {_scene_base(scene_ids[index]) for index in holdout_indices}
    assert train_groups.isdisjoint(holdout_groups)
    assert split_info["method"] == "group_shuffle"
    assert split_info["group_by"] == "scene_base"
    assert split_info["group_overlap"] == []
    assert set(labels[train_indices]) == {0, 1}
    assert set(labels[holdout_indices]) == {0, 1}


def test_group_holdout_keeps_reviewed_hard_case_groups_in_training():
    scene_ids = [
        "fall_a_C1",
        "fall_a_C2",
        "fall_b_C1",
        "fall_c_C1",
        "normal_a_C1",
        "normal_b_C1",
        "normal_c_C1",
    ]
    labels = np.asarray([1, 1, 1, 1, 0, 0, 0], dtype=np.int64)

    train_indices, holdout_indices, split_info = _group_holdout_indices(
        scene_ids,
        labels,
        test_size=0.25,
        random_state=42,
        forced_train_scene_ids={"fall_a_C1"},
    )

    train_groups = {_scene_base(scene_ids[index]) for index in train_indices}
    holdout_groups = {_scene_base(scene_ids[index]) for index in holdout_indices}
    assert "fall_a" in train_groups
    assert "fall_a" not in holdout_groups
    assert split_info["forced_train_groups"] == ["fall_a"]


def test_reviewed_hard_case_weights_apply_only_to_selected_training_rows():
    scene_ids = ["fall_a_C1", "fall_b_C1", "normal_a_C1"]
    base_weights = np.asarray([1.0, 2.0, 1.0])

    weighted = train_yolo_pose_fall_rf._apply_reviewed_hard_case_weights(
        base_weights,
        scene_ids,
        {"fall_a_C1": 3.0, "normal_a_C1": 4.0},
    )

    np.testing.assert_array_equal(weighted, np.asarray([3.0, 2.0, 4.0]))


def test_dataset_summary_reports_scene_group_counts():
    summary = _dataset_summary(
        ["fall_a_C1", "fall_a_C2", "normal_a_C1", "normal_b_C1"],
        np.asarray([1, 1, 0, 0], dtype=np.int64),
    )

    assert summary["groups"] == 3
    assert summary["group_class_counts"] == {"fall": 1, "non_fall": 2}


def test_select_rows_round_robins_across_scene_groups():
    rows = []
    for is_fall, prefix in ((False, "N"), (True, "F")):
        for group in ("A", "D", "E"):
            for camera in range(1, 4):
                rows.append(
                    {
                        "is_fall": is_fall,
                        "scene_id": f"{prefix}_{group}_C{camera}",
                    }
                )

    selected = _select_rows(rows, 6)
    selected_groups = {_scene_base(str(row["scene_id"])) for row in selected}

    assert selected_groups == {"N_A", "N_D", "N_E", "F_A", "F_D", "F_E"}


def test_select_rows_round_robins_across_scene_environments():
    rows = []
    for is_fall, label in ((False, "N"), (True, "F")):
        for location, position in (
            ("병원", "병실"),
            ("병원", "복도"),
            ("집", "거실"),
            ("요양병원", "화장실"),
        ):
            rows.append(
                {
                    "is_fall": is_fall,
                    "scene_id": f"{label}_{location}_{position}_C1",
                    "scene_location": location,
                    "scene_position": position,
                }
            )

    selected = _select_rows(rows, 4)
    selected_environments = {
        (str(row["scene_location"]), str(row["scene_position"]))
        for row in selected
    }

    assert selected_environments == {
        ("병원", "병실"),
        ("병원", "복도"),
        ("집", "거실"),
        ("요양병원", "화장실"),
    }


def test_summarize_frames_reports_temporal_fall_transition():
    frame_records = [
        {
            "fall_score": score,
            "bbox_aspect": aspect,
            "bbox_area_ratio": area,
            "visible_keypoints": 17,
            "mean_keypoint_confidence": 0.9,
            "detection_confidence": 0.95,
            "fall_reasons": [],
        }
        for score, aspect, area in (
            (0.5, 0.4, 0.1),
            (1.0, 0.5, 0.11),
            (3.5, 1.0, 0.15),
            (4.0, 1.2, 0.18),
        )
    ]

    summary = _summarize_frames(frame_records, frames_seen=4)
    values = dict(zip(summary["feature_names"], summary["feature_vector"]))

    assert values["fall_score_slope"] > 0
    assert values["fall_score_end_minus_start"] > 0
    assert values["max_fall_score_rise"] == 2.5
    assert values["late_score_ge_3_ratio"] == 1.0
    assert values["bbox_aspect_end_minus_start"] > 0


def test_pose_geometry_distinguishes_upright_and_horizontal_torso():
    upright = np.zeros((17, 3), dtype=np.float32)
    horizontal = np.zeros((17, 3), dtype=np.float32)
    for index in range(17):
        upright[index] = [50 + index % 2 * 10, 20 + index * 8, 0.95]
        horizontal[index] = [20 + index * 8, 100 + index % 2 * 10, 0.95]

    upright[5:7, :2] = [[40, 40], [60, 40]]
    upright[11:13, :2] = [[42, 100], [58, 100]]
    horizontal[5:7, :2] = [[30, 90], [30, 110]]
    horizontal[11:13, :2] = [[80, 90], [80, 110]]

    upright_geometry = _pose_geometry(
        upright,
        bbox=np.asarray([20, 10, 90, 180], dtype=np.float32),
        frame_width=200,
        frame_height=200,
        min_keypoint_confidence=0.3,
    )
    horizontal_geometry = _pose_geometry(
        horizontal,
        bbox=np.asarray([10, 70, 170, 130], dtype=np.float32),
        frame_width=200,
        frame_height=200,
        min_keypoint_confidence=0.3,
    )

    assert upright_geometry["torso_angle_from_vertical"] < 0.1
    assert horizontal_geometry["torso_angle_from_vertical"] > 0.9
    assert (
        horizontal_geometry["pose_width_height_ratio"]
        > upright_geometry["pose_width_height_ratio"]
    )


def test_summarize_frames_reports_pose_center_descent():
    frame_records = []
    for index, center_y in enumerate((0.3, 0.4, 0.65, 0.75)):
        frame_records.append(
            {
                "fall_score": float(index),
                "bbox_aspect": 0.5 + index * 0.1,
                "bbox_area_ratio": 0.1,
                "visible_keypoints": 17,
                "mean_keypoint_confidence": 0.9,
                "detection_confidence": 0.95,
                "fall_reasons": [],
                "pose_width_height_ratio": 0.4 + index * 0.2,
                "torso_angle_from_vertical": index / 3,
                "torso_length_bbox_ratio": 0.3,
                "hip_center_y_frame_ratio": center_y,
                "body_center_y_frame_ratio": center_y - 0.05,
                "bbox_center_y_frame_ratio": center_y,
            }
        )

    summary = _summarize_frames(frame_records, frames_seen=4)
    values = dict(zip(summary["feature_names"], summary["feature_vector"]))

    assert values["hip_center_y_end_minus_start"] > 0
    assert values["body_center_y_end_minus_start"] > 0
    assert values["max_hip_center_y_rise"] > 0
    assert values["torso_angle_end_minus_start"] > 0


def test_select_tracked_pose_uses_highest_confidence_without_history():
    selected_index = _select_tracked_pose_index(
        confidences=np.asarray([0.65, 0.9]),
        centers=np.asarray([[20.0, 20.0], [80.0, 80.0]]),
        previous_center=None,
        frame_diagonal=100.0,
    )

    assert selected_index == 1


def test_select_tracked_pose_prefers_nearby_person_when_confidence_is_close():
    selected_index = _select_tracked_pose_index(
        confidences=np.asarray([0.8, 0.9]),
        centers=np.asarray([[22.0, 20.0], [90.0, 90.0]]),
        previous_center=(20.0, 20.0),
        frame_diagonal=100.0,
    )

    assert selected_index == 0


def test_predict_pose_results_chunks_frames_for_static_engine_batch():
    class FakeModel:
        def __init__(self):
            self.batch_sizes = []

        def predict(self, frames, **_kwargs):
            self.batch_sizes.append(len(frames))
            return [f"result-{len(self.batch_sizes)}" for _ in frames]

    model = FakeModel()
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

    results = train_yolo_pose_fall_rf._predict_pose_results(
        model=model,
        frames=frames,
        imgsz=320,
        confidence_threshold=0.35,
        prediction_batch_size=1,
    )

    assert model.batch_sizes == [1, 1, 1]
    assert results == ["result-1", "result-2", "result-3"]


def test_extract_video_features_batches_sampled_frames(monkeypatch, tmp_path):
    class FakeCapture:
        def __init__(self, _path):
            self.position = 0

        def isOpened(self):
            return True

        def get(self, _property):
            return 2

        def set(self, _property, value):
            self.position = int(value)

        def read(self):
            return True, np.zeros((32, 32, 3), dtype=np.uint8)

        def release(self):
            return None

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def __len__(self):
            return len(self.value)

        def __getitem__(self, index):
            return FakeTensor(self.value[index])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeBoxes(SimpleNamespace):
        def __len__(self):
            return len(self.conf)

    result = SimpleNamespace(
        boxes=FakeBoxes(
            conf=FakeTensor([0.9]),
            xyxy=FakeTensor([[4.0, 4.0, 20.0, 28.0]]),
        ),
        keypoints=SimpleNamespace(
            xy=FakeTensor(np.ones((1, 17, 2), dtype=np.float32) * 10),
            conf=FakeTensor(np.ones((1, 17), dtype=np.float32) * 0.9),
        ),
    )

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def predict(self, frames, **_kwargs):
            self.calls += 1
            batch_size = len(frames) if isinstance(frames, list) else 1
            return [result for _ in range(batch_size)]

    detector = SimpleNamespace(
        min_keypoint_confidence=0.3,
        score_threshold=3.0,
        _score_fall=lambda *_args: SimpleNamespace(score=1.0, reasons=[]),
    )
    model = FakeModel()
    monkeypatch.setattr(
        "scripts.datasets.train_yolo_pose_fall_rf.cv2.VideoCapture",
        FakeCapture,
    )

    summary = _extract_video_features(
        model=model,
        detector=detector,
        video_path=tmp_path / "sample.mp4",
        max_frames=2,
        frame_stride=1,
        imgsz=320,
        confidence_threshold=0.35,
    )

    assert model.calls == 1
    assert summary["frames_seen"] == 2
    assert summary["frames_with_pose"] == 2
