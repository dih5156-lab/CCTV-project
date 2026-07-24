from types import SimpleNamespace

import numpy as np

from scripts.datasets.train_yolo_pose_fall_rf import (
    _dataset_summary,
    _extract_video_features,
    _group_holdout_indices,
    _scene_base,
    _select_rows,
    _summarize_frames,
)


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
