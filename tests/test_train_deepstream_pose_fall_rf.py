import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.datasets import train_deepstream_pose_fall_rf as trainer


def _capture_record(
    *,
    label: int,
    group_id: str,
    scene_id: str,
    feature_names: list[str] | None = None,
    feature_vector: list[float] | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "runtime": "deepstream_pose_inline",
        "camera_id": "sample_eval",
        "label": label,
        "is_fall": label == 1,
        "group_id": group_id,
        "scene_id": scene_id,
        "video_path": f"/dataset/{scene_id}.mp4",
        "feature_names": feature_names or ["a", "b"],
        "feature_vector": feature_vector or [0.1, 0.2],
    }


def _write_records(path, records: list[dict]):
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return path


def test_load_capture_datasets_combines_files_with_stable_features(
    tmp_path,
) -> None:
    normal_path = _write_records(
        tmp_path / "normal.jsonl",
        [
            _capture_record(
                label=0,
                group_id="normal-group",
                scene_id="normal-1",
                feature_vector=[0.1, 0.2],
            ),
            _capture_record(
                label=0,
                group_id="normal-group",
                scene_id="normal-2",
                feature_vector=[0.2, 0.3],
            ),
        ],
    )
    fall_path = _write_records(
        tmp_path / "fall.jsonl",
        [
            _capture_record(
                label=1,
                group_id="fall-group",
                scene_id="fall-1",
                feature_vector=[0.8, 0.9],
            ),
            _capture_record(
                label=1,
                group_id="fall-group",
                scene_id="fall-2",
                feature_vector=[0.9, 1.0],
            ),
        ],
    )

    dataset = trainer.load_capture_datasets([normal_path, fall_path])

    assert dataset.feature_names == ["a", "b"]
    assert dataset.x.shape == (4, 2)
    assert dataset.x.dtype == np.float32
    assert dataset.y.tolist() == [0, 0, 1, 1]
    assert set(dataset.groups.tolist()) == {
        "normal-group",
        "fall-group",
    }
    assert dataset.scene_ids == (
        "normal-1",
        "normal-2",
        "fall-1",
        "fall-2",
    )


@pytest.mark.parametrize(
    "mutate, expected_error",
    [
        (lambda record: record.update(schema_version=2), "schema_version"),
        (lambda record: record.update(runtime="offline"), "runtime"),
        (
            lambda record: record.update(
                feature_names=["a", "b"],
                feature_vector=[1.0],
            ),
            "feature length",
        ),
        (
            lambda record: record.update(feature_vector=[float("nan"), 1.0]),
            "finite",
        ),
        (lambda record: record.pop("group_id"), "group_id"),
        (lambda record: record.pop("scene_id"), "scene_id"),
    ],
)
def test_load_capture_datasets_rejects_invalid_records(
    tmp_path,
    mutate,
    expected_error,
) -> None:
    invalid_record = _capture_record(
        label=0,
        group_id="normal-group",
        scene_id="normal-1",
    )
    mutate(invalid_record)
    path = _write_records(
        tmp_path / "invalid.jsonl",
        [
            invalid_record,
            _capture_record(
                label=1,
                group_id="fall-group",
                scene_id="fall-1",
            ),
        ],
    )

    with pytest.raises(ValueError, match=expected_error):
        trainer.load_capture_datasets([path])


def test_load_capture_datasets_rejects_feature_order_mismatch(
    tmp_path,
) -> None:
    path = _write_records(
        tmp_path / "mismatch.jsonl",
        [
            _capture_record(
                label=0,
                group_id="normal-group",
                scene_id="normal-1",
            ),
            _capture_record(
                label=1,
                group_id="fall-group",
                scene_id="fall-1",
                feature_names=["b", "a"],
            ),
        ],
    )

    with pytest.raises(ValueError, match="feature_names"):
        trainer.load_capture_datasets([path])


def test_load_capture_datasets_requires_both_classes(tmp_path) -> None:
    path = _write_records(
        tmp_path / "normal-only.jsonl",
        [
            _capture_record(
                label=0,
                group_id="normal-group",
                scene_id="normal-1",
            )
        ],
    )

    with pytest.raises(ValueError, match="both fall and non-fall"):
        trainer.load_capture_datasets([path])


def test_load_capture_datasets_rejects_mixed_labels_in_group(
    tmp_path,
) -> None:
    path = _write_records(
        tmp_path / "mixed-group.jsonl",
        [
            _capture_record(
                label=0,
                group_id="shared-group",
                scene_id="normal-1",
            ),
            _capture_record(
                label=1,
                group_id="shared-group",
                scene_id="fall-1",
            ),
        ],
    )

    with pytest.raises(ValueError, match="mixed labels"):
        trainer.load_capture_datasets([path])


def test_assert_validation_disjoint_rejects_group_overlap() -> None:
    with pytest.raises(ValueError, match="group overlap"):
        trainer.assert_validation_disjoint(
            training_groups={"subject-001"},
            training_scene_ids={"scene-001"},
            validation_rows=[
                {
                    "scene_group": "subject-001",
                    "scene_id": "scene-900",
                }
            ],
        )


def test_assert_validation_disjoint_rejects_scene_overlap() -> None:
    with pytest.raises(ValueError, match="scene overlap"):
        trainer.assert_validation_disjoint(
            training_groups={"subject-001"},
            training_scene_ids={"scene-001"},
            validation_rows=[
                {
                    "scene_group": "subject-900",
                    "scene_id": "scene-001",
                }
            ],
        )


def test_assert_validation_disjoint_accepts_separate_groups_and_scenes() -> None:
    trainer.assert_validation_disjoint(
        training_groups={"subject-001"},
        training_scene_ids={"scene-001"},
        validation_rows=[
            {
                "scene_group": "subject-900",
                "scene_id": "scene-900",
            }
        ],
    )


def _balanced_grouped_dataset() -> trainer.CaptureDataset:
    feature_rows = []
    labels = []
    groups = []
    scene_ids = []
    for label, base_value in ((0, 0.1), (1, 0.8)):
        for index in range(4):
            feature_rows.append(
                [base_value + index * 0.01, base_value + index * 0.02]
            )
            labels.append(label)
            groups.append(f"class-{label}-group-{index}")
            scene_ids.append(f"class-{label}-scene-{index}")
    return trainer.CaptureDataset(
        x=np.asarray(feature_rows, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=object),
        scene_ids=tuple(scene_ids),
        feature_names=["a", "b"],
        source_paths=(Path("dataset.jsonl"),),
    )


def test_train_candidate_uses_group_split_and_runtime_bundle_contract() -> None:
    dataset = _balanced_grouped_dataset()

    bundle, metrics = trainer.train_candidate(
        dataset,
        random_state=42,
        validation_fraction=0.25,
    )

    assert bundle["bundle_schema_version"] == 1
    assert bundle["model_kind"] == "deepstream_pose_inline_rf"
    assert bundle["feature_source"] == "deepstream_pose_inline"
    assert bundle["feature_names"] == ["a", "b"]
    assert bundle["fall_class_label"] == 1
    assert bundle["training_config"]["decision_threshold"] == 0.7
    assert bundle["inference_config"]["max_frames"] == 48
    assert bundle["inference_config"]["candidate_window_seconds"] == 3.0
    assert set(metrics["train_groups"]).isdisjoint(
        metrics["holdout_groups"]
    )
    assert metrics["holdout"]["threshold"] == 0.7
    assert "fall_recall" in metrics["holdout"]
    assert "false_positive_rate" in metrics["holdout"]


def test_main_trains_candidate_and_writes_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    dataset_path = _write_records(
        tmp_path / "dataset.jsonl",
        [
            _capture_record(
                label=label,
                group_id=f"class-{label}-group-{index}",
                scene_id=f"class-{label}-scene-{index}",
                feature_vector=[
                    0.1 + label * 0.7 + index * 0.01,
                    0.2 + label * 0.7 + index * 0.01,
                ],
            )
            for label in (0, 1)
            for index in range(4)
        ],
    )
    validation_manifest_path = _write_records(
        tmp_path / "validation-manifest.jsonl",
        [
            {
                "scene_group": "validation-group",
                "scene_id": "validation-scene",
            }
        ],
    )
    output_model = tmp_path / "candidate.joblib"
    output_metrics = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_deepstream_pose_fall_rf.py",
            "--dataset",
            str(dataset_path),
            "--validation-manifest",
            str(validation_manifest_path),
            "--output-model",
            str(output_model),
            "--output-metrics",
            str(output_metrics),
        ],
    )

    assert trainer.main() == 0

    import joblib

    bundle = joblib.load(output_model)
    metrics = json.loads(output_metrics.read_text(encoding="utf-8"))
    assert bundle["feature_source"] == "deepstream_pose_inline"
    assert bundle["training_config"]["decision_threshold"] == 0.7
    assert metrics["holdout"]["threshold"] == 0.7


def test_main_refuses_to_overwrite_candidate_without_flag(
    tmp_path,
    monkeypatch,
) -> None:
    output_model = tmp_path / "candidate.joblib"
    output_model.write_bytes(b"existing")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_deepstream_pose_fall_rf.py",
            "--dataset",
            str(tmp_path / "dataset.jsonl"),
            "--validation-manifest",
            str(tmp_path / "validation.jsonl"),
            "--output-model",
            str(output_model),
            "--output-metrics",
            str(tmp_path / "metrics.json"),
        ],
    )

    with pytest.raises(SystemExit, match="already exists"):
        trainer.main()
