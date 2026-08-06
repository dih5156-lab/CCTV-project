#!/usr/bin/env python3
"""Train a fall RF candidate from DeepStream inline pose feature captures."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class CaptureDataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    scene_ids: tuple[str, ...]
    feature_names: list[str]
    source_paths: tuple[Path, ...]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"capture dataset does not exist: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: record must be an object"
                )
            records.append(record)
    if not records:
        raise ValueError(f"capture dataset is empty: {path}")
    return records


def load_capture_datasets(paths: Sequence[Path]) -> CaptureDataset:
    source_paths = tuple(Path(path) for path in paths)
    if not source_paths:
        raise ValueError("at least one capture dataset is required")

    feature_schema: list[str] | None = None
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    scene_ids: list[str] = []
    group_labels: dict[str, int] = {}

    for path in source_paths:
        for record_index, record in enumerate(_read_jsonl(path)):
            record_location = f"{path}:record {record_index}"
            schema_version = int(record.get("schema_version") or 0)
            if schema_version not in {1, 2}:
                raise ValueError(
                    f"{record_location}: unsupported schema_version"
                )
            if schema_version == 2:
                frame_records = record.get("frame_records")
                frame_feature_names = record.get("frame_feature_names")
                if not isinstance(frame_records, list) or not frame_records:
                    raise ValueError(
                        f"{record_location}: schema_version 2 requires frame_records"
                    )
                if not isinstance(frame_feature_names, list) or not frame_feature_names:
                    raise ValueError(
                        f"{record_location}: schema_version 2 requires frame_feature_names"
                    )
            if record.get("runtime") != "deepstream_pose_inline":
                raise ValueError(f"{record_location}: unexpected runtime")

            feature_names = record.get("feature_names")
            feature_vector = record.get("feature_vector")
            if not isinstance(feature_names, list) or not feature_names:
                raise ValueError(
                    f"{record_location}: feature_names must be a non-empty list"
                )
            if not isinstance(feature_vector, list):
                raise ValueError(
                    f"{record_location}: feature_vector must be a list"
                )
            if len(feature_names) != len(feature_vector):
                raise ValueError(f"{record_location}: feature length mismatch")
            if feature_schema is None:
                feature_schema = [str(name) for name in feature_names]
            elif feature_names != feature_schema:
                raise ValueError(f"{record_location}: feature_names mismatch")

            try:
                numeric_vector = [float(value) for value in feature_vector]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{record_location}: features must be numeric"
                ) from exc
            if not np.isfinite(np.asarray(numeric_vector)).all():
                raise ValueError(f"{record_location}: features must be finite")

            label = record.get("label")
            if label not in {0, 1}:
                raise ValueError(f"{record_location}: label must be 0 or 1")
            group_id = str(record.get("group_id") or "").strip()
            scene_id = str(record.get("scene_id") or "").strip()
            if not group_id:
                raise ValueError(f"{record_location}: missing group_id")
            if not scene_id:
                raise ValueError(f"{record_location}: missing scene_id")
            previous_label = group_labels.setdefault(group_id, int(label))
            if previous_label != int(label):
                raise ValueError(
                    f"{record_location}: group {group_id} has mixed labels"
                )

            features.append(numeric_vector)
            labels.append(int(label))
            groups.append(group_id)
            scene_ids.append(scene_id)

    if set(labels) != {0, 1}:
        raise ValueError("capture datasets must contain both fall and non-fall")
    if feature_schema is None:
        raise ValueError("capture datasets contain no feature schema")

    return CaptureDataset(
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=object),
        scene_ids=tuple(scene_ids),
        feature_names=feature_schema,
        source_paths=source_paths,
    )


def assert_validation_disjoint(
    *,
    training_groups: set[str],
    training_scene_ids: set[str],
    validation_rows: Sequence[dict],
) -> None:
    validation_groups: set[str] = set()
    validation_scene_ids: set[str] = set()
    for row_index, row in enumerate(validation_rows):
        group_id = str(
            row.get("group_id") or row.get("scene_group") or ""
        ).strip()
        scene_id = str(row.get("scene_id") or "").strip()
        if not group_id:
            raise ValueError(
                f"validation row {row_index}: missing group_id or scene_group"
            )
        if not scene_id:
            raise ValueError(
                f"validation row {row_index}: missing scene_id"
            )
        validation_groups.add(group_id)
        validation_scene_ids.add(scene_id)

    group_overlap = sorted(training_groups & validation_groups)
    if group_overlap:
        raise ValueError(f"training/validation group overlap: {group_overlap}")
    scene_overlap = sorted(training_scene_ids & validation_scene_ids)
    if scene_overlap:
        raise ValueError(f"training/validation scene overlap: {scene_overlap}")


def _evaluate_holdout(
    model: Any,
    *,
    x: np.ndarray,
    y: np.ndarray,
    scene_ids: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    probabilities = model.predict_proba(x)
    fall_class_index = list(model.classes_).index(1)
    fall_probabilities = probabilities[:, fall_class_index]
    predictions = (fall_probabilities >= threshold).astype(np.int64)

    true_positive = int(((y == 1) & (predictions == 1)).sum())
    false_negative = int(((y == 1) & (predictions == 0)).sum())
    false_positive = int(((y == 0) & (predictions == 1)).sum())
    true_negative = int(((y == 0) & (predictions == 0)).sum())
    false_positives = []
    false_negatives = []
    for index, (truth, prediction) in enumerate(zip(y, predictions)):
        error_record = {
            "scene_id": str(scene_ids[index]),
            "true": int(truth),
            "predicted": int(prediction),
            "fall_probability": float(fall_probabilities[index]),
        }
        if truth == 0 and prediction == 1:
            false_positives.append(error_record)
        elif truth == 1 and prediction == 0:
            false_negatives.append(error_record)

    return {
        "threshold": threshold,
        "confusion_matrix_labels": ["fall", "non_fall"],
        "confusion_matrix": [
            [true_positive, false_negative],
            [false_positive, true_negative],
        ],
        "fall_precision": true_positive
        / max(true_positive + false_positive, 1),
        "fall_recall": true_positive
        / max(true_positive + false_negative, 1),
        "false_positive_rate": false_positive
        / max(false_positive + true_negative, 1),
        "roc_auc": float(roc_auc_score(y, fall_probabilities)),
        "errors": {
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
    }


def train_candidate(
    dataset: CaptureDataset,
    *,
    random_state: int,
    validation_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(set(dataset.groups.tolist())) < 4:
        raise ValueError("group holdout requires at least four groups")

    selected_split: tuple[np.ndarray, np.ndarray] | None = None
    splitter = GroupShuffleSplit(
        n_splits=64,
        test_size=validation_fraction,
        random_state=random_state,
    )
    for train_indices, holdout_indices in splitter.split(
        dataset.x,
        dataset.y,
        groups=dataset.groups,
    ):
        if (
            set(dataset.y[train_indices].tolist()) == {0, 1}
            and set(dataset.y[holdout_indices].tolist()) == {0, 1}
        ):
            selected_split = (train_indices, holdout_indices)
            break
    if selected_split is None:
        raise ValueError(
            "unable to create group holdout containing both classes"
        )

    train_indices, holdout_indices = selected_split
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(dataset.x[train_indices], dataset.y[train_indices])

    train_groups = sorted(set(dataset.groups[train_indices].tolist()))
    holdout_groups = sorted(set(dataset.groups[holdout_indices].tolist()))
    holdout_scene_ids = [
        dataset.scene_ids[index] for index in holdout_indices
    ]
    threshold = 0.7
    holdout_metrics = _evaluate_holdout(
        model,
        x=dataset.x[holdout_indices],
        y=dataset.y[holdout_indices],
        scene_ids=holdout_scene_ids,
        threshold=threshold,
    )
    dataset_summary = {
        "rows": int(dataset.x.shape[0]),
        "groups": len(set(dataset.groups.tolist())),
        "class_counts": {
            "non_fall": int((dataset.y == 0).sum()),
            "fall": int((dataset.y == 1).sum()),
        },
        "source_paths": [str(path) for path in dataset.source_paths],
    }
    bundle = {
        "bundle_schema_version": 1,
        "model_kind": "deepstream_pose_inline_rf",
        "feature_schema_version": 1,
        "feature_source": "deepstream_pose_inline",
        "feature_names": list(dataset.feature_names),
        "fall_class_label": 1,
        "model": model,
        "inference_config": {
            "max_frames": 48,
            "candidate_window_seconds": 3.0,
        },
        "training_config": {
            "min_pose_frames": 1,
            "decision_threshold": threshold,
            "random_state": random_state,
            "validation_fraction": validation_fraction,
            "class_weight": "balanced",
        },
        "dataset_summary": dataset_summary,
    }
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_source": "deepstream_pose_inline",
        "feature_names": list(dataset.feature_names),
        "dataset_summary": dataset_summary,
        "train_groups": train_groups,
        "holdout_groups": holdout_groups,
        "holdout": holdout_metrics,
    }
    return bundle, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    import joblib

    args = parse_args()
    if not args.overwrite:
        existing_outputs = [
            path
            for path in (args.output_model, args.output_metrics)
            if path.exists()
        ]
        if existing_outputs:
            raise SystemExit(
                "output already exists: "
                + ", ".join(str(path) for path in existing_outputs)
            )

    dataset = load_capture_datasets(args.dataset)
    validation_rows = _read_jsonl(args.validation_manifest)
    assert_validation_disjoint(
        training_groups=set(dataset.groups.tolist()),
        training_scene_ids=set(dataset.scene_ids),
        validation_rows=validation_rows,
    )
    bundle, metrics = train_candidate(
        dataset,
        random_state=args.random_state,
        validation_fraction=args.validation_fraction,
    )
    metrics.update(
        {
            "datasets": [str(path) for path in args.dataset],
            "validation_manifest": str(args.validation_manifest),
            "output_model": str(args.output_model),
        }
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_model)
    args.output_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"candidate model: {args.output_model}")
    print(f"metrics: {args.output_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
