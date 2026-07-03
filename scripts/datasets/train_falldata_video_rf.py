#!/usr/bin/env python3
"""Train a small falldata-compatible video RF model from the sample manifest.

The generated model has the same input contract as the public falldata RF
models: one flattened 600 x 1662 MediaPipe Holistic feature sequence.

Labels follow the existing runtime convention:
- 0 = fall
- 1 = non-fall
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/fall_eval/sample_manifest.jsonl"
DEFAULT_FEATURE_CACHE = PROJECT_ROOT / "data/fall_eval/falldata_feature_cache"
DEFAULT_OUTPUT_MODEL = PROJECT_ROOT / "models/experiments/falldata_sample_rf.pkl"
DEFAULT_METRICS = PROJECT_ROOT / "models/experiments/falldata_sample_rf_metrics.json"
DEFAULT_MEDIAPIPE_PYTHON = PROJECT_ROOT / ".venv-mediapipe/bin/python"
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts/datasets/extract_falldata_mediapipe_features.py"
FRAME_FEATURES = 1662
TARGET_FRAMES = 600
FALL_LABEL = 0
NON_FALL_LABEL = 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_id(row: dict[str, Any]) -> str:
    value = str(row.get("scene_id") or Path(str(row["video_path"])).stem)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _sequence_dir(cache_dir: Path, row: dict[str, Any], max_frames: int) -> Path:
    return cache_dir / f"{_safe_id(row)}_max{max_frames}"


def _is_sequence_ready(sequence_dir: Path) -> bool:
    frame_files = list(sequence_dir.glob("*.npy"))
    return len(frame_files) == TARGET_FRAMES


def _extract_features(
    *,
    mediapipe_python: Path,
    video_path: Path,
    output_dir: Path,
    max_frames: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(mediapipe_python),
        str(EXTRACT_SCRIPT),
        "--video",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--max-frames",
        str(max_frames),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _load_sequence(sequence_dir: Path) -> np.ndarray:
    frame_files = sorted(
        sequence_dir.glob("*.npy"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if len(frame_files) != TARGET_FRAMES:
        raise ValueError(f"expected {TARGET_FRAMES} frames, found {len(frame_files)}: {sequence_dir}")

    frames = [np.load(path).reshape(-1) for path in frame_files]
    sequence = np.asarray(frames, dtype=np.float32)
    if sequence.shape != (TARGET_FRAMES, FRAME_FEATURES):
        raise ValueError(f"unexpected sequence shape {sequence.shape}: {sequence_dir}")
    return sequence.reshape(-1)


def _label_for_row(row: dict[str, Any]) -> int:
    return FALL_LABEL if bool(row.get("is_fall")) else NON_FALL_LABEL


def _class_counts(values: list[int] | np.ndarray) -> dict[str, int]:
    array = np.asarray(values, dtype=np.int64)
    return {str(label): int((array == label).sum()) for label in sorted(set(array.tolist()))}


def _group_for_row(row: dict[str, Any], group_by: str) -> str:
    if group_by == "scene_base":
        return _scene_base_for_row(row)
    value = row.get(group_by)
    if value is None:
        value = row.get("scene_id") or Path(str(row["video_path"])).stem
    return str(value)


def _scene_base_for_row(row: dict[str, Any]) -> str:
    scene_id = str(row.get("scene_id") or Path(str(row["video_path"])).stem)
    parts = scene_id.rsplit("_C", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return scene_id


def _select_rows(rows: list[dict[str, Any]], max_videos: int) -> list[dict[str, Any]]:
    if max_videos <= 0 or len(rows) <= max_videos:
        return rows
    fall_rows = [row for row in rows if bool(row.get("is_fall"))]
    non_fall_rows = [row for row in rows if not bool(row.get("is_fall"))]
    selected: list[dict[str, Any]] = []
    for group in (non_fall_rows, fall_rows):
        remaining = max_videos - len(selected)
        if remaining <= 0:
            break
        selected.extend(group[:remaining])
    return selected[:max_videos]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--mediapipe-python", type=Path, default=DEFAULT_MEDIAPIPE_PYTHON)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--dataset-version",
        default="sample_manifest",
        help="Human-readable dataset/version label stored in metrics JSON.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--cv-group-by", default="scene_base")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=1,
        help="Parallel MediaPipe extraction subprocesses (default: 1).",
    )
    return parser.parse_args()


def _build_model(args: argparse.Namespace) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )


def _evaluate_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    *,
    probabilities: list[list[float]] | None,
    scene_ids: list[str],
) -> dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix

    target_names = ["fall", "non_fall"]
    matrix = confusion_matrix(y_true, predictions, labels=[FALL_LABEL, NON_FALL_LABEL])
    return {
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[FALL_LABEL, NON_FALL_LABEL],
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix_labels": target_names,
        "confusion_matrix": matrix.tolist(),
        "predictions": [
            {
                "scene_id": scene_id,
                "true": int(true),
                "predicted": int(predicted),
                "probability": probabilities[idx] if probabilities is not None else None,
            }
            for idx, (scene_id, true, predicted) in enumerate(
                zip(scene_ids, y_true, predictions)
            )
        ],
    }


def _prediction_error_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    false_positives = [
        row
        for row in evaluation.get("predictions", [])
        if row.get("true") == NON_FALL_LABEL and row.get("predicted") == FALL_LABEL
    ]
    false_negatives = [
        row
        for row in evaluation.get("predictions", [])
        if row.get("true") == FALL_LABEL and row.get("predicted") == NON_FALL_LABEL
    ]
    return {
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _dataset_summary(rows: list[dict[str, Any]], groups: list[str]) -> dict[str, Any]:
    labels = [_label_for_row(row) for row in rows]
    cameras = sorted({str(row.get("camera")) for row in rows if row.get("camera") is not None})
    group_class_counts: dict[str, int] = {}
    for label, group in zip(labels, groups):
        class_name = "fall" if label == FALL_LABEL else "non_fall"
        group_class_counts.setdefault(class_name, 0)
    for class_name in list(group_class_counts):
        label_value = FALL_LABEL if class_name == "fall" else NON_FALL_LABEL
        group_class_counts[class_name] = len(
            {
                group
                for label, group in zip(labels, groups)
                if label == label_value
            }
        )
    return {
        "class_counts": _class_counts(labels),
        "groups": len(set(groups)),
        "group_class_counts": group_class_counts,
        "cameras": cameras,
        "scene_ids": [_safe_id(row) for row in rows],
    }


def _cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    row_ids: list[str],
    groups: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return {"enabled": False, "reason": "not enough groups"}

    splitter = GroupKFold(n_splits=len(unique_groups))
    all_true: list[int] = []
    all_pred: list[int] = []
    all_ids: list[str] = []
    all_probabilities: list[list[float]] = []
    folds: list[dict[str, Any]] = []
    group_array = np.asarray(groups)
    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(x, y, groups=group_array),
        start=1,
    ):
        model = _build_model(args)
        model.fit(x[train_idx], y[train_idx])
        fold_pred = model.predict(x[test_idx])
        fold_prob = (
            model.predict_proba(x[test_idx]).tolist()
            if hasattr(model, "predict_proba")
            else None
        )
        fold_ids = [row_ids[index] for index in test_idx]
        fold_true = y[test_idx]
        folds.append(
            {
                "fold": fold_index,
                "test_groups": sorted(set(group_array[test_idx].tolist())),
                **_evaluate_predictions(
                    fold_true,
                    fold_pred,
                    probabilities=fold_prob,
                    scene_ids=fold_ids,
                ),
            }
        )
        all_true.extend(int(value) for value in fold_true)
        all_pred.extend(int(value) for value in fold_pred)
        all_ids.extend(fold_ids)
        if fold_prob is not None:
            all_probabilities.extend(fold_prob)

    aggregate = _evaluate_predictions(
        np.asarray(all_true, dtype=np.int64),
        np.asarray(all_pred, dtype=np.int64),
        probabilities=all_probabilities if all_probabilities else None,
        scene_ids=all_ids,
    )
    return {
        "enabled": True,
        "group_by": args.cv_group_by,
        "groups": unique_groups,
        "folds": folds,
        "aggregate": aggregate,
    }


def _group_shuffle_candidates(
    groups: list[str],
    *,
    test_size: float,
    random_state: int,
    attempts: int = 50,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = sorted(set(groups))
    test_group_count = max(1, int(np.ceil(len(unique_groups) * test_size)))
    test_group_count = min(test_group_count, len(unique_groups) - 1)
    rng = np.random.default_rng(random_state)
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(attempts):
        shuffled = list(unique_groups)
        rng.shuffle(shuffled)
        test_groups = set(shuffled[:test_group_count])
        train_idx = np.asarray(
            [index for index, group in enumerate(groups) if group not in test_groups],
            dtype=np.int64,
        )
        test_idx = np.asarray(
            [index for index, group in enumerate(groups) if group in test_groups],
            dtype=np.int64,
        )
        candidates.append((train_idx, test_idx))
    return candidates


def _random_split(
    x: np.ndarray,
    y: np.ndarray,
    row_ids: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    test_count = max(1, int(np.ceil(len(row_ids) * float(args.test_size))))
    test_count = min(test_count, len(row_ids) - 1)
    rng = np.random.default_rng(args.random_state)
    indices = np.arange(len(row_ids), dtype=np.int64)
    rng.shuffle(indices)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return (
        x[train_idx],
        x[test_idx],
        y[train_idx],
        y[test_idx],
        [row_ids[index] for index in train_idx],
        [row_ids[index] for index in test_idx],
    )


def _train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    row_ids: list[str],
    groups: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    unique_groups = sorted(set(groups))
    if len(unique_groups) >= 2:
        group_test_size = args.test_size
        if isinstance(group_test_size, float):
            group_test_size = max(group_test_size, min(0.5, 2 / len(unique_groups)))
        split_candidates = _group_shuffle_candidates(
            groups,
            test_size=group_test_size,
            random_state=args.random_state,
        )
        first_split: tuple[np.ndarray, np.ndarray] | None = None
        train_idx: np.ndarray | None = None
        test_idx: np.ndarray | None = None
        split_warning: str | None = None
        for candidate_train_idx, candidate_test_idx in split_candidates:
            if first_split is None:
                first_split = (candidate_train_idx, candidate_test_idx)
            train_classes = set(y[candidate_train_idx].tolist())
            test_classes = set(y[candidate_test_idx].tolist())
            if len(train_classes) >= 2 and len(test_classes) >= 2:
                train_idx = candidate_train_idx
                test_idx = candidate_test_idx
                break
        if train_idx is None or test_idx is None:
            train_idx, test_idx = first_split or split_candidates[0]
            split_warning = "could not find a group holdout split containing both classes"
        split_info = {
            "method": "group_shuffle",
            "group_by": args.cv_group_by,
            "requested_test_size": args.test_size,
            "effective_group_test_size": group_test_size,
            "train_groups": sorted({groups[index] for index in train_idx}),
            "test_groups": sorted({groups[index] for index in test_idx}),
            "train_class_counts": _class_counts(y[train_idx]),
            "test_class_counts": _class_counts(y[test_idx]),
        }
        if split_warning:
            split_info["warning"] = split_warning
        return (
            x[train_idx],
            x[test_idx],
            y[train_idx],
            y[test_idx],
            [row_ids[index] for index in train_idx],
            [row_ids[index] for index in test_idx],
            split_info,
        )

    x_train, x_test, y_train, y_test, ids_train, ids_test = _random_split(
        x,
        y,
        row_ids,
        args,
    )
    split_info = {
        "method": "random",
        "group_by": None,
        "reason": "not enough groups",
        "train_class_counts": _class_counts(y_train),
        "test_class_counts": _class_counts(y_test),
    }
    return x_train, x_test, y_train, y_test, ids_train, ids_test, split_info


def main() -> int:
    import joblib

    args = parse_args()
    rows = _select_rows(_read_jsonl(args.manifest), args.max_videos)
    if not rows:
        raise SystemExit("no rows selected")

    def prepare_sequence(item: tuple[int, dict[str, Any]]) -> None:
        index, row = item
        sequence_dir = _sequence_dir(args.feature_cache, row, args.max_frames)
        if args.force_extract or not _is_sequence_ready(sequence_dir):
            print(
                f"[{index}/{len(rows)}] extract {_safe_id(row)} -> {sequence_dir}",
                flush=True,
            )
            _extract_features(
                mediapipe_python=args.mediapipe_python,
                video_path=PROJECT_ROOT / str(row["video_path"]),
                output_dir=sequence_dir,
                max_frames=args.max_frames,
            )
        else:
            print(f"[{index}/{len(rows)}] cache {_safe_id(row)}", flush=True)

    worker_count = max(1, int(args.extract_workers))
    indexed_rows = list(enumerate(rows, start=1))
    if worker_count == 1:
        for item in indexed_rows:
            prepare_sequence(item)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(prepare_sequence, indexed_rows))

    features: list[np.ndarray] = []
    labels: list[int] = []
    row_ids: list[str] = []
    groups: list[str] = []
    for row in rows:
        sequence_dir = _sequence_dir(args.feature_cache, row, args.max_frames)
        features.append(_load_sequence(sequence_dir))
        labels.append(_label_for_row(row))
        row_ids.append(_safe_id(row))
        groups.append(_group_for_row(row, args.cv_group_by))

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    counts = _class_counts(y)
    if len(set(y.tolist())) < 2:
        raise SystemExit(f"need both fall and non-fall classes, got {counts}")

    x_train, x_test, y_train, y_test, ids_train, ids_test, split_info = _train_test_split(
        x,
        y,
        row_ids,
        groups,
        args,
    )

    cross_validation = _cross_validate(x, y, row_ids, groups, args)

    model = _build_model(args)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = (
        model.predict_proba(x_test).tolist()
        if hasattr(model, "predict_proba")
        else None
    )

    holdout = _evaluate_predictions(
        y_test,
        predictions,
        probabilities=probabilities,
        scene_ids=ids_test,
    )
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "dataset_version": args.dataset_version,
        "feature_cache": str(args.feature_cache),
        "output_model": str(args.output_model),
        "max_frames": args.max_frames,
        "rows": len(rows),
        "class_counts": counts,
        "dataset_summary": _dataset_summary(rows, groups),
        "train_ids": ids_train,
        "test_ids": ids_test,
        "holdout_split": split_info,
        "labels": {"fall": FALL_LABEL, "non_fall": NON_FALL_LABEL},
        "model_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "class_weight": "balanced",
            "random_state": args.random_state,
        },
        "holdout": holdout,
        "holdout_errors": _prediction_error_summary(holdout),
        "classification_report": holdout["classification_report"],
        "confusion_matrix_labels": holdout["confusion_matrix_labels"],
        "confusion_matrix": holdout["confusion_matrix"],
        "test_predictions": holdout["predictions"],
        "cross_validation": cross_validation,
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_model)
    args.metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"features: {x.shape}")
    print(f"class_counts: {counts}")
    print(f"holdout confusion_matrix labels={holdout['confusion_matrix_labels']}:")
    print(np.asarray(holdout["confusion_matrix"]))
    if cross_validation.get("enabled"):
        cv_matrix = np.asarray(cross_validation["aggregate"]["confusion_matrix"])
        print(
            f"group_cv({args.cv_group_by}) confusion_matrix "
            f"labels={cross_validation['aggregate']['confusion_matrix_labels']}:"
        )
        print(cv_matrix)
    print(f"model: {args.output_model}")
    print(f"metrics: {args.metrics_json}")
    print(
        "manifest_readiness: "
        f"python scripts/health/check_fall_manifest_readiness.py --manifest {args.manifest}"
    )
    print(
        "promotion_check: "
        f"python scripts/health/check_falldata_model_report.py --metrics-json {args.metrics_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
