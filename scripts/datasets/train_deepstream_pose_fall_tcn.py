#!/usr/bin/env python3
"""Train a lightweight temporal fall model from DeepStream pose captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ai.fall_temporal_model import (  # noqa: E402
    FRAME_FEATURE_NAMES,
    FallTemporalTCN,
    encode_frame_sequence,
)


@dataclass(frozen=True)
class TemporalCaptureDataset:
    sequences: np.ndarray
    labels: np.ndarray
    scene_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    patience: int = 15
    batch_size: int = 16
    channels: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_state: int = 42
    device: str = "auto"
    sequence_length: int = 48
    minimum_threshold: float = 0.70
    minimum_recall: float = 0.75
    maximum_false_positive_rate: float = 0.10


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def load_temporal_capture_datasets(
    paths: Sequence[Path],
    *,
    sequence_length: int = 48,
) -> TemporalCaptureDataset:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    scene_ids: list[str] = []
    group_ids: list[str] = []
    metadata: list[dict[str, Any]] = []

    for path in paths:
        for row_index, row in enumerate(_read_jsonl(path)):
            row_name = f"{path}: row {row_index}"
            if int(row.get("schema_version") or 0) != 2:
                raise ValueError(f"{row_name}: schema_version must be 2")
            if row.get("runtime") != "deepstream_pose_inline":
                raise ValueError(f"{row_name}: unexpected runtime")
            if tuple(row.get("frame_feature_names") or ()) != FRAME_FEATURE_NAMES:
                raise ValueError(f"{row_name}: frame feature order mismatch")

            frame_records = row.get("frame_records")
            if not isinstance(frame_records, list) or not frame_records:
                raise ValueError(f"{row_name}: frame_records must be non-empty")

            label = row.get("label")
            if label not in {0, 1}:
                raise ValueError(f"{row_name}: label must be 0 or 1")
            scene_id = str(row.get("scene_id") or "").strip()
            group_id = str(row.get("group_id") or "").strip()
            if not scene_id:
                raise ValueError(f"{row_name}: scene_id is required")
            if not group_id:
                raise ValueError(f"{row_name}: group_id is required")

            sequence = encode_frame_sequence(
                frame_records,
                sequence_length=sequence_length,
            )
            if not np.isfinite(sequence).all():
                raise ValueError(f"{row_name}: non-finite sequence")

            sequences.append(sequence)
            labels.append(int(label))
            scene_ids.append(scene_id)
            group_ids.append(group_id)
            metadata.append(dict(row))

    if not sequences:
        raise ValueError("no temporal capture rows found")
    if set(labels) != {0, 1}:
        raise ValueError("dataset must contain both fall and non-fall rows")

    return TemporalCaptureDataset(
        sequences=np.stack(sequences).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scene_ids=tuple(scene_ids),
        group_ids=tuple(group_ids),
        metadata=tuple(metadata),
    )


def assert_validation_disjoint(
    training: TemporalCaptureDataset,
    validation: TemporalCaptureDataset,
) -> None:
    scene_overlap = sorted(set(training.scene_ids) & set(validation.scene_ids))
    if scene_overlap:
        raise ValueError(f"scene overlap: {', '.join(scene_overlap)}")

    group_overlap = sorted(set(training.group_ids) & set(validation.group_ids))
    if group_overlap:
        raise ValueError(f"group overlap: {', '.join(group_overlap)}")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    scene_ids: Sequence[str],
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "fall_precision": _safe_ratio(
            true_positive,
            true_positive + false_positive,
        ),
        "fall_recall": _safe_ratio(
            true_positive,
            true_positive + false_negative,
        ),
        "false_positive_rate": _safe_ratio(
            false_positive,
            false_positive + true_negative,
        ),
        "fall_support": true_positive + false_negative,
        "normal_support": false_positive + true_negative,
        "predictions": [
            {
                "scene_id": str(scene_id),
                "label": int(label),
                "prediction": int(prediction),
                "fall_probability": float(probability),
            }
            for scene_id, label, prediction, probability in zip(
                scene_ids,
                labels.tolist(),
                predictions.tolist(),
                probabilities.tolist(),
            )
        ],
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_threshold: float = 0.70,
    minimum_recall: float = 0.75,
    maximum_false_positive_rate: float = 0.10,
    scene_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have matching shapes")
    effective_scene_ids = (
        tuple(scene_ids)
        if scene_ids is not None
        else tuple(str(index) for index in range(len(labels)))
    )
    thresholds = np.arange(minimum_threshold, 0.951, 0.05)
    sweep = [
        _binary_metrics(
            labels,
            probabilities,
            threshold=round(float(threshold), 2),
            scene_ids=effective_scene_ids,
        )
        for threshold in thresholds
    ]
    selected = next(
        (
            item
            for item in sweep
            if item["fall_recall"] >= minimum_recall
            and item["false_positive_rate"] <= maximum_false_positive_rate
        ),
        None,
    )
    return {
        "passed": selected is not None,
        "decision_threshold": float(
            selected["threshold"] if selected is not None else minimum_threshold
        ),
        "selected": selected,
        "sweep": sweep,
        "requirements": {
            "minimum_threshold": minimum_threshold,
            "minimum_recall": minimum_recall,
            "maximum_false_positive_rate": maximum_false_positive_rate,
        },
    }


def _group_holdout_indices(
    dataset: TemporalCaptureDataset,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    group_labels: dict[str, int] = {}
    for group_id, label in zip(dataset.group_ids, dataset.labels.tolist()):
        previous = group_labels.setdefault(group_id, int(label))
        if previous != int(label):
            raise ValueError(f"group has mixed labels: {group_id}")

    groups_by_label: dict[int, list[str]] = {0: [], 1: []}
    for group_id, label in group_labels.items():
        groups_by_label[label].append(group_id)
    if any(len(groups) < 2 for groups in groups_by_label.values()):
        raise ValueError("training requires at least two groups per class")

    rng = random.Random(random_state)
    holdout_groups: set[str] = set()
    for label in (0, 1):
        groups = sorted(groups_by_label[label])
        rng.shuffle(groups)
        holdout_count = max(1, int(round(len(groups) * 0.25)))
        holdout_count = min(holdout_count, len(groups) - 1)
        holdout_groups.update(groups[:holdout_count])

    holdout_indices = np.asarray(
        [
            index
            for index, group_id in enumerate(dataset.group_ids)
            if group_id in holdout_groups
        ],
        dtype=np.int64,
    )
    train_indices = np.asarray(
        [
            index
            for index, group_id in enumerate(dataset.group_ids)
            if group_id not in holdout_groups
        ],
        dtype=np.int64,
    )
    split_payload = {
        "method": "stratified_group_holdout",
        "random_state": random_state,
        "training_groups": sorted(
            set(dataset.group_ids) - holdout_groups
        ),
        "holdout_groups": sorted(holdout_groups),
    }
    split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    split_payload["split_hash"] = split_hash
    return train_indices, holdout_indices, split_payload


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {requested}")
    return torch.device(requested)


def _as_float_tensor(values: np.ndarray) -> torch.Tensor:
    # The host test environment mixes a NumPy 2 runtime with a PyTorch build
    # compiled against NumPy 1. Convert through Python lists at this boundary.
    return torch.tensor(values.tolist(), dtype=torch.float32)


def _predict_probabilities(
    model: nn.Module,
    sequences: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = _as_float_tensor(
                sequences[start : start + batch_size]
            ).to(device)
            probabilities.extend(
                torch.sigmoid(model(batch)).detach().cpu().tolist()
            )
    return np.asarray(probabilities, dtype=np.float32)


def _aggregate_scene_probabilities(
    *,
    scene_ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if len(scene_ids) != len(labels) or labels.shape != probabilities.shape:
        raise ValueError("scene_ids, labels, and probabilities must align")

    scene_labels: dict[str, int] = {}
    scene_probabilities: dict[str, float] = {}
    for scene_id, label, probability in zip(
        scene_ids,
        labels.tolist(),
        probabilities.tolist(),
    ):
        normalized_scene_id = str(scene_id)
        normalized_label = int(label)
        previous_label = scene_labels.setdefault(
            normalized_scene_id,
            normalized_label,
        )
        if previous_label != normalized_label:
            raise ValueError(f"scene has mixed labels: {normalized_scene_id}")
        scene_probabilities[normalized_scene_id] = max(
            scene_probabilities.get(normalized_scene_id, 0.0),
            float(probability),
        )

    ordered_scene_ids = tuple(scene_labels)
    return (
        ordered_scene_ids,
        np.asarray(
            [scene_labels[scene_id] for scene_id in ordered_scene_ids],
            dtype=np.int64,
        ),
        np.asarray(
            [scene_probabilities[scene_id] for scene_id in ordered_scene_ids],
            dtype=np.float32,
        ),
    )


def train_candidate(
    training: TemporalCaptureDataset,
    validation: TemporalCaptureDataset,
    config: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_validation_disjoint(training, validation)
    if training.sequences.shape[1] != config.sequence_length:
        raise ValueError("training sequence length does not match config")
    if validation.sequences.shape[1] != config.sequence_length:
        raise ValueError("validation sequence length does not match config")

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)
    device = _resolve_device(config.device)

    train_indices, holdout_indices, split = _group_holdout_indices(
        training,
        random_state=config.random_state,
    )
    x_train = training.sequences[train_indices]
    y_train = training.labels[train_indices]
    x_holdout = training.sequences[holdout_indices]
    y_holdout = training.labels[holdout_indices]

    train_dataset = TensorDataset(
        _as_float_tensor(x_train),
        torch.tensor(y_train.tolist(), dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_state),
    )
    model = FallTemporalTCN(
        input_features=len(FRAME_FEATURE_NAMES),
        channels=config.channels,
    ).to(device)
    positive_count = max(int((y_train == 1).sum()), 1)
    negative_count = max(int((y_train == 0).sum()), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [negative_count / positive_count],
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    holdout_x = _as_float_tensor(x_holdout).to(device)
    holdout_y = torch.tensor(
        y_holdout.tolist(),
        dtype=torch.float32,
        device=device,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)

        model.eval()
        with torch.inference_mode():
            holdout_loss = float(
                criterion(model(holdout_x), holdout_y).item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_dataset),
                "holdout_loss": holdout_loss,
            }
        )
        if holdout_loss < best_loss - 1e-5:
            best_loss = holdout_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_probabilities = _predict_probabilities(
        model,
        validation.sequences,
        device=device,
        batch_size=config.batch_size,
    )
    (
        validation_scene_ids,
        validation_scene_labels,
        validation_scene_probabilities,
    ) = _aggregate_scene_probabilities(
        scene_ids=validation.scene_ids,
        labels=validation.labels,
        probabilities=validation_probabilities,
    )
    threshold_result = select_threshold(
        validation_scene_labels,
        validation_scene_probabilities,
        minimum_threshold=config.minimum_threshold,
        minimum_recall=config.minimum_recall,
        maximum_false_positive_rate=config.maximum_false_positive_rate,
        scene_ids=validation_scene_ids,
    )
    decision_threshold = float(threshold_result["decision_threshold"])
    validation_metrics = _binary_metrics(
        validation_scene_labels,
        validation_scene_probabilities,
        threshold=decision_threshold,
        scene_ids=validation_scene_ids,
    )
    validation_metrics["window_rows"] = len(validation.labels)
    validation_metrics["scene_rows"] = len(validation_scene_labels)

    checkpoint = {
        "format_version": 2,
        "model_type": "deepstream_pose_temporal_tcn",
        "state_dict": best_state,
        "input_features": len(FRAME_FEATURE_NAMES),
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "sequence_length": config.sequence_length,
        "channels": config.channels,
        "decision_threshold": decision_threshold,
        "split_hash": split["split_hash"],
    }
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "training": {
            "rows": len(training.labels),
            "epochs_completed": len(history),
            "best_holdout_loss": best_loss,
            "history": history,
            "split": split,
        },
        "validation": validation_metrics,
        "threshold_selection": threshold_result,
        "passed": bool(threshold_result["passed"]),
    }
    return checkpoint, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dataset",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for output_path in (args.output_model, args.metrics_json):
        if output_path.exists() and not args.overwrite:
            raise SystemExit(
                f"output already exists; pass --overwrite: {output_path}"
            )

    training = load_temporal_capture_datasets(args.train_dataset)
    validation = load_temporal_capture_datasets(args.validation_dataset)
    checkpoint, metrics = train_candidate(
        training,
        validation,
        TrainingConfig(
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            channels=args.channels,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            random_state=args.random_state,
            device=args.device,
        ),
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_model)
    args.metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"passed={metrics['passed']} "
        f"threshold={checkpoint['decision_threshold']:.2f} "
        f"recall={metrics['validation']['fall_recall']:.4f} "
        f"fpr={metrics['validation']['false_positive_rate']:.4f}"
    )
    print(f"model: {args.output_model}")
    print(f"metrics: {args.metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
