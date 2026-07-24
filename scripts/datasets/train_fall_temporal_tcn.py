#!/usr/bin/env python3
"""Train a lightweight temporal fall classifier from cached YOLO-pose frames."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.train_yolo_pose_fall_rf import (  # noqa: E402
    DEFAULT_FEATURE_CACHE,
    DEFAULT_MANIFEST,
    DEFAULT_VALIDATION_FEATURE_CACHE,
    DEFAULT_VALIDATION_MANIFEST,
    _class_counts,
    _dataset_summary,
    _feature_path,
    _group_holdout_indices,
    _read_jsonl,
    _safe_id,
    _select_rows,
    _summarize_frames,
)
from src.core.ai.fall_temporal_model import (  # noqa: E402
    FRAME_FEATURE_NAMES,
    FallTemporalHybrid,
    FallTemporalTCN,
    encode_frame_sequence,
)

DEFAULT_OUTPUT_MODEL = PROJECT_ROOT / "models/experiments/fall_temporal_tcn.pt"
DEFAULT_METRICS = PROJECT_ROOT / "models/experiments/fall_temporal_tcn_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument(
        "--validation-feature-cache",
        type=Path,
        default=DEFAULT_VALIDATION_FEATURE_CACHE,
    )
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--dataset-version", default="fall_temporal_tcn")
    parser.add_argument("--max-videos", type=int, default=200)
    parser.add_argument("--validation-max-videos", type=int, default=80)
    parser.add_argument(
        "--all-cached",
        action="store_true",
        help="Train from every manifest row with an existing feature-cache file.",
    )
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--min-pose-frames", type=int, default=3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument(
        "--model-type",
        choices=("temporal", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def _load_cached_dataset(
    rows: list[dict[str, Any]],
    *,
    feature_cache: Path,
    max_frames: int,
    frame_stride: int,
    sequence_length: int,
    min_pose_frames: int,
) -> dict[str, Any]:
    sequences: list[np.ndarray] = []
    summaries: list[np.ndarray] = []
    labels: list[int] = []
    scene_ids: list[str] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        feature_path = _feature_path(feature_cache, row, max_frames, frame_stride)
        if not feature_path.exists():
            excluded.append({"scene_id": _safe_id(row), "reason": "feature_cache_missing"})
            continue
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
        frame_records = list(payload.get("frame_records") or [])
        if len(frame_records) < min_pose_frames:
            excluded.append(
                {
                    "scene_id": _safe_id(row),
                    "reason": "frames_with_pose_below_minimum",
                    "frames_with_pose": len(frame_records),
                }
            )
            continue
        sequences.append(
            encode_frame_sequence(
                frame_records,
                sequence_length=sequence_length,
            )
        )
        summary = _summarize_frames(
            frame_records,
            int(payload.get("frames_seen") or len(frame_records)),
        )
        summaries.append(np.asarray(summary["feature_vector"], dtype=np.float32))
        labels.append(1 if bool(row.get("is_fall")) else 0)
        scene_ids.append(_safe_id(row))
    if not sequences:
        raise SystemExit("no cached sequences available for temporal training")
    return {
        "x": np.stack(sequences).astype(np.float32),
        "summary_x": np.stack(summaries).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "scene_ids": scene_ids,
        "excluded": excluded,
    }


def _select_cached_rows(
    rows: list[dict[str, Any]],
    *,
    feature_cache: Path,
    max_frames: int,
    frame_stride: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _feature_path(feature_cache, row, max_frames, frame_stride).exists()
    ]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _fit_summary_standardizer(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (features - mean) / scale, mean, scale


def _binary_metrics(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    scene_ids: list[str],
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positives = []
    false_negatives = []
    for scene_id, label, prediction, probability in zip(
        scene_ids,
        labels.tolist(),
        predictions.tolist(),
        probabilities.tolist(),
    ):
        item = {
            "scene_id": scene_id,
            "true": int(label),
            "predicted": int(prediction),
            "fall_probability": float(probability),
        }
        if label == 0 and prediction == 1:
            false_positives.append(item)
        elif label == 1 and prediction == 0:
            false_negatives.append(item)
    return {
        "threshold": threshold,
        "confusion_matrix_labels": ["fall", "non_fall"],
        "confusion_matrix": [
            [true_positive, false_negative],
            [false_positive, true_negative],
        ],
        "classification_report": {
            "fall": {
                "precision": _safe_ratio(true_positive, true_positive + false_positive),
                "recall": _safe_ratio(true_positive, true_positive + false_negative),
                "support": true_positive + false_negative,
            },
            "non_fall": {
                "precision": _safe_ratio(true_negative, true_negative + false_negative),
                "recall": _safe_ratio(true_negative, true_negative + false_positive),
                "support": true_negative + false_positive,
            },
            "accuracy": _safe_ratio(true_positive + true_negative, len(labels)),
        },
        "errors": {
            "false_positive_count": false_positive,
            "false_negative_count": false_negative,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
    }


def _predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    summary_features: np.ndarray,
    *,
    model_type: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    results: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            summary_batch = torch.from_numpy(
                summary_features[start : start + batch_size]
            ).to(device)
            logits = (
                model(batch, summary_batch)
                if model_type == "hybrid"
                else model(batch)
            )
            results.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(results)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def main() -> int:
    args = parse_args()
    _seed_everything(args.random_state)
    device = _resolve_device(args.device)

    all_train_rows = _read_jsonl(args.manifest)
    all_validation_rows = _read_jsonl(args.validation_manifest)
    if args.all_cached:
        train_rows = _select_cached_rows(
            all_train_rows,
            feature_cache=args.feature_cache,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
        )
        validation_rows = _select_cached_rows(
            all_validation_rows,
            feature_cache=args.validation_feature_cache,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
        )
    else:
        train_rows = _select_rows(all_train_rows, args.max_videos)
        validation_rows = _select_rows(
            all_validation_rows,
            args.validation_max_videos,
        )
    train_dataset = _load_cached_dataset(
        train_rows,
        feature_cache=args.feature_cache,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        sequence_length=args.sequence_length,
        min_pose_frames=args.min_pose_frames,
    )
    validation_dataset = _load_cached_dataset(
        validation_rows,
        feature_cache=args.validation_feature_cache,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        sequence_length=args.sequence_length,
        min_pose_frames=args.min_pose_frames,
    )
    train_indices, holdout_indices, holdout_split = _group_holdout_indices(
        train_dataset["scene_ids"],
        train_dataset["y"],
        test_size=0.25,
        random_state=args.random_state,
    )
    x_train = train_dataset["x"][train_indices]
    y_train = train_dataset["y"][train_indices]
    x_holdout = train_dataset["x"][holdout_indices]
    y_holdout = train_dataset["y"][holdout_indices]
    summary_train, summary_mean, summary_scale = _fit_summary_standardizer(
        train_dataset["summary_x"][train_indices]
    )
    summary_holdout = (
        train_dataset["summary_x"][holdout_indices] - summary_mean
    ) / summary_scale
    summary_validation = (
        validation_dataset["summary_x"] - summary_mean
    ) / summary_scale
    holdout_ids = [train_dataset["scene_ids"][index] for index in holdout_indices]

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(summary_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.random_state),
    )
    if args.model_type == "hybrid":
        model = FallTemporalHybrid(
            input_features=len(FRAME_FEATURE_NAMES),
            summary_features=summary_train.shape[1],
            channels=args.channels,
        ).to(device)
    else:
        model = FallTemporalTCN(
            input_features=len(FRAME_FEATURE_NAMES),
            channels=args.channels,
        ).to(device)
    positive_count = max(int((y_train == 1).sum()), 1)
    negative_count = max(int((y_train == 0).sum()), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative_count / positive_count], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history = []
    holdout_x_tensor = torch.from_numpy(x_holdout).to(device)
    holdout_summary_tensor = torch.from_numpy(
        summary_holdout.astype(np.float32)
    ).to(device)
    holdout_y_tensor = torch.from_numpy(y_holdout.astype(np.float32)).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        for batch_x, batch_summary, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_summary = batch_summary.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = (
                model(batch_x, batch_summary)
                if args.model_type == "hybrid"
                else model(batch_x)
            )
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * len(batch_x)
        model.eval()
        with torch.inference_mode():
            holdout_logits = (
                model(holdout_x_tensor, holdout_summary_tensor)
                if args.model_type == "hybrid"
                else model(holdout_x_tensor)
            )
            holdout_loss = float(criterion(holdout_logits, holdout_y_tensor).item())
        train_loss = train_loss_total / len(x_train)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "holdout_loss": holdout_loss}
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
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"holdout_loss={holdout_loss:.5f}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            break
    if best_state is None:
        raise SystemExit("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    holdout_probabilities = _predict_probabilities(
        model,
        x_holdout,
        summary_holdout.astype(np.float32),
        model_type=args.model_type,
        device=device,
        batch_size=args.batch_size,
    )
    validation_probabilities = _predict_probabilities(
        model,
        validation_dataset["x"],
        summary_validation.astype(np.float32),
        model_type=args.model_type,
        device=device,
        batch_size=args.batch_size,
    )
    holdout = _binary_metrics(
        labels=y_holdout,
        probabilities=holdout_probabilities,
        threshold=args.decision_threshold,
        scene_ids=holdout_ids,
    )
    validation = _binary_metrics(
        labels=validation_dataset["y"],
        probabilities=validation_probabilities,
        threshold=args.decision_threshold,
        scene_ids=validation_dataset["scene_ids"],
    )
    threshold_sweep = [
        _binary_metrics(
            labels=validation_dataset["y"],
            probabilities=validation_probabilities,
            threshold=round(float(threshold), 2),
            scene_ids=validation_dataset["scene_ids"],
        )
        for threshold in np.arange(0.35, 0.91, 0.05)
    ]

    checkpoint = {
        "format_version": 1,
        "model_type": f"fall_temporal_{args.model_type}",
        "state_dict": best_state,
        "input_features": len(FRAME_FEATURE_NAMES),
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "sequence_length": args.sequence_length,
        "channels": args.channels,
        "summary_features": int(summary_train.shape[1]),
        "summary_mean": summary_mean.tolist(),
        "summary_scale": summary_scale.tolist(),
        "decision_threshold": args.decision_threshold,
    }
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": args.dataset_version,
        "manifest": str(args.manifest),
        "validation_manifest": str(args.validation_manifest),
        "output_model": str(args.output_model),
        "model_type": f"fall_temporal_{args.model_type}",
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "rows": len(train_rows),
        "effective_rows": len(train_dataset["y"]),
        "validation_rows": len(validation_rows),
        "validation_effective_rows": len(validation_dataset["y"]),
        "class_counts": _class_counts(train_dataset["y"]),
        "dataset_summary": _dataset_summary(
            train_dataset["scene_ids"],
            train_dataset["y"],
        ),
        "excluded": train_dataset["excluded"],
        "validation_excluded": validation_dataset["excluded"],
        "model_params": {
            "sequence_length": args.sequence_length,
            "input_features": len(FRAME_FEATURE_NAMES),
            "channels": args.channels,
            "model_type": args.model_type,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "decision_threshold": args.decision_threshold,
            "random_state": args.random_state,
        },
        "training": {
            "epochs_completed": len(history),
            "best_holdout_loss": best_loss,
            "history": history,
        },
        "holdout_method": holdout_split["method"],
        "holdout_split": holdout_split,
        "holdout": holdout,
        "holdout_errors": holdout["errors"],
        "validation": validation,
        "validation_threshold_sweep": [
            {
                "threshold": item["threshold"],
                "fall_precision": item["classification_report"]["fall"]["precision"],
                "fall_recall": item["classification_report"]["fall"]["recall"],
                "false_positive_count": item["errors"]["false_positive_count"],
                "false_negative_count": item["errors"]["false_negative_count"],
            }
            for item in threshold_sweep
        ],
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_model)
    args.metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"device={device} validation={validation['confusion_matrix']} "
        f"precision={validation['classification_report']['fall']['precision']:.4f} "
        f"recall={validation['classification_report']['fall']['recall']:.4f}"
    )
    print(f"model: {args.output_model}")
    print(f"metrics: {args.metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
