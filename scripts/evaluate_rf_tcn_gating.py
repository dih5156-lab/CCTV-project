#!/usr/bin/env python3
"""Evaluate RF/TCN gating policies on the same cached validation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.datasets.train_fall_temporal_tcn import _load_cached_dataset, _read_jsonl
from src.core.ai.fall_temporal_model import FallTemporalHybrid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--rf-model", type=Path, required=True)
    parser.add_argument("--tcn-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--margin", type=int, default=120)
    args = parser.parse_args()

    rows = _read_jsonl(args.manifest)
    dataset = _load_cached_dataset(
        rows,
        feature_cache=args.feature_cache,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        fall_window_margin_frames=args.margin,
        sequence_length=60,
        min_pose_frames=3,
    )
    rf_bundle = joblib.load(args.rf_model)
    rf_model = rf_bundle.get("model", rf_bundle)
    rf_features = np.asarray(dataset["summary_x"], dtype=np.float32)
    rf_prob = rf_model.predict_proba(rf_features)[:, list(rf_model.classes_).index(1)]

    checkpoint = torch.load(args.tcn_model, map_location="cpu", weights_only=False)
    tcn = FallTemporalHybrid(
        input_features=int(checkpoint["input_features"]),
        summary_features=int(checkpoint["summary_features"]),
        channels=int(checkpoint["channels"]),
    )
    tcn.load_state_dict(checkpoint["state_dict"])
    tcn.eval()
    sequences = torch.from_numpy(dataset["x"])
    summary = torch.from_numpy(
        (dataset["summary_x"] - np.asarray(checkpoint["summary_mean"]))
        / np.asarray(checkpoint["summary_scale"])
    ).float()
    with torch.inference_mode():
        tcn_prob = torch.sigmoid(tcn(sequences, summary)).numpy()

    y = np.asarray(dataset["y"], dtype=np.int64)
    threshold = 0.7
    policies = {
        "rf": rf_prob >= threshold,
        "tcn": tcn_prob >= threshold,
        "and": (rf_prob >= threshold) & (tcn_prob >= threshold),
        "or": (rf_prob >= threshold) | (tcn_prob >= threshold),
        "rf_borderline_tcn": (rf_prob >= threshold)
        | ((rf_prob >= 0.5) & (tcn_prob >= threshold)),
    }
    for tcn_threshold in (0.75, 0.8, 0.85):
        policies[f"or_tcn_{tcn_threshold:.2f}"] = (rf_prob >= threshold) | (
            tcn_prob >= tcn_threshold
        )
    result = {"rows": int(len(y)), "threshold": threshold, "policies": {}}
    for name, pred in policies.items():
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result["policies"][name] = {
            "precision": precision,
            "recall": recall,
            "false_positive_count": fp,
            "false_negative_count": fn,
            "true_positive_count": tp,
            "true_negative_count": tn,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
