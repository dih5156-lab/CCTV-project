#!/usr/bin/env python3
"""Compare falldata-compatible video classifiers on cached sample features."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/fall_eval/sample_manifest.jsonl"
DEFAULT_FEATURE_CACHE = PROJECT_ROOT / "data/fall_eval/falldata_feature_cache"
DEFAULT_BASELINE_MODEL = (
    PROJECT_ROOT / "falldata/2. AI학습모델파일/영상/낙상분류/FNF_RF_SMOTE_CAM_1.pkl"
)
DEFAULT_CANDIDATE_MODEL = PROJECT_ROOT / "models/experiments/falldata_sample_rf_max120_all.pkl"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "models/experiments/falldata_model_compare_max120.json"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "models/experiments/falldata_model_compare_max120.csv"
TARGET_FRAMES = 600
FRAME_FEATURES = 1662
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


def _load_sequence(sequence_dir: Path) -> np.ndarray:
    frame_files = sorted(
        sequence_dir.glob("*.npy"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if len(frame_files) != TARGET_FRAMES:
        raise ValueError(f"expected {TARGET_FRAMES} frames, found {len(frame_files)}: {sequence_dir}")
    sequence = np.asarray([np.load(path).reshape(-1) for path in frame_files], dtype=np.float32)
    if sequence.shape != (TARGET_FRAMES, FRAME_FEATURES):
        raise ValueError(f"unexpected sequence shape {sequence.shape}: {sequence_dir}")
    return sequence.reshape(-1)


def _label_for_row(row: dict[str, Any]) -> int:
    return FALL_LABEL if bool(row.get("is_fall")) else NON_FALL_LABEL


def _fall_probability(model: Any, sample: np.ndarray, prediction: int) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(sample)[0]
    classes = list(getattr(model, "classes_", []))
    if FALL_LABEL in classes:
        return float(probabilities[classes.index(FALL_LABEL)])
    return float(probabilities[int(prediction)]) if len(probabilities) else None


def _predict(model: Any, sample: np.ndarray) -> tuple[int, float | None]:
    prediction = int(model.predict(sample)[0])
    return prediction, _fall_probability(model, sample, prediction)


def _summary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[FALL_LABEL, NON_FALL_LABEL])
    tp = int(matrix[0][0])
    fn = int(matrix[0][1])
    fp = int(matrix[1][0])
    tn = int(matrix[1][1])
    return {
        "confusion_matrix_labels": ["fall", "non_fall"],
        "confusion_matrix": matrix.tolist(),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fn + fp + tn, 1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[FALL_LABEL, NON_FALL_LABEL],
            target_names=["fall", "non_fall"],
            zero_division=0,
            output_dict=True,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--candidate-model", type=Path, default=DEFAULT_CANDIDATE_MODEL)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(args.manifest)
    baseline = joblib.load(args.baseline_model)
    candidate = joblib.load(args.candidate_model)

    y_true: list[int] = []
    baseline_pred: list[int] = []
    candidate_pred: list[int] = []
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        scene_id = _safe_id(row)
        true_label = _label_for_row(row)
        sample = _load_sequence(args.feature_cache / f"{scene_id}_max{args.max_frames}").reshape(1, -1)
        base_label, base_fall_prob = _predict(baseline, sample)
        cand_label, cand_fall_prob = _predict(candidate, sample)

        y_true.append(true_label)
        baseline_pred.append(base_label)
        candidate_pred.append(cand_label)
        comparisons.append(
            {
                "scene_id": scene_id,
                "label": row.get("label"),
                "camera": row.get("camera"),
                "true": true_label,
                "baseline_predicted": base_label,
                "baseline_fall_probability": base_fall_prob,
                "candidate_predicted": cand_label,
                "candidate_fall_probability": cand_fall_prob,
                "baseline_correct": base_label == true_label,
                "candidate_correct": cand_label == true_label,
                "changed": base_label != cand_label,
                "video_path": row.get("video_path"),
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "feature_cache": str(args.feature_cache),
        "baseline_model": str(args.baseline_model),
        "candidate_model": str(args.candidate_model),
        "labels": {"fall": FALL_LABEL, "non_fall": NON_FALL_LABEL},
        "rows": len(comparisons),
        "baseline": _summary(y_true, baseline_pred),
        "candidate": _summary(y_true, candidate_pred),
        "comparisons": comparisons,
    }

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    for name in ("baseline", "candidate"):
        summary = payload[name]
        print(
            f"{name}: accuracy={summary['accuracy']:.3f} precision={summary['precision']:.3f} "
            f"recall={summary['recall']:.3f} TP={summary['tp']} FN={summary['fn']} "
            f"FP={summary['fp']} TN={summary['tn']}"
        )
    changed = [row for row in comparisons if row["changed"]]
    print(f"changed: {len(changed)}")
    print(f"results_json: {args.results_json}")
    print(f"results_csv: {args.results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
