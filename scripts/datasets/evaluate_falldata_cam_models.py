#!/usr/bin/env python3
"""Evaluate CAM-specific Falldata RF models on one shared video manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/fall_eval/field_combined_manifest.jsonl"
DEFAULT_FEATURE_CACHE = PROJECT_ROOT / "data/fall_eval/falldata_cam_eval_feature_cache"
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "models/legacy/falldata_mediapipe"
)
DEFAULT_MEDIAPIPE_PYTHON = PROJECT_ROOT / ".venv-mediapipe/bin/python"
DEFAULT_RESULTS_JSON = (
    PROJECT_ROOT / "models/experiments/falldata_cam_models_threshold_070.json"
)
DEFAULT_RESULTS_CSV = (
    PROJECT_ROOT / "models/experiments/falldata_cam_models_threshold_070.csv"
)
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts/datasets/extract_falldata_mediapipe_features.py"
TARGET_FRAMES = 600
FRAME_FEATURES = 1662
FALL_LABEL = 0
NON_FALL_LABEL = 1
CAMERA_PATTERN = re.compile(r"(?:CAMERA|CAM|C)[_-]?(\d+)", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _safe_id(row: dict[str, Any]) -> str:
    value = str(row.get("scene_id") or Path(str(row["video_path"])).stem)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _camera_number(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    match = CAMERA_PATTERN.search(str(value))
    return int(match.group(1)) if match else None


def _model_camera_number(path: Path) -> int | None:
    return _camera_number(path.stem)


def _sequence_dir(cache_dir: Path, row: dict[str, Any], max_frames: int) -> Path:
    return cache_dir / f"{_safe_id(row)}_max{max_frames}"


def _sequence_ready(sequence_dir: Path) -> bool:
    return len(list(sequence_dir.glob("*.npy"))) == TARGET_FRAMES


def _video_path(row: dict[str, Any]) -> Path:
    video_path = Path(str(row["video_path"]))
    return video_path if video_path.is_absolute() else PROJECT_ROOT / video_path


def _extract_one(
    *,
    row: dict[str, Any],
    cache_dir: Path,
    max_frames: int,
    mediapipe_python: Path,
) -> None:
    output_dir = _sequence_dir(cache_dir, row, max_frames)
    if _sequence_ready(output_dir):
        return
    video_path = _video_path(row)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(mediapipe_python),
            str(EXTRACT_SCRIPT),
            "--video",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--max-frames",
            str(max_frames),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _prepare_features(
    *,
    rows: list[dict[str, Any]],
    cache_dir: Path,
    max_frames: int,
    mediapipe_python: Path,
    workers: int,
) -> None:
    def prepare(indexed_row: tuple[int, dict[str, Any]]) -> None:
        index, row = indexed_row
        sequence_dir = _sequence_dir(cache_dir, row, max_frames)
        status = "cache" if _sequence_ready(sequence_dir) else "extract"
        print(f"[{index}/{len(rows)}] {status} {_safe_id(row)}", flush=True)
        _extract_one(
            row=row,
            cache_dir=cache_dir,
            max_frames=max_frames,
            mediapipe_python=mediapipe_python,
        )

    indexed_rows = list(enumerate(rows, start=1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(prepare, indexed_rows))


def _transform_sequence(
    sequence: np.ndarray,
    *,
    source_frames: int,
    mode: str,
) -> np.ndarray:
    if mode == "postpad" or source_frames >= TARGET_FRAMES:
        return sequence
    source = sequence[:source_frames]
    if mode == "tail_align":
        transformed = np.zeros_like(sequence)
        transformed[-source_frames:] = source
        return transformed
    if mode == "stretch":
        source_indices = np.linspace(
            0,
            source_frames - 1,
            num=TARGET_FRAMES,
        ).round().astype(np.int64)
        return source[source_indices]
    raise ValueError(f"unknown sequence transform: {mode}")


def _load_sequence(
    sequence_dir: Path,
    *,
    source_frames: int,
    transform: str,
) -> tuple[np.ndarray, int]:
    frame_files = sorted(
        sequence_dir.glob("*.npy"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if len(frame_files) != TARGET_FRAMES:
        raise ValueError(
            f"expected {TARGET_FRAMES} frames, found {len(frame_files)}: {sequence_dir}"
        )
    sequence = np.asarray(
        [np.load(path).reshape(-1) for path in frame_files],
        dtype=np.float32,
    )
    if sequence.shape != (TARGET_FRAMES, FRAME_FEATURES):
        raise ValueError(f"unexpected sequence shape {sequence.shape}: {sequence_dir}")
    sequence = _transform_sequence(
        sequence,
        source_frames=min(max(source_frames, 1), TARGET_FRAMES),
        mode=transform,
    )
    nonzero_frames = int(np.any(sequence != 0, axis=1).sum())
    return sequence.reshape(1, -1), nonzero_frames


def _fall_probability(model: Any, sample: np.ndarray) -> float:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"model does not support predict_proba: {type(model).__name__}")
    classes = list(getattr(model, "classes_", []))
    if FALL_LABEL not in classes:
        raise ValueError(f"fall class {FALL_LABEL} missing from model classes: {classes}")
    probabilities = model.predict_proba(sample)[0]
    return float(probabilities[classes.index(FALL_LABEL)])


def _patch_legacy_sklearn_model(model: Any) -> Any:
    """현재 sklearn에서 필요한 기본 속성을 구버전 모델에 보완한다.

    카메라별 모델은 sklearn 1.3.x에서 저장되어 1.7.x에서 로드할 때
    DecisionTreeClassifier의 ``monotonic_cst`` 속성이 누락될 수 있다.
    모델 가중치는 변경하지 않고 호환 기본값만 주입한다.
    """
    estimators = getattr(model, "estimators_", ())
    for estimator in estimators:
        if hasattr(estimator, "monotonic_cst"):
            continue
        setattr(estimator, "monotonic_cst", None)
    return model


def _prediction_summary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if not y_true:
        return {
            "rows": 0,
            "tp": 0,
            "fn": 0,
            "fp": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
        }
    matrix = confusion_matrix(y_true, y_pred, labels=[FALL_LABEL, NON_FALL_LABEL])
    tp, fn = (int(value) for value in matrix[0])
    fp, tn = (int(value) for value in matrix[1])
    return {
        "rows": len(y_true),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / len(y_true),
    }


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _prediction_summary(
        [int(row["true"]) for row in records],
        [int(row["predicted"]) for row in records],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-glob", default="FNF_RF_SMOTE_CAM_*.pkl")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--sequence-transform",
        choices=("postpad", "tail_align", "stretch"),
        default="postpad",
        help="How extracted frames are placed into the 600-frame model input.",
    )
    parser.add_argument("--fall-threshold", type=float, default=0.7)
    parser.add_argument("--mediapipe-python", type=Path, default=DEFAULT_MEDIAPIPE_PYTHON)
    parser.add_argument("--extract-workers", type=int, default=2)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.fall_threshold <= 1.0:
        raise SystemExit("--fall-threshold must be between 0 and 1")
    rows = _read_jsonl(args.manifest)
    model_paths = sorted(args.model_dir.glob(args.model_glob))
    if not rows:
        raise SystemExit(f"manifest has no rows: {args.manifest}")
    if not model_paths:
        raise SystemExit(f"no models matched {args.model_glob!r} in {args.model_dir}")

    models: dict[str, Any] = {}
    model_cameras: dict[str, int | None] = {}
    for path in model_paths:
        models[path.name] = _patch_legacy_sklearn_model(joblib.load(path))
        model_cameras[path.name] = _model_camera_number(path)

    available_rows: list[dict[str, Any]] = []
    missing_videos: list[dict[str, str]] = []
    for row in rows:
        if _sequence_ready(_sequence_dir(args.feature_cache, row, args.max_frames)):
            available_rows.append(row)
            continue
        video_path = _video_path(row)
        if video_path.is_file():
            available_rows.append(row)
        else:
            missing_videos.append(
                {"scene_id": _safe_id(row), "video_path": str(video_path)}
            )
    if not available_rows:
        raise SystemExit("no videos or complete feature caches are available")
    if missing_videos:
        print(
            f"warning: skipping {len(missing_videos)} missing videos "
            f"out of {len(rows)} manifest rows",
            flush=True,
        )

    _prepare_features(
        rows=available_rows,
        cache_dir=args.feature_cache,
        max_frames=args.max_frames,
        mediapipe_python=args.mediapipe_python,
        workers=args.extract_workers,
    )

    model_records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in models
    }
    camera_matched_records: list[dict[str, Any]] = []
    ensemble_records: list[dict[str, Any]] = []
    feature_quality: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for index, row in enumerate(available_rows, start=1):
        scene_id = _safe_id(row)
        true_label = FALL_LABEL if bool(row.get("is_fall")) else NON_FALL_LABEL
        camera = _camera_number(row.get("camera"))
        sample, nonzero_frames = _load_sequence(
            _sequence_dir(args.feature_cache, row, args.max_frames),
            source_frames=args.max_frames,
            transform=args.sequence_transform,
        )
        feature_quality.append(
            {"scene_id": scene_id, "nonzero_feature_frames": nonzero_frames}
        )
        probabilities: dict[str, float] = {}
        for model_name, model in models.items():
            probability = _fall_probability(model, sample)
            predicted = (
                FALL_LABEL
                if probability >= args.fall_threshold
                else NON_FALL_LABEL
            )
            record = {
                "scene_id": scene_id,
                "camera": camera,
                "source": row.get("source", "sample"),
                "true": true_label,
                "predicted": predicted,
                "fall_probability": probability,
            }
            probabilities[model_name] = probability
            model_records[model_name].append(record)
            csv_rows.append({"strategy": model_name, **record})
            if model_cameras[model_name] == camera:
                matched_record = {"strategy": "camera_matched", **record}
                camera_matched_records.append(matched_record)
                csv_rows.append(matched_record)

        ensemble_probability = float(np.mean(list(probabilities.values())))
        ensemble_predicted = (
            FALL_LABEL
            if ensemble_probability >= args.fall_threshold
            else NON_FALL_LABEL
        )
        ensemble_record = {
            "strategy": "mean_ensemble",
            "scene_id": scene_id,
            "camera": camera,
            "source": row.get("source", "sample"),
            "true": true_label,
            "predicted": ensemble_predicted,
            "fall_probability": ensemble_probability,
        }
        ensemble_records.append(ensemble_record)
        csv_rows.append(ensemble_record)
        print(f"[predict {index}/{len(available_rows)}] {scene_id}", flush=True)

    per_model: dict[str, Any] = {}
    for model_name, records in model_records.items():
        model_camera = model_cameras[model_name]
        matching_records = [
            row for row in records if row["camera"] == model_camera
        ]
        per_model[model_name] = {
            "camera": model_camera,
            "all_rows": _summarize_records(records),
            "matching_camera_rows": _summarize_records(matching_records),
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "feature_cache": str(args.feature_cache),
        "model_dir": str(args.model_dir),
        "models": [str(path) for path in model_paths],
        "fall_threshold": args.fall_threshold,
        "max_frames": args.max_frames,
        "sequence_transform": args.sequence_transform,
        "labels": {"fall": FALL_LABEL, "non_fall": NON_FALL_LABEL},
        "manifest_rows": len(rows),
        "rows": len(available_rows),
        "missing_videos": missing_videos,
        "per_model": per_model,
        "camera_matched": _summarize_records(camera_matched_records),
        "mean_ensemble": _summarize_records(ensemble_records),
        "feature_quality": feature_quality,
        "predictions": {
            "camera_matched": camera_matched_records,
            "mean_ensemble": ensemble_records,
        },
    }

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    for model_name, result in per_model.items():
        summary = result["all_rows"]
        print(
            f"{model_name}: precision={summary['precision']:.3f} "
            f"recall={summary['recall']:.3f} FP={summary['fp']} FN={summary['fn']}"
        )
    for strategy in ("camera_matched", "mean_ensemble"):
        summary = payload[strategy]
        print(
            f"{strategy}: precision={summary['precision']:.3f} "
            f"recall={summary['recall']:.3f} FP={summary['fp']} FN={summary['fn']}"
        )
    print(f"results_json: {args.results_json}")
    print(f"results_csv: {args.results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
