#!/usr/bin/env python3
"""Probe public fall-data model files without running full inference."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

MODEL_SUFFIXES = {".pkl", ".sav", ".h5"}


def _model_info(model: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": f"{type(model).__module__}.{type(model).__name__}",
    }
    for attr in ("n_features_in_", "n_classes_", "classes_"):
        if hasattr(model, attr):
            value = getattr(model, attr)
            if hasattr(value, "tolist"):
                value = value.tolist()
            info[attr.rstrip("_")] = value
    if hasattr(model, "estimators_"):
        info["estimators"] = len(getattr(model, "estimators_", []))
    return info


def _probe_joblib_model(path: Path) -> dict[str, Any]:
    import joblib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(path)

    info = _model_info(model)
    if caught:
        info["warnings"] = [str(item.message) for item in caught[:5]]
    return info


def _probe_h5_model(path: Path) -> dict[str, Any]:
    import tensorflow as tf

    model = tf.keras.models.load_model(path, compile=False)
    return {
        "type": f"{type(model).__module__}.{type(model).__name__}",
        "input_shape": getattr(model, "input_shape", None),
        "output_shape": getattr(model, "output_shape", None),
    }


def _probe_one(path: Path, include_h5: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
    }
    try:
        if path.suffix.lower() in {".pkl", ".sav"}:
            result.update({"status": "loaded", **_probe_joblib_model(path)})
        elif path.suffix.lower() == ".h5":
            if not include_h5:
                result["status"] = "skipped_h5"
            else:
                result.update({"status": "loaded", **_probe_h5_model(path)})
        else:
            result["status"] = "skipped_unknown_suffix"
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    feature_counts: dict[str, int] = {}
    for item in results:
        status = item.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        if item.get("status") == "loaded" and "n_features_in" in item:
            key = str(item["n_features_in"])
            feature_counts[key] = feature_counts.get(key, 0) + 1
    return {
        "total": len(results),
        "by_status": by_status,
        "feature_counts": feature_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("falldata/2. AI학습모델파일"),
        help="Directory containing public fall model files.",
    )
    parser.add_argument(
        "--include-h5",
        action="store_true",
        help="Try to load Keras .h5 models. This requires TensorFlow.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a compact text report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_files = sorted(
        path
        for path in args.root.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )
    results = [_probe_one(path, include_h5=args.include_h5) for path in model_files]
    summary = _summarize(results)

    if args.json:
        print(json.dumps({"summary": summary, "models": results}, ensure_ascii=False, indent=2))
        return 0

    print(f"model_root: {args.root.resolve()}")
    print(f"summary: {summary}")
    for item in results:
        rel_path = Path(item["path"]).relative_to(args.root)
        pieces = [str(rel_path), f"status={item['status']}"]
        if "type" in item:
            pieces.append(f"type={item['type'].rsplit('.', 1)[-1]}")
        if "n_features_in" in item:
            pieces.append(f"features={item['n_features_in']}")
        if "classes" in item:
            pieces.append(f"classes={item['classes']}")
        if item.get("status") == "failed":
            pieces.append(f"error={item.get('error_type')}: {item.get('error')}")
        print("  " + " | ".join(pieces))

    print()
    print("integration_hint:")
    print("  video RF models expect 997200 features, likely 600 x 1662-frame features.")
    print("  sensor ML models expect 2484 engineered spatio-temporal features.")
    print("  current YOLO pose keypoints are not shape-compatible without an adapter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
