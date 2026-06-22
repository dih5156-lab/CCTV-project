#!/usr/bin/env python3
"""Smoke-test a public fall-data video RandomForest model interface.

This does not validate accuracy. It only confirms that the model can be loaded
and called with the expected flattened 600 x 1662 feature vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(
    "falldata/2. AI학습모델파일/영상/낙상분류/FNF_RF_SMOTE_CAM_1.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a falldata video RandomForest .pkl model.",
    )
    parser.add_argument(
        "--fill",
        type=float,
        default=0.0,
        help="Constant value for the synthetic input feature vector.",
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing 600 MediaPipe feature .npy files. "
            "When provided, this is used instead of synthetic input."
        ),
    )
    return parser.parse_args()


def _load_sequence(sequence_dir: Path, expected_features: int) -> np.ndarray:
    frame_files = sorted(
        [path for path in sequence_dir.glob("*.npy")],
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if len(frame_files) != 600:
        raise ValueError(f"expected 600 npy frames, found {len(frame_files)}: {sequence_dir}")

    frames = [np.load(path).reshape(-1) for path in frame_files]
    sequence = np.asarray(frames, dtype=np.float32)
    if sequence.shape != (600, 1662):
        raise ValueError(f"expected sequence shape (600, 1662), got {sequence.shape}")

    flattened = sequence.reshape(1, -1)
    if flattened.shape[1] != expected_features:
        raise ValueError(
            f"model expects {expected_features} features, sequence produced {flattened.shape[1]}"
        )
    return flattened


def main() -> int:
    import joblib
    import sklearn

    args = parse_args()
    model = joblib.load(args.model)
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is None:
        raise RuntimeError(f"model does not expose n_features_in_: {args.model}")

    if args.sequence_dir is None:
        sample = np.full((1, feature_count), args.fill, dtype=np.float32)
        input_source = f"synthetic fill={args.fill}"
    else:
        sample = _load_sequence(args.sequence_dir, feature_count)
        input_source = str(args.sequence_dir)

    prediction = model.predict(sample)
    probability = (
        model.predict_proba(sample).tolist()
        if hasattr(model, "predict_proba")
        else None
    )

    print(f"sklearn: {sklearn.__version__}")
    print(f"model: {args.model}")
    print(f"model_type: {type(model).__module__}.{type(model).__name__}")
    print(f"input_source: {input_source}")
    print(f"input_shape: {sample.shape}")
    print(f"classes: {getattr(model, 'classes_', []).tolist()}")
    print(f"prediction: {prediction.tolist()}")
    if probability is not None:
        print(f"predict_proba: {probability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
