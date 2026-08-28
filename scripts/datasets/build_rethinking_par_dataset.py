#!/usr/bin/env python3
"""Build a Rethinking_of_PAR compatible dataset_all.pkl from a manifest CSV."""

from __future__ import annotations

import argparse
import csv
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.datasets.build_appearance_training_lists import COLORS

ATTRIBUTES = (
    "gender_female",
    *(f"upper_{color}" for color in COLORS),
    *(f"lower_{color}" for color in COLORS),
    "has_bag",
    "has_hat",
)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader if row.get("image_path")]


def _label_vector(row: dict[str, str]) -> list[int]:
    vector = [0] * len(ATTRIBUTES)
    if row.get("gender", "").strip().lower() == "female":
        vector[0] = 1

    upper_color = row.get("upper_color", "").strip().lower()
    if upper_color in COLORS:
        vector[1 + COLORS.index(upper_color)] = 1

    lower_color = row.get("lower_color", "").strip().lower()
    if lower_color in COLORS:
        vector[1 + len(COLORS) + COLORS.index(lower_color)] = 1

    if row.get("bag", "").strip().lower() == "yes":
        vector[1 + len(COLORS) * 2] = 1
    if row.get("hat", "").strip().lower() == "yes":
        vector[2 + len(COLORS) * 2] = 1
    return vector


def _image_name(path_value: str, *, image_root: Path | None) -> str:
    path = Path(path_value)
    if image_root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(image_root.resolve()))
    except ValueError:
        return str(path)


def build_dataset(
    rows: list[dict[str, str]],
    *,
    image_root: Path | None,
    val_ratio: float,
    seed: int,
) -> SimpleNamespace:
    labels = np.asarray([_label_vector(row) for row in rows], dtype=np.int64)
    explicit_splits = {row.get("split", "").strip().lower() for row in rows}
    if explicit_splits and explicit_splits <= {"train", "validation", "val"} and "train" in explicit_splits:
        trainval = np.asarray(
            [index for index, row in enumerate(rows) if row.get("split", "").strip().lower() == "train"],
            dtype=np.int64,
        )
        test = np.asarray(
            [index for index, row in enumerate(rows) if row.get("split", "").strip().lower() in {"validation", "val"}],
            dtype=np.int64,
        )
    else:
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        test_count = max(1, round(len(indices) * max(0.0, min(0.9, val_ratio))))
        test = np.asarray(indices[:test_count], dtype=np.int64)
        trainval = np.asarray(indices[test_count:], dtype=np.int64)
        if trainval.size == 0:
            trainval = test
            test = np.asarray([], dtype=np.int64)

    dataset = SimpleNamespace()
    dataset.description = "cctv_appearance_manifest_for_rethinking_par"
    dataset.root = str(image_root.resolve()) if image_root is not None else "."
    dataset.image_name = [
        _image_name(row["image_path"], image_root=image_root)
        for row in rows
    ]
    dataset.label = labels
    dataset.attr_name = list(ATTRIBUTES)
    dataset.label_idx = SimpleNamespace()
    dataset.label_idx.eval = list(range(len(ATTRIBUTES)))
    dataset.label_idx.color = list(range(1, 1 + len(COLORS) * 2))
    dataset.label_idx.extra = [0, 1 + len(COLORS) * 2, 2 + len(COLORS) * 2]
    dataset.partition = SimpleNamespace()
    dataset.partition.trainval = [trainval]
    dataset.partition.test = [test]
    dataset.partition.train = [trainval]
    dataset.partition.val = [test]
    dataset.weight_trainval = [labels[trainval].mean(axis=0).astype(np.float32)]
    dataset.weight_train = dataset.weight_trainval
    return dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create dataset_all.pkl for valencebond/Rethinking_of_PAR.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Canonical appearance manifest CSV.")
    parser.add_argument("--output-pkl", type=Path, required=True, help="Output dataset_all.pkl path.")
    parser.add_argument("--image-root", type=Path, help="Image root used by Rethinking_of_PAR dataset loader.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Test split ratio for the generated pkl.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    rows = _read_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"no rows found in {args.manifest}")
    dataset = build_dataset(
        rows,
        image_root=args.image_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as handle:
        pickle.dump(dataset, handle)
    print(
        f"wrote {args.output_pkl} rows={len(rows)} "
        f"attrs={len(dataset.attr_name)} trainval={len(dataset.partition.trainval[0])} "
        f"test={len(dataset.partition.test[0])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
