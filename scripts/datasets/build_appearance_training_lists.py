#!/usr/bin/env python3
"""Build train/val multi-label lists from an appearance manifest CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

COLORS = (
    "black", "white", "gray", "red", "blue", "green", "yellow", "brown", "purple",
    "navy", "orange",
)
LABELS = (
    {"index": 0, "field": "gender", "value": "female", "threshold": 0.5},
    *(
        {"index": 1 + index, "field": "upper_color", "value": color, "threshold": 0.5}
        for index, color in enumerate(COLORS)
    ),
    *(
        {"index": 1 + len(COLORS) + index, "field": "lower_color", "value": color, "threshold": 0.5}
        for index, color in enumerate(COLORS)
    ),
    {"index": 1 + len(COLORS) * 2, "field": "has_bag", "value": True, "threshold": 0.5},
    {"index": 2 + len(COLORS) * 2, "field": "has_hat", "value": True, "threshold": 0.5},
)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader if row.get("image_path")]


def _split_rows(
    rows: list[dict[str, str]], *, val_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    explicit_splits = {row.get("split", "").strip().lower() for row in rows}
    if explicit_splits and explicit_splits <= {"train", "validation", "val"} and "train" in explicit_splits:
        train_rows = [row for row in rows if row.get("split", "").strip().lower() == "train"]
        val_rows = [
            row
            for row in rows
            if row.get("split", "").strip().lower() in {"validation", "val"}
        ]
        return train_rows, val_rows

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * max(0.0, min(0.9, val_ratio))))
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    if not train_rows:
        train_rows, val_rows = val_rows, []
    return train_rows, val_rows


def _vector_for_row(row: dict[str, str]) -> list[int]:
    vector = [0] * len(LABELS)
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


def _write_list(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            vector = " ".join(str(value) for value in _vector_for_row(row))
            handle.write(f"{row['image_path']} {vector}\n")


def _write_label_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"labels": list(LABELS)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    counts = {
        "rows": len(rows),
        "gender_female": 0,
        "gender_male": 0,
        "gender_unknown": 0,
        "bag_yes": 0,
        "hat_yes": 0,
    }
    upper_colors = {color: 0 for color in COLORS}
    lower_colors = {color: 0 for color in COLORS}
    for row in rows:
        gender = row.get("gender", "unknown").strip().lower()
        if gender == "female":
            counts["gender_female"] += 1
        elif gender == "male":
            counts["gender_male"] += 1
        else:
            counts["gender_unknown"] += 1
        upper_color = row.get("upper_color", "").strip().lower()
        lower_color = row.get("lower_color", "").strip().lower()
        if upper_color in upper_colors:
            upper_colors[upper_color] += 1
        if lower_color in lower_colors:
            lower_colors[lower_color] += 1
        if row.get("bag", "").strip().lower() == "yes":
            counts["bag_yes"] += 1
        if row.get("hat", "").strip().lower() == "yes":
            counts["hat_yes"] += 1
    return {
        **counts,
        "upper_colors": upper_colors,
        "lower_colors": lower_colors,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create train/val list files from appearance_manifest.csv.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Canonical appearance manifest CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for train/val outputs.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    rows = _read_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"no rows found in {args.manifest}")

    train_rows, val_rows = _split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)

    _write_list(args.output_dir / "train_list.txt", train_rows)
    _write_list(args.output_dir / "val_list.txt", val_rows)
    _write_label_map(args.output_dir / "appearance_label_map.json")
    summary = {
        "manifest": str(args.manifest),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "labels": len(LABELS),
        "summary": _summarize(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir} train={len(train_rows)} val={len(val_rows)} labels={len(LABELS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
