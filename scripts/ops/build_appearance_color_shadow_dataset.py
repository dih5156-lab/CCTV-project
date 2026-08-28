#!/usr/bin/env python3
"""Add human-reviewed shadow ROIs to a YOLO classification dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_base_dataset(base_dir: Path, output_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for split in ("train", "val"):
        for source in sorted((base_dir / split).glob("*/*")):
            if not source.is_file():
                continue
            destination = output_dir / split / source.parent.name / source.name
            _link_or_copy(source, destination)
            counts[split] += 1
    return counts


def build_dataset(
    *,
    base_dir: Path,
    comparison_path: Path,
    labels_path: Path,
    output_dir: Path,
    validation_groups: set[str],
    review_repeat: int = 1,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    if review_repeat < 1:
        raise ValueError("review_repeat must be at least 1")

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    comparison_by_id = {
        int(item["id"]): item for item in comparison.get("items", [])
    }
    class_names = {
        path.name for path in (base_dir / "train").iterdir() if path.is_dir()
    }
    base_counts = _copy_base_dataset(base_dir, output_dir)

    added_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    archived_labels: list[dict[str, Any]] = []
    for label in labels.get("items", []):
        item_id = int(label["id"])
        color = label.get("upper_color")
        if color in (None, "exclude"):
            skipped_counts[str(color or "unreviewed")] += 1
            continue
        if color not in class_names:
            raise ValueError(f"unsupported color for id {item_id}: {color}")
        comparison_item = comparison_by_id.get(item_id)
        if comparison_item is None:
            raise ValueError(f"comparison id not found: {item_id}")

        group = (
            f"{comparison_item['camera_id']}:{comparison_item['track_id']}"
        )
        split = "val" if group in validation_groups else "train"
        roi_path = comparison_path.parent / comparison_item["roi_path"]
        if not roi_path.is_file():
            raise ValueError(f"ROI not found for id {item_id}: {roi_path}")
        repetitions = review_repeat if split == "train" else 1
        for repeat_index in range(repetitions):
            suffix = "" if repeat_index == 0 else f"_r{repeat_index + 1}"
            destination = (
                output_dir
                / split
                / color
                / f"human_upper_{item_id}{suffix}.jpg"
            )
            _link_or_copy(roi_path, destination)
            added_counts[f"{split}:{color}"] += 1
        group_counts[f"{split}:{group}:{color}"] += 1
        archived_labels.append(
            {
                "id": item_id,
                "camera_id": comparison_item["camera_id"],
                "track_id": comparison_item["track_id"],
                "split": split,
                "upper_color": color,
                "source_roi": str(roi_path),
            }
        )

    summary: dict[str, Any] = {
        "base": str(base_dir),
        "comparison": str(comparison_path),
        "labels": str(labels_path),
        "output": str(output_dir),
        "validation_groups": sorted(validation_groups),
        "review_repeat": review_repeat,
        "base_counts": dict(sorted(base_counts.items())),
        "review_added": dict(sorted(added_counts.items())),
        "review_groups": dict(sorted(group_counts.items())),
        "skipped": dict(sorted(skipped_counts.items())),
    }
    (output_dir / "human_labels.json").write_text(
        json.dumps(
            {"schema_version": 1, "items": archived_labels},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--val-group",
        action="append",
        default=[],
        help="Track group '<camera_id>:<track_id>' reserved for validation.",
    )
    parser.add_argument("--review-repeat", type=int, default=1)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = build_dataset(
        base_dir=args.base,
        comparison_path=args.comparison,
        labels_path=args.labels,
        output_dir=args.output,
        validation_groups=set(args.val_group),
        review_repeat=args.review_repeat,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
