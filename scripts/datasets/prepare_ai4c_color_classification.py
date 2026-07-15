#!/usr/bin/env python3
"""Prepare a license-filtered YOLO classification dataset from AI4C colors."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

COLORS = (
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
)

COLOR_ALIASES = {"grey": "gray"}

ALLOWED_LICENSES = (
    "creativecommons.org/licenses/by/4.0",
    "creativecommons.org/publicdomain/mark/1.0",
    "creativecommons.org/publicdomain/zero/1.0",
)


def _is_allowed_license(value: object) -> bool:
    license_name = str(value or "")
    return any(allowed in license_name for allowed in ALLOWED_LICENSES)


def _find_images(image_roots: list[Path]) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for root in image_roots:
        for path in root.rglob("*"):
            if path.is_file():
                images.setdefault(path.name, path)
    return images


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_dataset(
    annotation_path: Path,
    image_roots: list[Path],
    output_dir: Path,
    *,
    val_ratio: float,
    seed: int,
) -> dict[str, object]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        item["id"]: COLOR_ALIASES.get(item["label"].lower(), item["label"].lower())
        for item in payload["categories"]
    }
    image_metadata = {item["id"]: item for item in payload["images"]}
    colors_by_image: dict[str, set[str]] = defaultdict(set)
    for annotation in payload["annotations"]:
        color = categories.get(annotation.get("category_id"), "")
        if color in COLORS:
            colors_by_image[str(annotation["image_id"])].add(color)

    available_images = _find_images(image_roots)
    eligible: list[tuple[Path, str]] = []
    skipped = Counter()
    for image_id, colors in colors_by_image.items():
        metadata = image_metadata.get(image_id)
        if not metadata or not _is_allowed_license(metadata.get("license")):
            skipped["license"] += 1
            continue
        if len(colors) != 1:
            skipped["ambiguous_color"] += 1
            continue
        source = available_images.get(Path(str(metadata["file_name"])).name)
        if source is None:
            skipped["missing_image"] += 1
            continue
        eligible.append((source, next(iter(colors))))

    random.Random(seed).shuffle(eligible)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for source, color in eligible:
        grouped[color].append(source)

    split_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }
    bounded_val_ratio = max(0.0, min(0.9, val_ratio))
    for color in COLORS:
        sources = grouped.get(color, [])
        if not sources:
            continue
        val_count = max(1, round(len(sources) * bounded_val_ratio)) if len(sources) > 1 else 0
        for index, source in enumerate(sources):
            split = "val" if index < val_count else "train"
            destination = output_dir / split / color / source.name
            if destination.exists():
                skipped["duplicate_destination"] += 1
                continue
            _link_or_copy(source, destination)
            split_counts[split][color] += 1

    summary: dict[str, object] = {
        "annotation_path": str(annotation_path),
        "image_roots": [str(path) for path in image_roots],
        "output_dir": str(output_dir),
        "colors": list(COLORS),
        "allowed_licenses": list(ALLOWED_LICENSES),
        "seed": seed,
        "val_ratio": bounded_val_ratio,
        "train": dict(split_counts["train"]),
        "val": dict(split_counts["val"]),
        "train_total": sum(split_counts["train"].values()),
        "val_total": sum(split_counts["val"].values()),
        "skipped": dict(skipped),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a YOLO classification dataset from AI4C Fashion Color.",
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = prepare_dataset(
        args.annotations,
        args.images,
        args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        f"prepared {args.output_dir} "
        f"train={summary['train_total']} val={summary['val_total']} "
        f"skipped={summary['skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
