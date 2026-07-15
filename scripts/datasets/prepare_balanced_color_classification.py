#!/usr/bin/env python3
"""Combine real apparel images and synthesize missing color classes for YOLO cls."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts.datasets.prepare_ai4c_color_classification import COLORS
except ModuleNotFoundError:  # 직접 실행 시 scripts/datasets가 sys.path 루트다.
    from prepare_ai4c_color_classification import COLORS


APPAREL_TYPES = {"dress", "pants", "shirt", "shorts"}
DONOR_COLORS = {"blue", "green"}

TARGET_HSV = {
    "brown": (12, 145),
    "gray": (0, 8),
    "orange": (14, 220),
    "pink": (170, 125),
    "purple": (145, 180),
    "yellow": (27, 205),
}

SOURCE_HUE_RANGES = {
    "blue": ((85, 135),),
    "green": ((35, 85),),
}


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _apparel_groups(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            color, apparel_type = directory.name.split("_", 1)
        except ValueError:
            continue
        if color not in COLORS or apparel_type not in APPAREL_TYPES:
            continue
        grouped[color].extend(sorted(path for path in directory.iterdir() if path.is_file()))
    return grouped


def _recolor_apparel(
    image: np.ndarray,
    *,
    source_color: str,
    target_color: str,
    rng: random.Random,
) -> np.ndarray | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = np.zeros(hue.shape, dtype=np.uint8)
    for lower_hue, upper_hue in SOURCE_HUE_RANGES[source_color]:
        in_range = cv2.inRange(
            hsv,
            np.array((lower_hue, 60, 25), dtype=np.uint8),
            np.array((upper_hue, 255, 255), dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, in_range)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    coverage = cv2.countNonZero(mask) / float(mask.size)
    if coverage < 0.06:
        return None

    target_hue, target_saturation = TARGET_HSV[target_color]
    selected = mask > 0
    hue[selected] = np.clip(target_hue + rng.randint(-3, 3), 0, 179)

    original_saturation = saturation[selected].astype(np.float32)
    saturation_scale = 0.80 + (original_saturation / 1275.0)
    saturation[selected] = np.clip(target_saturation * saturation_scale, 0, 255).astype(np.uint8)

    original_value = value[selected].astype(np.float32)
    if target_color == "brown":
        value[selected] = np.clip(original_value * 0.68, 35, 170).astype(np.uint8)
    elif target_color == "gray":
        saturation[selected] = rng.randint(2, 14)
        value[selected] = np.clip(original_value * rng.uniform(0.60, 0.88), 65, 195).astype(np.uint8)
    elif target_color == "orange":
        value[selected] = np.clip(original_value * 1.15, 105, 245).astype(np.uint8)
    elif target_color == "pink":
        value[selected] = np.clip(original_value * 1.08, 120, 245).astype(np.uint8)
    elif target_color == "yellow":
        value[selected] = np.clip(original_value * 1.10, 125, 245).astype(np.uint8)

    recolored = cv2.cvtColor(cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2BGR)
    feathered = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[:, :, None] / 255.0
    return np.clip(
        recolored.astype(np.float32) * feathered + image.astype(np.float32) * (1.0 - feathered),
        0,
        255,
    ).astype(np.uint8)


def _copy_ai4c_split(source_root: Path, output_root: Path, split: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    split_root = source_root / split
    if not split_root.exists():
        return counts
    for color_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        if color_dir.name not in COLORS:
            continue
        for source in sorted(path for path in color_dir.iterdir() if path.is_file()):
            destination = output_root / split / color_dir.name / f"ai4c_{source.name}"
            _link_or_copy(source, destination)
            counts[color_dir.name] += 1
    return counts


def prepare_balanced_dataset(
    ai4c_root: Path,
    apparel_root: Path,
    output_root: Path,
    *,
    target_train_per_color: int,
    apparel_val_per_color: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    train_counts = _copy_ai4c_split(ai4c_root, output_root, "train")
    val_counts = _copy_ai4c_split(ai4c_root, output_root, "val")

    apparel_groups = _apparel_groups(apparel_root)
    donor_candidates: list[tuple[Path, str]] = []
    for color, sources in sorted(apparel_groups.items()):
        shuffled = list(sources)
        rng.shuffle(shuffled)
        val_count = min(apparel_val_per_color, max(0, len(shuffled) // 5))
        val_sources = shuffled[:val_count]
        train_sources = shuffled[val_count:]
        for source in val_sources:
            _link_or_copy(source, output_root / "val" / color / f"apparel_{source.name}")
            val_counts[color] += 1
        available_slots = max(0, target_train_per_color - train_counts[color])
        for source in train_sources[:available_slots]:
            _link_or_copy(source, output_root / "train" / color / f"apparel_{source.name}")
            train_counts[color] += 1
        if color in DONOR_COLORS:
            donor_candidates.extend((source, color) for source in train_sources)

    rng.shuffle(donor_candidates)
    synthetic_counts: Counter[str] = Counter()
    donor_index = 0
    for target_color in TARGET_HSV:
        failed_attempts = 0
        while train_counts[target_color] < target_train_per_color:
            if failed_attempts >= max(100, len(donor_candidates) * 3):
                break
            source, source_color = donor_candidates[donor_index % len(donor_candidates)]
            donor_index += 1
            image = cv2.imread(str(source))
            if image is None:
                failed_attempts += 1
                continue
            recolored = _recolor_apparel(
                image,
                source_color=source_color,
                target_color=target_color,
                rng=rng,
            )
            if recolored is None:
                failed_attempts += 1
                continue
            destination = (
                output_root
                / "train"
                / target_color
                / f"synthetic_{synthetic_counts[target_color]:04d}_{source.stem}.jpg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), recolored, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"failed to write {destination}")
            train_counts[target_color] += 1
            synthetic_counts[target_color] += 1

    summary: dict[str, object] = {
        "ai4c_root": str(ai4c_root),
        "apparel_root": str(apparel_root),
        "output_root": str(output_root),
        "target_train_per_color": target_train_per_color,
        "apparel_val_per_color": apparel_val_per_color,
        "seed": seed,
        "train": dict(train_counts),
        "val": dict(val_counts),
        "synthetic_train": dict(synthetic_counts),
        "train_total": sum(train_counts.values()),
        "val_total": sum(val_counts.values()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a balanced 11-color classification dataset.")
    parser.add_argument("--ai4c-root", type=Path, required=True)
    parser.add_argument("--apparel-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-train-per-color", type=int, default=500)
    parser.add_argument("--apparel-val-per-color", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = prepare_balanced_dataset(
        args.ai4c_root,
        args.apparel_root,
        args.output_dir,
        target_train_per_color=max(1, args.target_train_per_color),
        apparel_val_per_color=max(0, args.apparel_val_per_color),
        seed=args.seed,
    )
    print(
        f"prepared {args.output_dir} train={summary['train_total']} "
        f"val={summary['val_total']} synthetic={summary['synthetic_train']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
