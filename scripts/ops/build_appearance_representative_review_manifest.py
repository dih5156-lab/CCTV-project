#!/usr/bin/env python3
"""Build a stratified human-review set from the canonical appearance manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def build(manifest: Path, image_root: Path, output: Path, limit: int, seed: int) -> int:
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        image = image_root / row["image_path"]
        if not image.is_file():
            continue
        upper = row.get("upper_color") if row.get("upper_color_defined") else "exclude"
        lower = row.get("lower_color") if row.get("lower_color_defined") else "exclude"
        camera = str(row.get("camera", "unknown"))
        groups[(str(upper), str(lower), camera)].append(row)

    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)

    # Round-robin across color/camera strata prevents black/black from dominating.
    selected: list[dict] = []
    buckets = list(groups.values())
    while len(selected) < limit and buckets:
        next_buckets = []
        for bucket in buckets:
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
            if bucket:
                next_buckets.append(bucket)
        buckets = next_buckets

    items = []
    for item_id, row in enumerate(selected, start=1):
        upper = row.get("upper_color") if row.get("upper_color_defined") else "exclude"
        lower = row.get("lower_color") if row.get("lower_color_defined") else "exclude"
        items.append(
            {
                "id": item_id,
                "crop_path": str((image_root / row["image_path"]).resolve()),
                "stored": {"upper_color": upper, "lower_color": lower},
                "candidates": {
                    "upper_color": {"hsv_color": "", "lab_color": "", "model_color": "", "model_confidence": ""},
                    "lower_color": {"hsv_color": "", "lab_color": "", "model_color": "", "model_confidence": ""},
                },
                "source": {"camera": row.get("camera"), "split": row.get("split"), "image_path": row["image_path"]},
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": 1, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output} items={len(items)} strata={len(groups)}")
    return len(items)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    build(args.manifest, args.image_root, args.output, args.limit, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
