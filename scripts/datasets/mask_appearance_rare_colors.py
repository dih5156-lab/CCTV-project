#!/usr/bin/env python3
"""Create a masked appearance manifest without changing the source labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ALL_COLORS = (
    "black", "white", "gray", "red", "blue", "green", "yellow", "brown",
    "purple", "navy", "orange",
)
DEFAULT_ALLOWED_LOWER = ("black", "blue", "navy", "gray", "white", "brown")


def mask_manifest(input_path: Path, output_dir: Path, allowed_lower: set[str]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest.jsonl"
    masked_counts: Counter[str] = Counter()
    row_count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            row_count += 1
            lower_color = str(row.get("lower_color") or "").strip().lower()
            if row.get("lower_color_defined", True) and lower_color not in allowed_lower:
                if lower_color in ALL_COLORS:
                    masked_counts[lower_color] += 1
                row["lower_color"] = ""
                row["lower_color_defined"] = False
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Keep the link relative so it remains valid inside the /app bind mount.
    source_images = Path("..") / "images"
    target_images = output_dir / "images"
    if not target_images.exists():
        target_images.symlink_to(source_images, target_is_directory=True)
    report = {
        "source_manifest": str(input_path),
        "output_manifest": str(output_path),
        "rows": row_count,
        "allowed_lower_colors": sorted(allowed_lower),
        "masked_lower_colors": dict(sorted(masked_counts.items())),
        "masked_rows": sum(masked_counts.values()),
    }
    (output_dir / "mask_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-lower", default=",".join(DEFAULT_ALLOWED_LOWER))
    args = parser.parse_args()
    allowed = {value.strip().lower() for value in args.allowed_lower.split(",") if value.strip()}
    invalid = sorted(allowed - set(ALL_COLORS))
    if invalid:
        parser.error(f"unknown lower colors: {', '.join(invalid)}")
    print(json.dumps(mask_manifest(args.manifest, args.output_dir, allowed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
