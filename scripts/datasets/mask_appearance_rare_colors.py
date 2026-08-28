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


def mask_manifest(
    input_path: Path,
    output_dir: Path,
    allowed_lower: set[str],
    drop_missing: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest.jsonl"
    masked_counts: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    split_people: dict[str, set[str]] = {}
    missing_images: list[str] = []
    row_count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = str(row.get("image_path") or "")
            image_file = input_path.parent / Path(image_path)
            if drop_missing and not image_file.is_file():
                missing_images.append(image_path)
                continue
            row_count += 1
            split = str(row.get("split") or "unknown")
            split_rows[split] += 1
            split_people.setdefault(split, set()).add(str(row.get("person_id") or ""))
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
        "drop_missing": drop_missing,
        "missing_images": missing_images,
        "allowed_lower_colors": sorted(allowed_lower),
        "masked_lower_colors": dict(sorted(masked_counts.items())),
        "masked_rows": sum(masked_counts.values()),
    }
    split_report = {
        "malformed_xml": None,
        "missing_images": len(missing_images),
        "parsed_rows": None,
        "selected_rows": row_count,
        "splits": {
            split: {"persons": len(people), "rows": split_rows[split]}
            for split, people in sorted(split_people.items())
        },
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(split_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "mask_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-lower", default=",".join(DEFAULT_ALLOWED_LOWER))
    parser.add_argument("--drop-missing", action="store_true")
    args = parser.parse_args()
    allowed = {value.strip().lower() for value in args.allowed_lower.split(",") if value.strip()}
    invalid = sorted(allowed - set(ALL_COLORS))
    if invalid:
        parser.error(f"unknown lower colors: {', '.join(invalid)}")
    print(json.dumps(mask_manifest(args.manifest, args.output_dir, allowed, args.drop_missing), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
