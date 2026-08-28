#!/usr/bin/env python3
"""Import reviewed appearance color labels into a separate classification set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ALLOWED = {"black", "blue", "gray", "brown", "green", "navy", "orange", "pink", "purple", "red", "white", "yellow"}


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    by_id = {int(item["id"]): item.get("review_label") for item in labels.get("items", [])}
    copied = 0
    skipped = 0
    for item in manifest.get("items", []):
        label = by_id.get(int(item["id"]))
        source = Path(item.get("crop_path", ""))
        if label not in ALLOWED or not source.exists():
            skipped += 1
            continue
        destination = args.output_dir / args.split / label / f"appearance_{item['id']}_{source.name}"
        link_or_copy(source, destination)
        copied += 1

    summary = {
        "manifest": str(args.manifest),
        "labels": str(args.labels),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "copied": copied,
        "skipped": skipped,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"import_{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
