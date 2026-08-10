#!/usr/bin/env python3
"""Build a training manifest with extra weight for temporal holdout errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=4)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    hard_ids = {
        str(item["scene_id"])
        for key in ("false_negatives", "false_positives")
        for item in metrics.get("holdout_errors", {}).get(key, [])
        if item.get("scene_id")
    }
    hard_rows = [row for row in rows if str(row.get("scene_id")) in hard_ids]
    if not hard_rows:
        raise SystemExit("no holdout hard cases matched the training manifest")
    augmented = hard_rows * max(1, args.repeat)
    # Keep hard cases first so the training selector cannot truncate them.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in augmented + rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"hard_cases={len(hard_rows)} repeat={args.repeat} output_rows={len(augmented) + len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
