#!/usr/bin/env python3
"""Create a deterministic class-balanced subset from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {"fall": [], "not_fall": []}
    for row in rows:
        label = str(row.get("label", ""))
        if label in groups:
            groups[label].append(row)
    rng = random.Random(args.seed)
    selected: list[dict] = []
    for label, candidates in groups.items():
        if len(candidates) < args.per_class:
            raise SystemExit(f"{label}: {len(candidates)} rows, need {args.per_class}")
        selected.extend(rng.sample(candidates, args.per_class))
    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(f"wrote {args.output} rows={len(selected)} per_class={args.per_class}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
