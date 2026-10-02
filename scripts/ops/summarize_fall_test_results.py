#!/usr/bin/env python3
"""실제 낙상 테스트 manifest의 모델 결과를 집계한다."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def summarize_manifest(path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    evaluated = []
    for row in rows:
        expected = row.get("expected_fall", "").strip().lower()
        predicted = row.get("model_prediction", "").strip().lower()
        if expected not in {"0", "1", "true", "false"} or predicted not in {
            "0",
            "1",
            "true",
            "false",
        }:
            continue
        expected_bool = expected in {"1", "true"}
        predicted_bool = predicted in {"1", "true"}
        evaluated.append((row, expected_bool, predicted_bool))

    counts = Counter()
    pose_counts: Counter[str] = Counter()
    for row, expected, predicted in evaluated:
        pose = row.get("pose", "unknown") or "unknown"
        pose_counts[pose] += 1
        if expected and predicted:
            counts["tp"] += 1
        elif not expected and not predicted:
            counts["tn"] += 1
        elif not expected and predicted:
            counts["fp"] += 1
        else:
            counts["fn"] += 1

    total = len(evaluated)
    return {
        "manifest": str(path),
        "total_rows": len(rows),
        "evaluated_rows": total,
        "pending_rows": len(rows) - total,
        "confusion": {key: counts[key] for key in ("tp", "tn", "fp", "fn")},
        "accuracy": (counts["tp"] + counts["tn"]) / total if total else None,
        "pose_counts": dict(sorted(pose_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_manifest(args.manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
