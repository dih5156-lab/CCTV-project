#!/usr/bin/env python3
"""DeepStream 테스트 영상 결과의 운영 품질 게이트를 계산한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def evaluate_results(rows: list[dict[str, Any]], *, min_precision: float, min_recall: float) -> dict[str, Any]:
    counts = Counter(str(row.get("result") or "NO_RESULT") for row in rows)
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "total": len(rows),
        "counts": dict(counts),
        "precision": precision,
        "recall": recall,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "passed": precision >= min_precision and recall >= min_recall and counts["NO_RESULT"] == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/experiments/fall_replay_quality_gate.json"))
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.80)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = evaluate_results(rows, min_precision=args.min_precision, min_recall=args.min_recall)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
