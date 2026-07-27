"""Evaluate a fixed fall/non-fall JSONL result set against release thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_results(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no evaluation rows")
    return rows


def evaluate(fall_path: Path, nonfall_path: Path, min_recall: float, min_precision: float) -> dict[str, object]:
    rows = _load_results(fall_path) + _load_results(nonfall_path)
    expected = [bool(row.get("expected_fall")) for row in rows]
    detected = [bool(row.get("detected")) for row in rows]
    tp = sum(actual and predicted for actual, predicted in zip(expected, detected))
    fn = sum(actual and not predicted for actual, predicted in zip(expected, detected))
    fp = sum(not actual and predicted for actual, predicted in zip(expected, detected))
    tn = sum(not actual and not predicted for actual, predicted in zip(expected, detected))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    passed = recall >= min_recall and precision >= min_precision
    return {
        "passed": passed,
        "thresholds": {"min_recall": min_recall, "min_precision": min_precision},
        "counts": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "total": len(rows)},
        "metrics": {"recall": recall, "precision": precision},
        "inputs": [str(fall_path), str(nonfall_path)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fall", type=Path, required=True)
    parser.add_argument("--nonfall", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--min-precision", type=float, default=0.70)
    args = parser.parse_args()
    try:
        result = evaluate(args.fall, args.nonfall, args.min_recall, args.min_precision)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
