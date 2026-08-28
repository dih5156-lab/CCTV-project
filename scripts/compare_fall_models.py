#!/usr/bin/env python3
"""기존/신규 낙상 모델 metrics를 비교해 운영 후보를 선정한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _fall_metrics(report: dict[str, Any]) -> dict[str, float]:
    for section in ("validation", "holdout"):
        fall = ((report.get(section) or {}).get("classification_report") or {}).get(
            "fall"
        )
        if isinstance(fall, dict):
            precision = float(fall.get("precision", 0.0))
            recall = float(fall.get("recall", 0.0))
            f1 = float(fall.get("f1-score", 0.0))
            if not f1 and precision + recall:
                f1 = 2.0 * precision * recall / (precision + recall)
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any], *, min_precision: float = 0.0
) -> dict[str, Any]:
    base = _fall_metrics(baseline)
    new = _fall_metrics(candidate)
    checks = {
        "precision_not_lower": new["precision"]
        >= max(base["precision"], min_precision),
        "recall_not_lower": new["recall"] >= base["recall"],
        "f1_not_lower": new["f1"] >= base["f1"],
    }
    candidate_wins = sum(new[key] > base[key] for key in ("precision", "recall", "f1"))
    checks["strict_improvement"] = candidate_wins >= 1
    baseline_threshold = float(
        (baseline.get("model_params") or {}).get("decision_threshold", 0.7)
    )
    candidate_threshold = float(
        (candidate.get("model_params") or {}).get("decision_threshold", 0.7)
    )
    return {
        "baseline": base,
        "candidate": new,
        "deltas": {key: new[key] - base[key] for key in base},
        "checks": checks,
        "promote_candidate": all(checks.values()),
        "baseline_decision_threshold": baseline_threshold,
        "candidate_decision_threshold": candidate_threshold,
        "decision_threshold": (
            candidate_threshold if all(checks.values()) else baseline_threshold
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models/experiments/fall_model_comparison.json",
    )
    parser.add_argument("--min-precision", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.baseline_metrics.exists() or not args.candidate_metrics.exists():
        raise SystemExit("baseline/candidate metrics file is not ready")
    result = compare_reports(
        _load(args.baseline_metrics),
        _load(args.candidate_metrics),
        min_precision=args.min_precision,
    )
    result.update(
        {
            "baseline_metrics": str(args.baseline_metrics),
            "candidate_metrics": str(args.candidate_metrics),
            "baseline_model": str(args.baseline_model) if args.baseline_model else None,
            "candidate_model": str(args.candidate_model)
            if args.candidate_model
            else None,
        }
    )
    if result["promote_candidate"] and args.candidate_model:
        result["best_candidate"] = str(args.candidate_model)
    else:
        result["best_candidate"] = (
            str(args.baseline_model) if args.baseline_model else None
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["promote_candidate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
