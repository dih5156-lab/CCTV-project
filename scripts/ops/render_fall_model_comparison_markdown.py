#!/usr/bin/env python3
"""낙상 모델 비교 JSON을 인수인계용 Markdown 표로 변환한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render_markdown(result: dict[str, Any]) -> str:
    baseline = result.get("baseline") or {}
    candidate = result.get("candidate") or {}
    deltas = result.get("deltas") or {}
    checks = result.get("checks") or {}
    decision = "PROMOTE" if result.get("promote_candidate") else "HOLD"

    lines = [
        "# 낙상 모델 비교 결과",
        "",
        f"## 판정: `{decision}`",
        "",
        "| 지표 | Baseline | Candidate | 변화 |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("precision", "recall", "f1"):
        lines.append(
            f"| {metric} | {float(baseline.get(metric, 0.0)):.4f} | "
            f"{float(candidate.get(metric, 0.0)):.4f} | "
            f"{float(deltas.get(metric, 0.0)):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## 자동 검사",
            "",
            f"- precision 기준: `{checks.get('precision_not_lower', False)}`",
            f"- recall 기준: `{checks.get('recall_not_lower', False)}`",
            f"- F1 기준: `{checks.get('f1_not_lower', False)}`",
            f"- 데이터셋 호환성: `{checks.get('dataset_compatible', True)}`",
            "",
            f"- Baseline dataset: `{result.get('baseline_dataset_version') or 'unknown'}`",
            f"- Candidate dataset: `{result.get('candidate_dataset_version') or 'unknown'}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = json.loads(args.comparison.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid comparison JSON: {exc}")
    if not isinstance(result, dict):
        raise SystemExit("comparison JSON must be an object")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(result), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
