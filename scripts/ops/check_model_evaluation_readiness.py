#!/usr/bin/env python3
"""모델 비교 전에 최종 산출물과 metrics 준비 상태를 읽기 전용으로 점검한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPLORATORY_MARKERS = ("partial", "smoke", "probe")


def check_readiness(model: Path, metrics: Path) -> dict[str, Any]:
    """모델과 metrics가 비교 가능한 최종 산출물인지 반환한다."""
    reasons: list[str] = []
    if not model.is_file():
        reasons.append("model file is missing")
    if not metrics.is_file():
        reasons.append("metrics file is missing")

    expected_metrics_stem = f"{model.stem}_metrics"
    if metrics.stem != expected_metrics_stem:
        reasons.append(
            f"metrics filename does not match model (expected {expected_metrics_stem}.json)"
        )

    model_name = model.name.lower()
    if any(marker in model_name for marker in EXPLORATORY_MARKERS):
        reasons.append("model is marked as exploratory (partial/smoke/probe)")

    metrics_data: dict[str, Any] | None = None
    if metrics.is_file():
        try:
            loaded = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reasons.append("metrics file is not valid JSON")
        else:
            if not isinstance(loaded, dict):
                reasons.append("metrics JSON must be an object")
            else:
                metrics_data = loaded
                if not any(
                    isinstance(loaded.get(section), dict)
                    for section in ("validation", "holdout")
                ):
                    reasons.append("metrics JSON has no validation or holdout section")

    ready = not reasons
    return {
        "ready": ready,
        "decision": "ready" if ready else "hold",
        "model": str(model),
        "metrics": str(metrics),
        "reasons": reasons,
        "metrics_keys": sorted(metrics_data) if metrics_data is not None else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = check_readiness(args.model, args.metrics)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
