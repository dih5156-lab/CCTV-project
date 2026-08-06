#!/usr/bin/env python3
"""품질 게이트를 통과한 낙상 모델만 운영 경로로 승격한다.

기본은 dry-run이다. 실제 파일 교체는 명시적으로 ``--approve``를 지정해야 한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def promotion_plan(comparison: dict[str, Any], candidate: Path, target: Path, *, approve: bool) -> dict[str, Any]:
    passed = bool(comparison.get("promote_candidate"))
    plan = {
        "passed": passed,
        "approved": bool(approve),
        "candidate": str(candidate),
        "target": str(target),
        "action": "promote" if passed and approve else "keep_current",
    }
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("models/experiments/fall_model_promotion.json"))
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    plan = promotion_plan(comparison, args.candidate_model, args.target_model, approve=args.approve)
    if plan["action"] == "promote":
        if not args.candidate_model.exists():
            raise SystemExit("candidate model does not exist")
        args.target_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate_model, args.target_model)
        plan["copied"] = True
    else:
        plan["copied"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0 if plan["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
