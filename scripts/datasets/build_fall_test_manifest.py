#!/usr/bin/env python3
"""학습/검증 manifest에서 방향·낙상 여부가 균형 잡힌 테스트 manifest를 만든다."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _direction(row: dict[str, Any]) -> str:
    value = str(row.get("scene_category") or "").lower()
    if "전면" in value:
        return "front"
    if "측면" in value:
        return "side"
    if "후면" in value:
        return "back"
    return "non_fall" if not row.get("is_fall") else "other_fall"


def _group(row: dict[str, Any]) -> str:
    return str(row.get("scene_group") or row.get("scene_id") or row.get("video_path"))


def select_test_rows(rows: list[dict[str, Any]], per_group: int = 5) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_direction(row)].append(row)
    selected: list[dict[str, Any]] = []
    for key in ("front", "side", "back", "non_fall", "other_fall"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(buckets.get(key, []), key=lambda item: str(item.get("scene_id"))):
            groups[_group(row)].append(row)
        for group_rows in list(groups.values())[: max(per_group, 0)]:
            selected.append(group_rows[0])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/fall_eval/test_manifest.jsonl"))
    parser.add_argument("--per-group", type=int, default=5)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_test_rows(rows, per_group=args.per_group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(selected), "directions": {key: sum(_direction(row) == key for row in selected) for key in ("front", "side", "back", "non_fall", "other_fall")}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
