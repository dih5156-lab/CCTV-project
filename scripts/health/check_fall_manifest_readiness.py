#!/usr/bin/env python3
"""Check fall video manifest readiness before falldata RF training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FALL_LABEL = "fall"
NON_FALL_LABEL = "non_fall"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _scene_base(row: dict[str, Any]) -> str:
    scene_id = str(row.get("scene_id") or Path(str(row.get("video_path", ""))).stem)
    parts = scene_id.rsplit("_C", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return scene_id


def _class_name(row: dict[str, Any]) -> str:
    return FALL_LABEL if bool(row.get("is_fall")) else NON_FALL_LABEL


def build_summary(rows: list[dict[str, Any]], *, min_class_groups: int) -> dict[str, Any]:
    class_counts = {FALL_LABEL: 0, NON_FALL_LABEL: 0}
    group_sets = {FALL_LABEL: set(), NON_FALL_LABEL: set()}
    missing_videos: list[str] = []
    for row in rows:
        class_name = _class_name(row)
        class_counts[class_name] += 1
        group_sets[class_name].add(_scene_base(row))
        video_path = row.get("video_path")
        if video_path and not Path(str(video_path)).exists():
            missing_videos.append(str(video_path))

    group_counts = {
        class_name: len(groups)
        for class_name, groups in group_sets.items()
    }
    checks = [
        {
            "name": "rows",
            "actual": len(rows),
            "expected": "> 0",
            "passed": len(rows) > 0,
        },
        {
            "name": "missing_videos",
            "actual": len(missing_videos),
            "expected": 0,
            "passed": not missing_videos,
        },
        {
            "name": "fall_group_count",
            "actual": group_counts[FALL_LABEL],
            "expected": f">= {min_class_groups}",
            "passed": group_counts[FALL_LABEL] >= min_class_groups,
        },
        {
            "name": "non_fall_group_count",
            "actual": group_counts[NON_FALL_LABEL],
            "expected": f">= {min_class_groups}",
            "passed": group_counts[NON_FALL_LABEL] >= min_class_groups,
        },
    ]
    needed = {
        class_name: max(0, min_class_groups - group_counts[class_name])
        for class_name in (FALL_LABEL, NON_FALL_LABEL)
    }
    return {
        "passed": all(check["passed"] for check in checks),
        "rows": len(rows),
        "class_counts": class_counts,
        "group_counts": group_counts,
        "needed_group_counts": needed,
        "groups": {
            class_name: sorted(groups)
            for class_name, groups in group_sets.items()
        },
        "missing_videos": missing_videos,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/fall_eval/sample_manifest.jsonl"))
    parser.add_argument("--min-class-groups", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = _read_jsonl(args.manifest)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    payload = build_summary(rows, min_class_groups=args.min_class_groups)
    payload["manifest"] = str(args.manifest)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
