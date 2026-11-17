#!/usr/bin/env python3
"""Compare fall and non-fall replay diagnostics without changing the model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

NUMERIC_FIELDS = (
    "max_fall_score",
    "max_compare_fall_probability",
    "max_near_miss_score",
    "near_miss_record_count",
    "compare_model_record_count",
    "fall_candidate_count",
    "shadow_record_count",
)


def _load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_source"] = path.name
                rows.append(row)
    return rows


def _summary(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {"fall": [], "non_fall": []}
    for row in rows:
        groups["fall" if row.get("expected_fall") else "non_fall"].append(row)

    result: dict[str, object] = {"total": len(rows), "groups": {}, "camera_breakdown": {}}
    for label, group in groups.items():
        values: dict[str, object] = {"count": len(group)}
        for field in NUMERIC_FIELDS:
            numbers = [float(row[field]) for row in group if row.get(field) is not None]
            values[field] = {
                "count": len(numbers),
                "mean": mean(numbers) if numbers else None,
                "max": max(numbers) if numbers else None,
            }
        values["results"] = dict(Counter(row.get("result", "unknown") for row in group))
        values["near_miss_types"] = dict(
            Counter(
                near_miss_type
                for row in group
                for near_miss_type in row.get("near_miss_types", [])
            )
        )
        values["scenes"] = [
            {
                "scene_id": row.get("scene_id"),
                "result": row.get("result"),
                "max_fall_score": row.get("max_fall_score"),
                "max_compare_fall_probability": row.get("max_compare_fall_probability"),
                "frames": row.get("scene_length"),
                "near_miss_types": row.get("near_miss_types", []),
            }
            for row in group
        ]
        result["groups"][label] = values

    camera_groups: dict[str, list[dict]] = {}
    for row in rows:
        scene_id = str(row.get("scene_id") or "unknown")
        camera = scene_id.rsplit("_C", 1)[-1] if "_C" in scene_id else "unknown"
        camera_groups.setdefault(camera, []).append(row)
    for camera, group in sorted(camera_groups.items()):
        fall_rows = [row for row in group if row.get("expected_fall")]
        detected = sum(row.get("result") == "TP" for row in fall_rows)
        result["camera_breakdown"][camera] = {
            "count": len(group),
            "fall_count": len(fall_rows),
            "fall_detected": detected,
            "fall_recall": detected / len(fall_rows) if fall_rows else None,
            "fall_near_miss_types": dict(
                Counter(
                    near_miss_type
                    for row in fall_rows
                    for near_miss_type in row.get("near_miss_types", [])
                )
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = _load_rows(args.results)
    summary = _summary(rows)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
