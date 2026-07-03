#!/usr/bin/env python3
"""Merge reviewed fall-shadow clips into a training manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_LOG = PROJECT_ROOT / "data/logs/fall_shadow_review.jsonl"
DEFAULT_BASE_MANIFEST = PROJECT_ROOT / "data/fall_eval/sample_manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/fall_eval/field_combined_manifest.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _local_clip_path(raw_path: str, clip_dir: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path.resolve()
    return (clip_dir / path.name).resolve()


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _record_date(row: dict[str, Any]) -> str:
    created_at = str(row.get("created_at") or "")
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime(
            "%Y%m%d"
        )
    except ValueError:
        event_id = str(row.get("event_id") or "")
        for part in event_id.split("_"):
            if len(part) >= 8 and part[:8].isdigit():
                return part[:8]
    return "unknown_date"


def build_field_rows(
    review_rows: list[dict[str, Any]], *, clip_dir: Path
) -> list[dict[str, Any]]:
    grouped_indexes: defaultdict[str, int] = defaultdict(int)
    field_rows: list[dict[str, Any]] = []
    for row in sorted(review_rows, key=lambda item: str(item.get("created_at") or "")):
        label = row.get("label")
        if label not in {"fall", "non_fall"} or row.get("review_status") != "reviewed":
            continue
        raw_clip_path = row.get("clip_path")
        if not raw_clip_path:
            continue
        clip_path = _local_clip_path(str(raw_clip_path), clip_dir)
        if not clip_path.is_file():
            continue
        camera_id = str(row.get("camera_id") or "camera")
        group = f"field_{camera_id}_{_record_date(row)}"
        grouped_indexes[group] += 1
        scene_id = f"{group}_C{grouped_indexes[group]}"
        field_rows.append(
            {
                "camera": camera_id,
                "event_id": row.get("event_id"),
                "fall_end_frame": 0,
                "fall_start_frame": 0,
                "is_fall": label == "fall",
                "label": "fall" if label == "fall" else "not_fall",
                "scene_category": "field_review",
                "scene_id": scene_id,
                "source": "fall_shadow_review",
                "video_exists": True,
                "video_path": _relative_to_project(clip_path),
            }
        )
    return field_rows


def limit_rows_per_scene_group(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Select evenly spaced clips per date group to reduce near-duplicates."""
    if limit <= 0:
        return rows
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_id = str(row["scene_id"])
        group = scene_id.rsplit("_C", 1)[0]
        groups[group].append(row)

    selected: list[dict[str, Any]] = []
    for group in sorted(groups):
        group_rows = groups[group]
        if len(group_rows) <= limit:
            selected.extend(group_rows)
            continue
        if limit == 1:
            selected.append(group_rows[len(group_rows) // 2])
            continue
        indexes = [round(index * (len(group_rows) - 1) / (limit - 1)) for index in range(limit)]
        selected.extend(group_rows[index] for index in indexes)
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-log", type=Path, default=DEFAULT_REVIEW_LOG)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--clip-dir", type=Path, default=PROJECT_ROOT / "data/fall_review_clips")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-field-per-group",
        type=int,
        default=0,
        help="Evenly sample at most N field clips per camera/date group (0=all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_rows = read_jsonl(args.base_manifest)
    all_field_rows = build_field_rows(read_jsonl(args.review_log), clip_dir=args.clip_dir)
    field_rows = limit_rows_per_scene_group(all_field_rows, args.max_field_per_group)
    combined_rows = base_rows + field_rows
    write_jsonl(args.output, combined_rows)
    counts = Counter("fall" if row.get("is_fall") else "non_fall" for row in combined_rows)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "base_rows": len(base_rows),
                "field_rows": len(field_rows),
                "available_field_rows": len(all_field_rows),
                "total_rows": len(combined_rows),
                "class_counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
