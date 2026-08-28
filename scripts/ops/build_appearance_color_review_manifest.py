#!/usr/bin/env python3
"""Build a human-review manifest for appearance color hard cases.

The manifest deliberately does not invent labels. It collects crops where the
classical color estimators and the learned classifier disagree so they can be
verified before being added to a fine-tuning set.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_crop(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return str(path)
    if str(path).startswith("/app/"):
        candidate = PROJECT_ROOT / str(path).removeprefix("/app/")
        if candidate.exists():
            return str(candidate)
    return None


def build_manifest(
    db_path: Path,
    limit: int,
    focus_field: str | None = None,
    *,
    min_color_observations: int = 0,
    max_per_track: int = 0,
) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, event_id, timestamp, camera_id, track_id,
                   upper_color, lower_color, attribute_backend,
                   crop_path, attribute_metadata
            FROM appearance_log
            WHERE crop_path IS NOT NULL AND crop_path != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, limit * (100 if focus_field else 5)),),
        ).fetchall()

    items: list[dict[str, Any]] = []
    selected_per_track: dict[tuple[str, int | None], int] = {}
    for row in rows:
        try:
            metadata = json.loads(row["attribute_metadata"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        candidates = metadata.get("color_candidates") or {}
        sources = metadata.get("color_sources") or {}
        observations = metadata.get("color_observations") or {}
        if focus_field:
            source = sources.get(focus_field)
            try:
                observation_count = int(observations.get(focus_field) or 0)
            except (TypeError, ValueError):
                observation_count = 0
            if source in {"not_visible", "too_small"}:
                continue
            if observation_count < min_color_observations:
                continue
        track_key = (str(row["camera_id"] or ""), row["track_id"])
        if max_per_track > 0 and selected_per_track.get(track_key, 0) >= max_per_track:
            continue
        hard_fields: list[str] = []
        for field in ("upper_color", "lower_color"):
            candidate = candidates.get(field) or {}
            hsv = candidate.get("hsv_color")
            lab = candidate.get("lab_color")
            model = candidate.get("model_color")
            if focus_field == field and model:
                hard_fields.append(field)
            elif not focus_field and model and ((hsv and model != hsv) or (lab and model != lab)):
                hard_fields.append(field)
            elif not focus_field and sources.get(field) in {"hsv_lab_conflict_model_veto", "color_yolov8n"}:
                hard_fields.append(field)
        crop_path = _resolve_crop(row["crop_path"])
        if not hard_fields or crop_path is None:
            continue
        items.append(
            {
                "id": row["id"],
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "camera_id": row["camera_id"],
                "track_id": row["track_id"],
                "crop_path": crop_path,
                "stored": {
                    "upper_color": row["upper_color"],
                    "lower_color": row["lower_color"],
                },
                "hard_fields": hard_fields,
                "color_sources": {field: sources.get(field) for field in hard_fields},
                "candidates": {field: candidates.get(field, {}) for field in hard_fields},
                "review_label": None,
            }
        )
        selected_per_track[track_key] = selected_per_track.get(track_key, 0) + 1
        if len(items) >= limit:
            break

    return {
        "description": "Human-review manifest; review_label must be filled before training.",
        "db_path": str(db_path),
        "limit": limit,
        "focus_field": focus_field,
        "min_color_observations": min_color_observations,
        "max_per_track": max_per_track,
        "count": len(items),
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/runtime/appearances.db"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--focus-field", choices=("upper_color", "lower_color"))
    parser.add_argument("--min-color-observations", type=int, default=0)
    parser.add_argument("--max-per-track", type=int, default=0)
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}")
    manifest = build_manifest(
        args.db,
        args.limit,
        args.focus_field,
        min_color_observations=max(0, args.min_color_observations),
        max_per_track=max(0, args.max_per_track),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} items={manifest['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
