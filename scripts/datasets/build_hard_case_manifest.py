#!/usr/bin/env python3
"""Select deterministic, uncached multi-camera fall hard cases for retraining."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_available(row: dict[str, object], cache: Path, max_frames: int, frame_stride: int) -> bool:
    scene_id = str(row.get("scene_id") or "")
    video_path = Path(str(row.get("video_path") or ""))
    return bool(scene_id and video_path.exists() and not (cache / f"{scene_id}_uniform_max{max_frames}_stride{frame_stride}.json").exists())


def select_rows(
    rows: list[dict[str, object]],
    *,
    cache: Path,
    max_fall: int,
    max_notfall: int,
    min_camera: int,
    max_frames: int,
    frame_stride: int,
    seed: int,
) -> list[dict[str, object]]:
    candidates = [
        row
        for row in rows
        if int(row.get("camera") or 0) >= min_camera
        and _is_available(row, cache, max_frames, frame_stride)
    ]
    random.Random(seed).shuffle(candidates)
    falls = [row for row in candidates if bool(row.get("is_fall"))][:max_fall]
    notfalls = [row for row in candidates if not bool(row.get("is_fall"))][:max_notfall]
    return sorted(falls + notfalls, key=lambda row: str(row.get("scene_id") or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fall", type=int, default=40)
    parser.add_argument("--max-notfall", type=int, default=40)
    parser.add_argument("--min-camera", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    rows = select_rows(
        _read_rows(args.manifest),
        cache=args.cache,
        max_fall=args.max_fall,
        max_notfall=args.max_notfall,
        min_camera=args.min_camera,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"selected={len(rows)} fall={sum(bool(row.get('is_fall')) for row in rows)} notfall={sum(not bool(row.get('is_fall')) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
