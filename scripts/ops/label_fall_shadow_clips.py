#!/usr/bin/env python3
"""Interactively label fall shadow review clips with OpenCV.

Keys: F=fall, N=non-fall, S=needs review, SPACE=pause, R=replay, Q=quit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_LOG = PROJECT_ROOT / "data/logs/fall_shadow_review.jsonl"
DEFAULT_CLIP_DIR = PROJECT_ROOT / "data/fall_review_clips"


def read_review_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at line {line_number}")
            rows.append(row)
    return rows


def resolve_clip_path(
    raw_path: str, *, clip_dir: Path = DEFAULT_CLIP_DIR
) -> Path:
    """Map container clip paths to the local project clip directory."""
    path = Path(raw_path)
    if path.exists():
        return path
    return clip_dir / path.name


def select_candidates(
    rows: list[dict[str, Any]],
    *,
    camera: str | None,
    include_sample_eval: bool,
    clip_dir: Path = DEFAULT_CLIP_DIR,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if (
            row.get("label") is not None
            or row.get("review_status") == "needs_review"
            or not row.get("clip_path")
        ):
            continue
        camera_id = str(row.get("camera_id") or "")
        if camera and camera_id != camera:
            continue
        if not include_sample_eval and camera_id == "sample_eval":
            continue
        local_clip = resolve_clip_path(str(row["clip_path"]), clip_dir=clip_dir)
        if not local_clip.is_file():
            continue
        candidate = dict(row)
        candidate["local_clip_path"] = str(local_clip)
        candidates.append(candidate)
    return candidates


def write_review_log_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as fp:
        temporary_path = Path(fp.name)
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        fp.flush()
    temporary_path.replace(path)


def apply_label(
    path: Path,
    *,
    event_id: str,
    label: str | None,
    review_status: str,
) -> None:
    """Reload and update one event so each key press is saved immediately."""
    rows = read_review_log(path)
    matches = [row for row in rows if row.get("event_id") == event_id]
    if len(matches) != 1:
        raise ValueError(f"expected one event_id={event_id!r}, found {len(matches)}")
    matches[0]["label"] = label
    matches[0]["review_status"] = review_status
    write_review_log_atomic(path, rows)


def create_backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = path.with_name(f"{path.name}.{stamp}.bak")
    shutil.copy2(path, backup)
    return backup


def _draw_overlay(frame: Any, candidate: dict[str, Any], index: int, total: int) -> None:
    import cv2

    lines = [
        f"[{index + 1}/{total}] {candidate.get('event_id')}",
        "F: fall   N: non-fall   S: skip/review   SPACE: pause   R: replay   Q: quit",
    ]
    for line_index, text in enumerate(lines):
        y = 32 + line_index * 30
        cv2.putText(
            frame,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def review_candidate(
    candidate: dict[str, Any], index: int, total: int
) -> tuple[str | None, str] | None:
    import cv2

    clip_path = str(candidate["local_clip_path"])
    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open clip: {clip_path}")

    paused = False
    last_frame = None
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                last_frame = frame
            if last_frame is None:
                continue
            display_frame = last_frame.copy()
            _draw_overlay(display_frame, candidate, index, total)
            cv2.imshow("Fall shadow clip labeler", display_frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("f"), ord("F")):
                return "fall", "reviewed"
            if key in (ord("n"), ord("N")):
                return "non_fall", "reviewed"
            if key in (ord("s"), ord("S")):
                return None, "needs_review"
            if key in (ord("q"), ord("Q"), 27):
                return None
            if key == ord(" "):
                paused = not paused
            if key in (ord("r"), ord("R")):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-log", type=Path, default=DEFAULT_REVIEW_LOG)
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--camera", default="camera_1")
    parser.add_argument("--include-sample-eval", action="store_true")
    parser.add_argument("--list", action="store_true", help="List candidates without GUI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_review_log(args.review_log)
    candidates = select_candidates(
        rows,
        camera=None if args.include_sample_eval else args.camera,
        include_sample_eval=args.include_sample_eval,
        clip_dir=args.clip_dir,
    )
    print(f"Unlabeled clips: {len(candidates)}")
    if args.list:
        for candidate in candidates[:20]:
            print(candidate["event_id"], candidate["local_clip_path"])
        return 0
    if not candidates:
        return 0

    backup = create_backup(args.review_log)
    print(f"Backup: {backup}")
    for index, candidate in enumerate(candidates):
        result = review_candidate(candidate, index, len(candidates))
        if result is None:
            print("Saved. Resume with the same command.")
            break
        label, review_status = result
        apply_label(
            args.review_log,
            event_id=str(candidate["event_id"]),
            label=label,
            review_status=review_status,
        )
        print(f"{candidate['event_id']}: {label or review_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
