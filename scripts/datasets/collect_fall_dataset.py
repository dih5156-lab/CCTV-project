#!/usr/bin/env python3
"""Initialize a fall-video dataset or add a video to its labeling queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "fall_dataset"


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    pending_dir: Path
    fall_dir: Path
    non_fall_dir: Path
    annotation_dir: Path
    review_log: Path
    manifest_dir: Path


def dataset_paths(root: Path) -> DatasetPaths:
    return DatasetPaths(
        root=root,
        pending_dir=root / "clips" / "pending",
        fall_dir=root / "clips" / "labeled" / "fall",
        non_fall_dir=root / "clips" / "labeled" / "non_fall",
        annotation_dir=root / "annotations",
        review_log=root / "annotations" / "review.jsonl",
        manifest_dir=root / "manifests",
    )


def initialize_dataset(root: Path) -> DatasetPaths:
    paths = dataset_paths(root)
    for directory in (
        paths.pending_dir,
        paths.fall_dir,
        paths.non_fall_dir,
        paths.annotation_dir,
        paths.manifest_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    paths.review_log.touch(exist_ok=True)
    return paths


def _safe_name(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    )
    return normalized.strip("_") or "camera"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
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


def collect_video(
    video_path: Path,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    camera_id: str,
    source_name: str = "manual",
    note: str = "",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    source_path = video_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"video not found: {source_path}")

    paths = initialize_dataset(dataset_root.expanduser().resolve())
    video_hash = _sha256(source_path)
    existing_rows = _read_rows(paths.review_log)
    if any(row.get("sha256") == video_hash for row in existing_rows):
        raise ValueError(f"duplicate video: sha256={video_hash}")

    captured_at = created_at or datetime.now(timezone.utc)
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=timezone.utc)
    event_id = (
        f"{_safe_name(camera_id)}_"
        f"{captured_at.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    suffix = source_path.suffix.lower() or ".mp4"
    destination = paths.pending_dir / f"{event_id}{suffix}"
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    shutil.copy2(source_path, destination)

    record: dict[str, Any] = {
        "event_id": event_id,
        "created_at": captured_at.astimezone(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "event_type": "manual_collection",
        "review_source": source_name,
        "clip_path": str(destination),
        "sha256": video_hash,
        "label": None,
        "review_status": "unlabeled",
        "note": note,
    }
    paths.review_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with paths.review_log.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--init", action="store_true", help="Create directories only")
    parser.add_argument("--video", type=Path, help="Video to copy into clips/pending")
    parser.add_argument("--camera", default="camera_1")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.init:
        paths = initialize_dataset(args.dataset_root)
        print(f"Fall dataset initialized: {paths.root}")
        return 0
    if args.video is None:
        raise SystemExit("--video is required unless --init is used")
    record = collect_video(
        args.video,
        dataset_root=args.dataset_root,
        camera_id=args.camera,
        source_name=args.source,
        note=args.note,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
