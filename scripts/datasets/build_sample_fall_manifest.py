#!/usr/bin/env python3
"""Build a video manifest from the local Sample fall dataset.

The Sample package is organized as:

  Sample/01.원천데이터/영상/{Y,N}/.../<scene_id>.mp4
  Sample/02.라벨링데이터/영상/{Y,N}/.../<scene_id>.json

This script joins source videos with their label JSON and writes a JSONL
manifest that can be used for DeepStream replay checks or offline evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_ROOT = Path("Sample")
DEFAULT_OUTPUT = Path("data/fall_eval/sample_manifest.jsonl")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _label_from_parts(path: Path, metadata: dict[str, Any]) -> str:
    scene_info = metadata.get("scene_info") or {}
    scene_is_fall = str(scene_info.get("scene_IsFall") or "").strip()
    if scene_is_fall == "낙상":
        return "fall"
    if scene_is_fall == "비낙상":
        return "not_fall"
    parts = set(path.parts)
    if "Y" in parts:
        return "fall"
    if "N" in parts:
        return "not_fall"
    return "unknown"


def _source_video_for_label(sample_root: Path, label_json: Path) -> Path:
    relative = label_json.relative_to(sample_root / "02.라벨링데이터" / "영상")
    return sample_root / "01.원천데이터" / "영상" / relative.with_suffix(".mp4")


def build_manifest(sample_root: Path) -> list[dict[str, Any]]:
    label_root = sample_root / "02.라벨링데이터" / "영상"
    rows: list[dict[str, Any]] = []
    for label_json in sorted(label_root.rglob("*.json")):
        metadata = _read_json(label_json)
        source_video = _source_video_for_label(sample_root, label_json)
        scene_info = metadata.get("scene_info") or {}
        sensor_data = metadata.get("sensordata") or {}
        meta = metadata.get("metadata") or {}
        actor_info = metadata.get("actor_info") or {}

        rows.append(
            {
                "scene_id": str(meta.get("scene_id") or label_json.stem),
                "video_path": str(source_video),
                "label_path": str(label_json),
                "label": _label_from_parts(label_json, metadata),
                "is_fall": _label_from_parts(label_json, metadata) == "fall",
                "fall_start_frame": int(sensor_data.get("fall_start_frame") or 0),
                "fall_end_frame": int(sensor_data.get("fall_end_frame") or 0),
                "scene_length": int(scene_info.get("scene_length") or 0),
                "camera": int(scene_info.get("cam_num") or 0),
                "scene_category": scene_info.get("scene_cat_name"),
                "fall_type": scene_info.get("fall_type"),
                "scene_location": scene_info.get("scene_loc"),
                "scene_position": scene_info.get("scene_pos"),
                "scene_method": scene_info.get("scene_method"),
                "actor_age": actor_info.get("actor_age"),
                "actor_sex": actor_info.get("actor_sex"),
                "video_exists": source_video.exists(),
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    fall_count = sum(1 for row in rows if row["label"] == "fall")
    not_fall_count = sum(1 for row in rows if row["label"] == "not_fall")
    missing_videos = sum(1 for row in rows if not row["video_exists"])
    cameras = sorted({row["camera"] for row in rows})
    print(f"rows: {len(rows)}")
    print(f"fall: {fall_count}")
    print(f"not_fall: {not_fall_count}")
    print(f"missing_videos: {missing_videos}")
    print(f"cameras: {','.join(str(camera) for camera in cameras)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("data/fall_eval/sample_manifest.csv"),
    )
    args = parser.parse_args()

    rows = build_manifest(args.sample_root)
    write_jsonl(rows, args.output)
    write_csv(rows, args.csv_output)
    print_summary(rows)
    print(f"jsonl: {args.output}")
    print(f"csv: {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
