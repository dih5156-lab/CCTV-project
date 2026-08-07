#!/usr/bin/env python3
"""학습/검증 manifest를 FiftyOne 데이터셋으로 등록하고 App을 실행한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import fiftyone as fo


HOST_DATASET_ROOT = Path("/media/sawwave/Learning11/낙상학습데이터")
CONTAINER_DATASET_ROOT = "/app/낙상학습데이터"
DEFAULT_TRAIN_MANIFEST = Path("data/fall_eval/auto/train_manifest.jsonl")
DEFAULT_VALIDATION_MANIFEST = Path("data/fall_eval/auto/validation_manifest.jsonl")
DEFAULT_DATASET_NAME = "fall_eval_auto"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield row


def _host_media_path(video_path: str) -> Path:
    if video_path.startswith(CONTAINER_DATASET_ROOT):
        relative = video_path[len(CONTAINER_DATASET_ROOT) :].lstrip("/")
        return HOST_DATASET_ROOT / relative
    return Path(video_path)


def _sample_from_row(row: dict[str, Any], split: str, media_path: Path) -> fo.Sample:
    label = str(row.get("label") or ("fall" if row.get("is_fall") else "not_fall"))
    sample = fo.Sample(
        filepath=str(media_path),
        ground_truth=fo.Classification(label=label),
        split=split,
        label=label,
        is_fall=bool(row.get("is_fall", label == "fall")),
        fall_type=str(row.get("fall_type") or "none"),
        scene_category=str(row.get("scene_category") or ""),
        scene_id=str(row.get("scene_id") or media_path.stem),
        scene_group=str(row.get("scene_group") or ""),
        camera=int(row["camera"]) if row.get("camera") is not None else None,
        actor_age=str(row.get("actor_age") or ""),
        actor_sex=str(row.get("actor_sex") or ""),
        scene_location=str(row.get("scene_location") or ""),
        scene_position=str(row.get("scene_position") or ""),
        fall_start_frame=int(row.get("fall_start_frame") or 0),
        fall_end_frame=int(row.get("fall_end_frame") or 0),
        scene_length=int(row.get("scene_length") or 0),
        source_manifest=split,
    )
    sample.tags = [split, label]
    return sample


def register_dataset(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    dataset_name: str,
) -> tuple[fo.Dataset, dict[str, int]]:
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        dataset.delete_samples(dataset.values("id"))
    else:
        dataset = fo.Dataset(dataset_name, persistent=True)

    counts = {"train": 0, "validation": 0, "missing": 0}
    for split, manifest in (("train", train_manifest), ("validation", validation_manifest)):
        samples = []
        for row in _read_jsonl(manifest):
            media_path = _host_media_path(str(row.get("video_path") or ""))
            if not media_path.is_file():
                counts["missing"] += 1
                continue
            samples.append(_sample_from_row(row, split, media_path))
            counts[split] += 1
            if len(samples) >= 500:
                dataset.add_samples(samples, dynamic=True)
                samples = []
        if samples:
            dataset.add_samples(samples, dynamic=True)

    # 영상 메타데이터(ffprobe)는 18k개 전체를 순회하므로 등록 단계에서는 생략한다.
    # App에서 선택한 샘플을 열 때 필요한 정보만 지연 로드한다.
    dataset.save()
    return dataset, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--port", type=int, default=5151)
    args = parser.parse_args()

    dataset, counts = register_dataset(
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        dataset_name=args.dataset_name,
    )
    print(f"dataset={dataset.name} samples={len(dataset)} counts={counts}", flush=True)
    if args.launch:
        session = fo.launch_app(dataset, address="0.0.0.0", port=args.port, auto=False)
        print(f"app=http://localhost:{args.port}", flush=True)
        session.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
