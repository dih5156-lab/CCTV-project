#!/usr/bin/env python3
"""낙상 데이터 준비·검증·학습을 한 명령으로 실행하는 오케스트레이터.

기본 동작은 원천/라벨 디렉터리에서 상세 라벨을 포함한 train/validation
manifest를 만들고 요약 리포트를 저장하는 것이다. ``--train``을 지정하면
기존 GPU 학습 래퍼를 호출한다. 장치 출력이나 운영 DB에는 접근하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.build_sample_fall_manifest import build_manifest, write_jsonl  # noqa: E402


def _group_key(row: dict[str, Any]) -> str:
    return str(row.get("scene_group") or row.get("scene_id") or row.get("label_path"))


def split_manifest(
    rows: list[dict[str, Any]], *, validation_ratio: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """같은 scene_group이 train/validation에 섞이지 않도록 결정론적으로 분할한다."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    validation: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    cutoff = int(validation_ratio * 10000)
    for group in sorted(groups):
        bucket = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) % 10000
        (validation if bucket < cutoff else train).extend(groups[group])

    # 작은 샘플에서도 두 split 모두 fall/non-fall을 유지한다.
    for label in ("fall", "not_fall"):
        if not any(row.get("label") == label for row in validation):
            candidates = [row for row in train if row.get("label") == label]
            if candidates:
                moved_group = _group_key(candidates[0])
                moved = [row for row in train if _group_key(row) == moved_group]
                train = [row for row in train if _group_key(row) != moved_group]
                validation.extend(moved)
        if not any(row.get("label") == label for row in train):
            candidates = [row for row in validation if row.get("label") == label]
            if candidates:
                moved_group = _group_key(candidates[0])
                moved = [row for row in validation if _group_key(row) == moved_group]
                validation = [row for row in validation if _group_key(row) != moved_group]
                train.extend(moved)
    return sorted(train, key=lambda row: str(row.get("scene_id"))), sorted(
        validation, key=lambda row: str(row.get("scene_id"))
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fall_rows = [row for row in rows if row.get("label") == "fall"]
    return {
        "rows": len(rows),
        "labels": dict(Counter(str(row.get("label")) for row in rows)),
        "fall_scene_categories": dict(
            Counter(str(row.get("scene_category") or "unknown") for row in fall_rows)
        ),
        "fall_types": dict(
            Counter(str(row.get("fall_type") or "unknown") for row in fall_rows)
        ),
        "missing_videos": sum(1 for row in rows if not row.get("video_exists")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root_group = parser.add_mutually_exclusive_group(required=True)
    root_group.add_argument("--dataset-root", type=Path, help="041 데이터셋 루트")
    root_group.add_argument("--source-video-root", type=Path)
    parser.add_argument("--label-video-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/fall_eval/auto")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--camera", type=int, choices=range(1, 9))
    parser.add_argument("--max-videos", type=int, default=200)
    parser.add_argument("--validation-max-videos", type=int, default=80)
    parser.add_argument("--train", action="store_true", help="GPU 학습 래퍼까지 실행")
    parser.add_argument("--train-direction", action="store_true", help="방향 보조 분류기까지 학습")
    parser.add_argument("--decision-threshold", type=float, default=0.7)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=PROJECT_ROOT / "models/experiments/yolo_pose_fall_cam2_continuous_200_80_640_metrics.json",
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=PROJECT_ROOT / "models/experiments/yolo_pose_fall_rf_metrics.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset_root:
        args.source_video_root = args.dataset_root / "01.원천데이터" / "extracted_TS" / "영상"
        args.label_video_root = args.dataset_root / "02.라벨링데이터" / "영상"
    if not args.source_video_root or not args.label_video_root:
        raise SystemExit("--source-video-root와 --label-video-root를 함께 지정하거나 --dataset-root를 사용하세요")
    rows = build_manifest(
        Path("."),
        source_video_root=args.source_video_root,
        label_video_root=args.label_video_root,
        camera=args.camera,
    )
    train_rows, validation_rows = split_manifest(rows, validation_ratio=args.validation_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = args.output_dir / "train_manifest.jsonl"
    validation_manifest = args.output_dir / "validation_manifest.jsonl"
    report_path = args.output_dir / "dataset_report.json"
    write_jsonl(train_rows, train_manifest)
    write_jsonl(validation_rows, validation_manifest)
    report = {
        "source_video_root": str(args.source_video_root),
        "label_video_root": str(args.label_video_root),
        "train": _summary(train_rows),
        "validation": _summary(validation_rows),
        "detail_fields": ["scene_category", "fall_type"],
        "decision_threshold": args.decision_threshold,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train_manifest": str(train_manifest), "validation_manifest": str(validation_manifest), "report": str(report_path), **report}, ensure_ascii=False, indent=2))

    if not args.train and not args.train_direction:
        return 0
    if report["train"]["missing_videos"] or report["validation"]["missing_videos"]:
        raise SystemExit("missing source videos; refusing to start training")
    if args.train:
        in_container = Path("/.dockerenv").exists()
        train_entrypoint = (
            [sys.executable, str(PROJECT_ROOT / "scripts/datasets/train_yolo_pose_fall_rf.py")]
            if in_container
            else [str(PROJECT_ROOT / "scripts/train_yolo_pose_fall_gpu.sh")]
        )
        command = [
            *train_entrypoint,
            "--manifest", str(train_manifest),
            "--validation-manifest", str(validation_manifest),
            "--max-videos", str(args.max_videos),
            "--validation-max-videos", str(args.validation_max_videos),
            "--decision-threshold", str(args.decision_threshold),
        ]
        if args.force_extract:
            command.append("--force-extract")
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    if args.train_direction:
        in_container = Path("/.dockerenv").exists()
        direction_entrypoint = (
            [sys.executable, str(PROJECT_ROOT / "scripts/datasets/train_fall_direction_rf.py")]
            if in_container
            else [str(PROJECT_ROOT / "scripts/train_fall_direction_gpu.sh")]
        )
        direction_command = [
            *direction_entrypoint,
            "--manifest", str(train_manifest),
            "--validation-manifest", str(validation_manifest),
            "--feature-cache", str(PROJECT_ROOT / "data/fall_eval/yolo_pose_fall_feature_cache"),
            "--validation-feature-cache", str(PROJECT_ROOT / "data/fall_eval/yolo_pose_fall_validation_feature_cache"),
        ]
        subprocess.run(direction_command, check=True, cwd=PROJECT_ROOT)
    if args.train and args.baseline_metrics.exists() and args.candidate_metrics.exists():
        compare_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/compare_fall_models.py"),
            "--baseline-metrics", str(args.baseline_metrics),
            "--candidate-metrics", str(args.candidate_metrics),
            "--baseline-model", str(PROJECT_ROOT / "models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl"),
            "--candidate-model", str(PROJECT_ROOT / "models/experiments/yolo_pose_fall_rf.pkl"),
        ]
        comparison = subprocess.run(compare_command, check=False, cwd=PROJECT_ROOT)
        print(f"model comparison exit={comparison.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
