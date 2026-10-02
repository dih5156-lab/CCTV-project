#!/usr/bin/env python3
"""학습 manifest에서 장면 그룹이 겹치지 않는 분할 manifest를 만든다."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _scene_group(row: dict[str, Any]) -> str:
    """동일 장면의 여러 카메라 행을 하나로 묶을 식별자를 반환한다."""
    return str(row.get("scene_group") or row.get("scene_id") or row.get("video_path") or "").strip()


def rewrite_video_paths(
    rows: list[dict[str, Any]],
    *,
    source_root: str,
    target_root: str,
) -> list[dict[str, Any]]:
    """컨테이너 등 실행 환경에 맞게 영상 루트 경로만 변환한다."""
    normalized_source = source_root.rstrip("/")
    normalized_target = target_root.rstrip("/")
    rewritten_rows = []
    for row in rows:
        rewritten = dict(row)
        video_path = rewritten.get("video_path")
        if isinstance(video_path, str) and video_path.startswith(normalized_source + "/"):
            rewritten["video_path"] = normalized_target + video_path[len(normalized_source) :]
        rewritten_rows.append(rewritten)
    return rewritten_rows


def split_rows_by_scene_group(
    rows: list[dict[str, Any]],
    *,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """라벨별 장면 그룹을 train·validation·test로 결정론적으로 나눈다."""
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("분할 비율은 0 이상이어야 하며 validation과 test의 합은 1보다 작아야 한다")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_labels: dict[str, str] = {}
    for row in rows:
        group = _scene_group(row)
        if not group:
            raise ValueError("scene_group 또는 scene_id가 없는 행이 있다")
        label = str(row.get("label") or ("fall" if row.get("is_fall") else "not_fall"))
        previous_label = group_labels.setdefault(group, label)
        if previous_label != label:
            raise ValueError(f"혼합된 라벨을 가진 scene_group: {group}")
        grouped[group].append(row)

    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group, label in group_labels.items():
        groups_by_label[label].append(group)

    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for label in sorted(groups_by_label):
        groups = sorted(groups_by_label[label])
        test_count = round(len(groups) * test_ratio)
        validation_count = round(len(groups) * validation_ratio)
        if len(groups) >= 3:
            test_count = max(1, test_count)
            validation_count = max(1, validation_count)
        test_groups = set(groups[:test_count])
        validation_groups = set(groups[test_count : test_count + validation_count])
        for group in groups:
            split_name = (
                "test"
                if group in test_groups
                else "validation"
                if group in validation_groups
                else "train"
            )
            result[split_name].extend(grouped[group])

    return result


def _direction(row: dict[str, Any]) -> str:
    value = str(row.get("scene_category") or "").lower()
    if "전면" in value:
        return "front"
    if "측면" in value:
        return "side"
    if "후면" in value:
        return "back"
    return "non_fall" if not row.get("is_fall") else "other_fall"


def _group(row: dict[str, Any]) -> str:
    return str(row.get("scene_group") or row.get("scene_id") or row.get("video_path"))


def select_test_rows(
    rows: list[dict[str, Any]],
    per_group: int = 5,
    cameras_per_group: int = 1,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_direction(row)].append(row)
    selected: list[dict[str, Any]] = []
    for key in ("front", "side", "back", "non_fall", "other_fall"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(buckets.get(key, []), key=lambda item: str(item.get("scene_id"))):
            groups[_group(row)].append(row)
        for group_rows in list(groups.values())[: max(per_group, 0)]:
            selected.extend(group_rows[: max(cameras_per_group, 0)])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/fall_eval/test_manifest.jsonl"))
    parser.add_argument("--per-group", type=int, default=5, help="방향별 선택 그룹 수.")
    parser.add_argument(
        "--cameras-per-group",
        type=int,
        default=1,
        help="선택한 장면 그룹마다 포함할 카메라 영상 수.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=None,
        help="지정하면 장면 그룹 기준 train·validation·test manifest를 생성한다.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="validation에 배정할 장면 그룹 비율.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="test에 배정할 장면 그룹 비율.",
    )
    parser.add_argument(
        "--source-video-root",
        type=str,
        help="manifest에 기록된 원본 영상 루트. target과 함께 지정한다.",
    )
    parser.add_argument(
        "--target-video-root",
        type=str,
        help="실행 환경에서 사용할 영상 루트. source와 함께 지정한다.",
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if bool(args.source_video_root) != bool(args.target_video_root):
        raise SystemExit("source-video-root와 target-video-root는 함께 지정해야 한다")
    if args.source_video_root and args.target_video_root:
        rows = rewrite_video_paths(
            rows,
            source_root=args.source_video_root,
            target_root=args.target_video_root,
        )
    if args.split_output_dir is not None:
        splits = split_rows_by_scene_group(
            rows,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
        )
        args.split_output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_rows in splits.items():
            output_path = args.split_output_dir / f"{split_name}.jsonl"
            output_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows),
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "output_dir": str(args.split_output_dir),
                    "rows": {name: len(split_rows) for name, split_rows in splits.items()},
                    "groups": {
                        name: len({_scene_group(row) for row in split_rows})
                        for name, split_rows in splits.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    selected = select_test_rows(
        rows,
        per_group=args.per_group,
        cameras_per_group=args.cameras_per_group,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(selected), "directions": {key: sum(_direction(row) == key for row in selected) for key in ("front", "side", "back", "non_fall", "other_fall")}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
