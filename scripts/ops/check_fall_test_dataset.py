#!/usr/bin/env python3
"""실제 CCTV 낙상 테스트 영상의 폴더·파일명 규칙을 점검한다."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

POSES = {
    "forward_fall",
    "backward_fall",
    "left_fall",
    "right_fall",
    "upper_body_occluded",
    "lower_body_occluded",
}
FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<pose>[a-z_]+)\.mp4$"
)


def check_dataset(root: Path) -> dict[str, object]:
    errors: list[str] = []
    files: list[dict[str, str]] = []
    for category in ("fall", "notfall"):
        directory = root / category
        if not directory.is_dir():
            errors.append(f"missing directory: {directory}")
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() != ".mp4":
                errors.append(f"unsupported file: {path}")
                continue
            match = FILENAME_RE.fullmatch(path.name)
            if not match:
                errors.append(f"invalid filename: {path.name}")
                continue
            pose = match.group("pose")
            if category == "fall" and pose not in POSES:
                errors.append(f"invalid fall pose: {path.name}")
            if category == "notfall" and pose in POSES:
                errors.append(f"fall pose in notfall directory: {path.name}")
            files.append({"category": category, "pose": pose, "path": str(path)})
    return {
        "root": str(root),
        "file_count": len(files),
        "by_category": {
            category: sum(item["category"] == category for item in files)
            for category in ("fall", "notfall")
        },
        "files": files,
        "errors": errors,
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = check_dataset(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
