#!/usr/bin/env python3
"""Build a replay manifest from camera error case IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("data/fall_eval/sample_manifest.jsonl"))
    parser.add_argument("--cases", type=Path, default=Path("data/fall_eval/camera_error_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("data/fall_eval/camera_error_cases_manifest.jsonl"))
    args = parser.parse_args()

    source_rows = {str(row.get("scene_id")): row for row in _read_jsonl(args.source_manifest)}
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    rows: list[dict] = []
    missing: list[str] = []
    for case in cases:
        scene_id = str(case["scene_id"])
        row = source_rows.get(scene_id)
        if row is None:
            missing.append(scene_id)
            continue
        rows.append({**row, "hard_case_error": case["error"], "hard_case_reason": case["reason"]})
    if missing:
        raise SystemExit(f"missing scene IDs: {', '.join(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {args.output} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
