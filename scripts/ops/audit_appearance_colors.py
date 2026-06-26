#!/usr/bin/env python3
"""최근 외형 로그의 색상 판정과 저장 crop 재분석 결과를 비교한다."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ai._appearance_analyzer import AppearanceAnalyzer  # noqa: E402


def _resolve_path(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    text = str(raw_path)
    if text.startswith("/app/"):
        candidate = PROJECT_ROOT / text.removeprefix("/app/")
        if candidate.exists():
            return candidate
    candidate = PROJECT_ROOT / text
    if candidate.exists():
        return candidate
    return None


def _query_rows(db_path: Path, limit: int, backend: Optional[str]) -> list[sqlite3.Row]:
    where = "crop_path IS NOT NULL AND crop_path != ''"
    params: list[object] = []
    if backend:
        where += " AND attribute_backend = ?"
        params.append(backend)
    params.append(limit)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return list(
            conn.execute(
                f"""
                SELECT id, camera_id, track_id, attribute_backend,
                       upper_color, lower_color, crop_path, timestamp
                FROM appearance_log
                WHERE {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            )
        )


def audit_colors(db_path: Path, *, limit: int, backend: Optional[str]) -> int:
    analyzer = AppearanceAnalyzer()
    rows = _query_rows(db_path, limit, backend)
    stored_distribution: Counter[tuple[str, str]] = Counter()
    recalculated_distribution: Counter[tuple[str, str]] = Counter()
    mismatches = []
    missing_crops = 0

    for row in rows:
        stored = (row["upper_color"] or "unknown", row["lower_color"] or "unknown")
        stored_distribution[stored] += 1
        crop_path = _resolve_path(row["crop_path"])
        if crop_path is None:
            missing_crops += 1
            continue
        image = cv2.imread(str(crop_path))
        if image is None:
            missing_crops += 1
            continue
        height, width = image.shape[:2]
        attrs = analyzer.extract_attributes(image, 0, 0, width, height)
        recalculated = (
            str(attrs.get("upper_color") or "unknown"),
            str(attrs.get("lower_color") or "unknown"),
        )
        recalculated_distribution[recalculated] += 1
        if stored != recalculated:
            mismatches.append((row, stored, recalculated, crop_path))

    print(f"db_path={db_path}")
    print(f"checked_rows={len(rows)} missing_crops={missing_crops} mismatches={len(mismatches)}")
    print("stored_distribution:")
    for (upper, lower), count in stored_distribution.most_common(10):
        print(f"  {upper}/{lower}: {count}")
    print("recalculated_distribution:")
    for (upper, lower), count in recalculated_distribution.most_common(10):
        print(f"  {upper}/{lower}: {count}")
    if mismatches:
        print("mismatch_samples:")
        for row, stored, recalculated, crop_path in mismatches[:10]:
            print(
                "  "
                f"id={row['id']} track={row['track_id']} backend={row['attribute_backend']} "
                f"stored={stored[0]}/{stored[1]} recalculated={recalculated[0]}/{recalculated[1]} "
                f"crop={crop_path}"
            )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("APPEARANCES_DB", "data/runtime/appearances.db")),
        help="appearance_log SQLite DB path",
    )
    parser.add_argument("--limit", type=int, default=80, help="최근 crop 재분석 개수")
    parser.add_argument("--backend", default=None, help="attribute_backend 필터")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = _resolve_path(str(args.db)) or args.db
    if not db_path.exists():
        print(f"DB를 찾을 수 없습니다: {db_path}", file=sys.stderr)
        return 2
    return audit_colors(db_path, limit=max(1, args.limit), backend=args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
