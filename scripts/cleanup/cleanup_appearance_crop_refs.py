#!/usr/bin/env python3
"""삭제된 외형 crop 파일을 가리키는 DB 참조를 정리한다."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_db_path() -> Path:
    return Path(os.environ.get("APPEARANCES_DB", PROJECT_ROOT / "data" / "runtime" / "appearances.db"))


def _default_crop_dir() -> Path:
    return Path(os.environ.get("APPEARANCE_CROP_DIR", PROJECT_ROOT / "data" / "runtime" / "appearance_crops"))


def _has_appearance_log(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'appearance_log'"
    ).fetchone()
    return row is not None


def _crop_exists(crop_path: str, crop_dirs: list[Path]) -> bool:
    filename = Path(crop_path).name
    return any((crop_dir / filename).is_file() for crop_dir in crop_dirs)


def cleanup_missing_crop_refs(db_path: Path, crop_dir: Path, *, apply: bool) -> tuple[int, int]:
    """실제 파일이 없는 crop_path 개수를 확인하고 필요하면 NULL로 변경한다."""
    if not db_path.exists():
        print(f"외형 DB 없음: {db_path}")
        return 0, 0

    legacy_crop_dir = PROJECT_ROOT / "data" / "crops"
    crop_dirs = [crop_dir]
    if legacy_crop_dir != crop_dir:
        crop_dirs.append(legacy_crop_dir)

    with sqlite3.connect(db_path, timeout=30) as conn:
        if not _has_appearance_log(conn):
            print(f"appearance_log 테이블 없음: {db_path}")
            return 0, 0

        rows = conn.execute(
            """
            SELECT id, crop_path
            FROM appearance_log
            WHERE crop_path IS NOT NULL AND crop_path != ''
            """
        ).fetchall()
        stale_ids = [
            row_id
            for row_id, crop_path in rows
            if not _crop_exists(str(crop_path), crop_dirs)
        ]

        if apply and stale_ids:
            conn.executemany(
                "UPDATE appearance_log SET crop_path = NULL WHERE id = ?",
                ((row_id,) for row_id in stale_ids),
            )
            conn.commit()

    return len(rows), len(stale_ids)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="삭제된 외형 crop 파일을 가리키는 SQLite 참조를 정리합니다."
    )
    parser.add_argument("--apply", action="store_true", help="누락된 crop_path를 NULL로 변경합니다.")
    parser.add_argument("--db-path", type=Path, default=_default_db_path())
    parser.add_argument("--crop-dir", type=Path, default=_default_crop_dir())
    args = parser.parse_args()

    checked, stale = cleanup_missing_crop_refs(args.db_path, args.crop_dir, apply=args.apply)
    print(f"crop 참조 확인 수: {checked}")
    print(f"파일 없는 crop 참조 수: {stale}")
    print(f"모드: {'적용' if args.apply else '미리보기'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
