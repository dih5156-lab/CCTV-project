"""삭제된 외형 crop DB 참조 정리 스크립트 테스트."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.cleanup.cleanup_appearance_crop_refs import cleanup_missing_crop_refs


def _create_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE appearance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_path TEXT
            )
            """
        )


def test_preview_reports_missing_refs_without_update(tmp_path: Path) -> None:
    db_path = tmp_path / "appearances.db"
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    _create_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO appearance_log(crop_path) VALUES ('data/crops/deleted.jpg')")

    checked, stale = cleanup_missing_crop_refs(db_path, crop_dir, apply=False)

    assert (checked, stale) == (1, 1)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT crop_path FROM appearance_log").fetchone()[0] == "data/crops/deleted.jpg"


def test_apply_clears_only_missing_refs(tmp_path: Path) -> None:
    db_path = tmp_path / "appearances.db"
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    (crop_dir / "present.jpg").write_bytes(b"jpeg")
    _create_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO appearance_log(crop_path) VALUES (?)",
            [("data/crops/present.jpg",), ("data/crops/deleted.jpg",), (None,)],
        )

    checked, stale = cleanup_missing_crop_refs(db_path, crop_dir, apply=True)

    assert (checked, stale) == (2, 1)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT crop_path FROM appearance_log ORDER BY id").fetchall()
    assert rows == [("data/crops/present.jpg",), (None,), (None,)]


def test_missing_db_is_noop(tmp_path: Path) -> None:
    checked, stale = cleanup_missing_crop_refs(
        tmp_path / "missing.db",
        tmp_path / "crops",
        apply=True,
    )
    assert (checked, stale) == (0, 0)
