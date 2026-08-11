"""검수한 상·하의 색상을 SQLite에 안전하게 반영하는 테스트."""

from __future__ import annotations

import importlib
import json
import sqlite3

import pytest


def _review_apply_module():
    return importlib.import_module(
        "scripts.ops.apply_appearance_color_review_labels"
    )


def _create_appearance_db(tmp_path, upper: str = "black", lower: str = "blue"):
    db_path = tmp_path / "appearances.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE appearance_log (
                id INTEGER PRIMARY KEY,
                upper_color TEXT,
                lower_color TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO appearance_log (id, upper_color, lower_color)
            VALUES (1, ?, ?)
            """,
            (upper, lower),
        )
    return db_path


def _write_labels(tmp_path, payload: dict, name: str = "labels.json"):
    labels_path = tmp_path / name
    labels_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return labels_path


def _read_colors(db_path, item_id: int = 1):
    with sqlite3.connect(db_path) as connection:
        return connection.execute(
            """
            SELECT upper_color, lower_color
            FROM appearance_log
            WHERE id = ?
            """,
            (item_id,),
        ).fetchone()


def test_dry_run_plans_partial_update_without_changing_database(tmp_path):
    db_path = _create_appearance_db(tmp_path)
    labels_path = _write_labels(
        tmp_path,
        {
            "schema_version": 1,
            "items": [
                {"id": 1, "upper_color": "white", "lower_color": None}
            ],
        },
    )

    summary = _review_apply_module().apply_review_labels(db_path, labels_path)

    assert summary["mode"] == "dry-run"
    assert summary["updates"] == 1
    assert summary["changes"][0]["after"] == {"upper_color": "white"}
    assert _read_colors(db_path) == ("black", "blue")


def test_apply_creates_backup_and_updates_both_fields(tmp_path):
    db_path = _create_appearance_db(tmp_path)
    labels_path = _write_labels(
        tmp_path,
        {
            "schema_version": 1,
            "items": [
                {"id": 1, "upper_color": "white", "lower_color": "gray"}
            ],
        },
    )
    backup_path = tmp_path / "before.db"

    summary = _review_apply_module().apply_review_labels(
        db_path,
        labels_path,
        apply=True,
        backup_path=backup_path,
    )

    assert summary["mode"] == "applied"
    assert summary["updates"] == 1
    assert summary["backup"] == str(backup_path)
    assert _read_colors(db_path) == ("white", "gray")
    assert _read_colors(backup_path) == ("black", "blue")


@pytest.mark.parametrize(
    "items",
    [
        [{"id": 1, "upper_color": "cyan", "lower_color": None}],
        [
            {"id": 1, "upper_color": "red", "lower_color": None},
            {"id": 1, "upper_color": None, "lower_color": "black"},
        ],
        [{"id": 999, "upper_color": "red", "lower_color": None}],
    ],
)
def test_invalid_review_input_changes_nothing(tmp_path, items):
    db_path = _create_appearance_db(tmp_path)
    labels_path = _write_labels(
        tmp_path,
        {"schema_version": 1, "items": items},
    )

    with pytest.raises(ValueError):
        _review_apply_module().apply_review_labels(
            db_path,
            labels_path,
            apply=True,
        )

    assert _read_colors(db_path) == ("black", "blue")


def test_null_exclude_and_same_values_do_not_update_database(tmp_path):
    db_path = _create_appearance_db(tmp_path)
    labels_path = _write_labels(
        tmp_path,
        {
            "schema_version": 1,
            "items": [
                {
                    "id": 1,
                    "upper_color": "black",
                    "lower_color": "exclude",
                }
            ],
        },
    )

    summary = _review_apply_module().apply_review_labels(
        db_path,
        labels_path,
        apply=True,
        backup_path=tmp_path / "before.db",
    )

    assert summary["updates"] == 0
    assert _read_colors(db_path) == ("black", "blue")


def test_apply_refuses_to_overwrite_existing_backup(tmp_path):
    db_path = _create_appearance_db(tmp_path)
    labels_path = _write_labels(
        tmp_path,
        {
            "schema_version": 1,
            "items": [
                {"id": 1, "upper_color": "white", "lower_color": None}
            ],
        },
    )
    backup_path = tmp_path / "before.db"
    backup_path.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        _review_apply_module().apply_review_labels(
            db_path,
            labels_path,
            apply=True,
            backup_path=backup_path,
        )

    assert backup_path.read_bytes() == b"keep"
    assert _read_colors(db_path) == ("black", "blue")
