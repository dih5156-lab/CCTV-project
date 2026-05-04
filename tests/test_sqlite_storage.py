"""SQLite 공통 저장소 헬퍼 테스트."""

from __future__ import annotations

import sqlite3

from src.storage import SQLiteDatabase


def test_sqlite_database_initializes_schema_and_row_factory(tmp_path):
    db = SQLiteDatabase(tmp_path / "sample.db")
    conn = db.initialize(
        """
        CREATE TABLE IF NOT EXISTS sample (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        );
        """
    )
    try:
        conn.execute("INSERT INTO sample (name) VALUES (?)", ("alpha",))
        conn.commit()
        row = conn.execute("SELECT name FROM sample").fetchone()
        assert isinstance(row, sqlite3.Row)
        assert row["name"] == "alpha"
    finally:
        conn.close()


def test_sqlite_database_session_commits_and_rolls_back(tmp_path):
    db = SQLiteDatabase(tmp_path / "session.db")
    with db.session() as conn:
        conn.execute("CREATE TABLE sample (name TEXT)")
        conn.execute("INSERT INTO sample (name) VALUES ('committed')")

    try:
        with db.session() as conn:
            conn.execute("INSERT INTO sample (name) VALUES ('rolled_back')")
            raise RuntimeError("force rollback")
    except RuntimeError:
        pass

    with db.session() as conn:
        rows = conn.execute("SELECT name FROM sample ORDER BY name").fetchall()

    assert [row["name"] for row in rows] == ["committed"]
