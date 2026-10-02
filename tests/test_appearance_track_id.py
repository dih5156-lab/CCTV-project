"""Lossless tracker IDs across SQLite, reopen, migration, and the search API."""

import sqlite3

import pytest

from src.services.appearance_log import AppearanceLog


@pytest.mark.parametrize("track_id", [None, 0, -1, 2**53 + 1, 2**63 - 1, 2**63, 2**64 - 1])
def test_track_id_survives_storage_reopen_and_api(tmp_path, track_id):
    from src.api.v1.search import _to_record

    path = str(tmp_path / "appearance.db")
    with AppearanceLog(path) as log:
        assert log.insert(camera_id="cam1", track_id=track_id, timestamp=1000)
    with AppearanceLog(path) as log:
        row = log.search()[0]
        assert row["track_id"] == track_id
        assert row["track_id_text"] == (str(track_id) if track_id is not None else None)
        record = _to_record(row).model_dump(mode="json")
        assert record["track_id"] == track_id
        assert record["track_id_text"] == row["track_id_text"]
    with sqlite3.connect(path) as connection:
        stored, exact, kind = connection.execute(
            "SELECT track_id, track_id_text, typeof(track_id_text) FROM appearance_log"
        ).fetchone()
    assert stored == (track_id if track_id is None or track_id < 2**63 else None)
    assert kind == ("null" if track_id is None else "text")
    assert exact == (None if track_id is None else str(track_id))


def test_adjacent_large_ids_do_not_collide_or_share_cooldown(tmp_path):
    with AppearanceLog(str(tmp_path / "appearance.db")) as log:
        ids = [2**63, 2**63 + 1, 2**64 - 2, 2**64 - 1]
        for track_id in ids:
            assert log.insert(camera_id="cam1", track_id=track_id, timestamp=1000)
            assert not log.insert(camera_id="cam1", track_id=track_id, timestamp=1001)
        assert log.insert(camera_id="cam2", track_id=ids[0], timestamp=1001)
        rows = log.search(camera_id="cam1")
        assert {row["track_id"] for row in rows} == set(ids)
        assert len({row["event_id"] for row in rows}) == len(ids)


def test_old_integer_database_migration_preserves_records_and_is_repeatable(tmp_path):
    path = str(tmp_path / "legacy.db")
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE appearance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, timestamp REAL NOT NULL,
                camera_id TEXT NOT NULL, track_id INTEGER, upper_color TEXT, lower_color TEXT,
                has_helmet INTEGER DEFAULT 0, helmet_color TEXT, has_backpack INTEGER DEFAULT 0,
                has_handbag INTEGER DEFAULT 0, has_suitcase INTEGER DEFAULT 0,
                gender TEXT, age_group TEXT, face_name TEXT, attribute_backend TEXT,
                attribute_metadata TEXT, crop_path TEXT, bbox_x INTEGER, bbox_y INTEGER,
                bbox_w INTEGER, bbox_h INTEGER);
            INSERT INTO appearance_log (id,event_id,timestamp,camera_id,track_id,upper_color)
            VALUES (42,'legacy-event',1000,'cam1',9223372036854775807,'blue');
        """)
    for _ in range(2):
        with AppearanceLog(path) as log:
            old = log.search(camera_id="cam1")[0]
            assert old["id"] == 42 and old["event_id"] == "legacy-event"
            assert old["track_id"] == 2**63 - 1
            assert old["track_id_text"] == str(2**63 - 1)
            assert old["upper_color"] == "blue"
    with AppearanceLog(path) as log:
        assert log.insert(camera_id="cam2", track_id=2**64 - 1, timestamp=2000)
        assert log.search(camera_id="cam2")[0]["id"] > 42
        assert log.count() == 2


def test_failed_write_does_not_suppress_valid_retry(tmp_path):
    with AppearanceLog(str(tmp_path / "appearance.db")) as log:
        assert not log.insert(camera_id="cam1", track_id=2**63, timestamp=1000,
                              attribute_metadata={"bad": object()})
        assert log.insert(camera_id="cam1", track_id=2**63, timestamp=1001)
        assert log.count() == 1


def test_sqlite_rejection_allows_retry_without_waiting_for_cooldown(tmp_path):
    with AppearanceLog(str(tmp_path / "appearance.db")) as log:
        log._conn.execute("""CREATE TRIGGER reject_test BEFORE INSERT ON appearance_log
            BEGIN SELECT RAISE(ABORT, 'temporary test rejection'); END""")
        assert not log.insert(camera_id="cam1", track_id=2**64 - 1, timestamp=1000)
        assert log.count() == 0
        log._conn.execute("DROP TRIGGER reject_test")
        assert log.insert(camera_id="cam1", track_id=2**64 - 1, timestamp=1001)
        assert log.count() == 1


@pytest.mark.parametrize("track_id", [True, 3.5, "123", 2**64, -(2**63) - 1])
def test_invalid_track_id_never_silently_rounds_or_truncates(tmp_path, track_id):
    with AppearanceLog(str(tmp_path / "appearance.db")) as log:
        assert not log.insert(camera_id="cam1", track_id=track_id, timestamp=1000)
        assert log.count() == 0
