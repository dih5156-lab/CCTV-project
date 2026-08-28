"""Tests for appearance-color hard-case manifest selection."""

from __future__ import annotations

import json
import sqlite3

from scripts.ops import build_appearance_color_review_manifest as builder


def _create_db(path, crop_path: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE appearance_log (
                id INTEGER PRIMARY KEY,
                event_id TEXT,
                timestamp REAL,
                camera_id TEXT,
                track_id INTEGER,
                upper_color TEXT,
                lower_color TEXT,
                attribute_backend TEXT,
                crop_path TEXT,
                attribute_metadata TEXT
            )
            """
        )
        rows = [
            (1, 10, "not_visible", 0),
            (2, 20, "color_yolov8n", 3),
            (3, 20, "color_yolov8n", 4),
            (4, 20, "color_yolov8n", 5),
            (5, 30, "hsv_lab_consensus", 2),
        ]
        for item_id, track_id, source, observations in rows:
            metadata = {
                "color_sources": {"lower_color": source},
                "color_observations": {"lower_color": observations},
                "color_candidates": {
                    "lower_color": {
                        "model_color": "black",
                        "model_confidence": 0.9,
                    }
                },
            }
            conn.execute(
                "INSERT INTO appearance_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item_id,
                    f"event-{item_id}",
                    float(item_id),
                    "camera_1",
                    track_id,
                    "blue",
                    "black",
                    "test",
                    crop_path,
                    json.dumps(metadata),
                ),
            )


def test_lower_review_manifest_filters_visibility_and_limits_each_track(tmp_path) -> None:
    crop = tmp_path / "person.jpg"
    crop.write_bytes(b"image")
    database = tmp_path / "appearances.db"
    _create_db(database, str(crop))

    manifest = builder.build_manifest(
        database,
        limit=10,
        focus_field="lower_color",
        min_color_observations=3,
        max_per_track=2,
    )

    assert [item["id"] for item in manifest["items"]] == [4, 3]
    assert manifest["min_color_observations"] == 3
    assert manifest["max_per_track"] == 2
