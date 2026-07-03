"""외형 색상 감사 스크립트 테스트."""

from __future__ import annotations

import json
import sqlite3

import cv2
import numpy as np

from scripts.ops.audit_appearance_colors import _build_report


def test_build_report_includes_color_metadata_stats(tmp_path):
    db_path = tmp_path / "appearances.db"
    crop_path = tmp_path / "crop.jpg"
    image = np.zeros((120, 60, 3), dtype=np.uint8)
    image[:60, :] = (255, 0, 0)
    image[60:, :] = (0, 0, 0)
    assert cv2.imwrite(str(crop_path), image)

    metadata = {
        "color_sources": {"upper_color": "pa100k_sgie", "lower_color": "lab"},
        "color_candidates": {
            "upper_color": {
                "selected": "blue",
                "hsv_color": "black",
                "hsv_ratio": 0.18,
                "lab_color": "blue",
            },
            "lower_color": {
                "selected": "black",
                "hsv_color": "black",
                "hsv_ratio": 0.92,
                "lab_color": "black",
            },
        },
    }

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE appearance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                track_id INTEGER,
                attribute_backend TEXT,
                upper_color TEXT,
                lower_color TEXT,
                crop_path TEXT,
                timestamp REAL,
                attribute_metadata TEXT
            )
            """
        )
        conn.execute(
            """INSERT INTO appearance_log
               (camera_id, track_id, attribute_backend, upper_color, lower_color,
                crop_path, timestamp, attribute_metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "cam01",
                7,
                "pa100k_sgie",
                "blue",
                "black",
                str(crop_path),
                1000.0,
                json.dumps(metadata),
            ),
        )

    report = _build_report(db_path, limit=10, backend=None)

    assert report["checked_rows"] == 1
    assert report["metadata"]["rows_with_metadata"] == 1
    assert report["metadata"]["source_counts"]["upper_color"]["pa100k_sgie"] == 1
    assert report["metadata"]["hsv_lab_disagreements"]["upper_color"] == 1
    assert report["metadata"]["model_overrides"]["upper_color"] == 1
