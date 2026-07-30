from __future__ import annotations

from pathlib import Path

import pytest

from scripts.datasets.prepare_temporal_fall_clips import (
    _clip_window_seconds,
    _manifest_video_path,
    _resolve_video_backend,
    _select_rows_by_positions,
    _select_rows_by_scene_ids,
)


def test_clip_window_seconds_uses_annotated_fall_with_margin() -> None:
    start_seconds, duration_seconds = _clip_window_seconds(
        {
            "fall_start_frame": 120,
            "fall_end_frame": 180,
            "scene_length": 600,
        },
        fps=30.0,
        margin_frames=30,
    )

    assert start_seconds == 3.0
    assert duration_seconds == 4.0


def test_clip_window_seconds_clamps_to_video_bounds() -> None:
    start_seconds, duration_seconds = _clip_window_seconds(
        {
            "fall_start_frame": 10,
            "fall_end_frame": 590,
            "scene_length": 600,
        },
        fps=30.0,
        margin_frames=30,
    )

    assert start_seconds == 0.0
    assert duration_seconds == 20.0


def test_clip_window_seconds_rejects_missing_annotation() -> None:
    with pytest.raises(ValueError, match="valid fall frame annotation"):
        _clip_window_seconds(
            {
                "fall_start_frame": 0,
                "fall_end_frame": 0,
                "scene_length": 600,
            },
            fps=30.0,
            margin_frames=30,
        )


def test_select_rows_by_positions_keeps_quota_and_unique_groups() -> None:
    rows = [
        {
            "scene_id": f"scene-{index}",
            "scene_group": f"group-{index}",
            "scene_position": position,
            "is_fall": True,
        }
        for index, position in enumerate(
            ["복도", "복도", "복도", "병실", "병실", "병실"]
        )
    ]

    selected = _select_rows_by_positions(
        rows,
        positions=("복도", "병실"),
        per_position=2,
    )

    assert [row["scene_id"] for row in selected] == [
        "scene-0",
        "scene-1",
        "scene-3",
        "scene-4",
    ]


def test_select_rows_by_positions_rejects_insufficient_rows() -> None:
    with pytest.raises(ValueError, match="병실.*required 2, found 1"):
        _select_rows_by_positions(
            [
                {
                    "scene_id": "scene-1",
                    "scene_group": "group-1",
                    "scene_position": "병실",
                    "is_fall": True,
                }
            ],
            positions=("병실",),
            per_position=2,
        )


def test_resolve_video_backend_falls_back_to_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)

    assert _resolve_video_backend("auto") == "opencv"


def test_resolve_video_backend_prefers_gstreamer_before_opencv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/gst-launch-1.0"
        if name == "gst-launch-1.0"
        else None,
    )

    assert _resolve_video_backend("auto") == "gstreamer"


def test_manifest_video_path_is_relative_for_project_file() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert _manifest_video_path(project_root / "data/example.mp4") == "data/example.mp4"


def test_select_rows_by_scene_ids_keeps_requested_order() -> None:
    rows = [
        {"scene_id": "scene-a", "scene_group": "group-a", "is_fall": True},
        {"scene_id": "scene-b", "scene_group": "group-b", "is_fall": True},
    ]

    selected = _select_rows_by_scene_ids(rows, ("scene-b", "scene-a"))

    assert [row["scene_id"] for row in selected] == ["scene-b", "scene-a"]
