import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "datasets"
    / "build_field_fall_manifest.py"
)
spec = importlib.util.spec_from_file_location("build_field_fall_manifest", SCRIPT_PATH)
builder = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["build_field_fall_manifest"] = builder
spec.loader.exec_module(builder)


def test_build_field_rows_uses_date_as_scene_group(tmp_path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("one.mp4", "two.mp4"):
        (clips / name).write_bytes(b"video")
    rows = [
        {
            "event_id": f"event-{index}",
            "created_at": f"2026-07-03T01:00:0{index}+00:00",
            "camera_id": "camera_1",
            "clip_path": f"/app/data/fall_review_clips/{name}",
            "label": "non_fall",
            "review_status": "reviewed",
        }
        for index, name in enumerate(("one.mp4", "two.mp4"), start=1)
    ]

    result = builder.build_field_rows(rows, clip_dir=clips)

    assert len(result) == 2
    assert result[0]["scene_id"] == "field_camera_1_20260703_C1"
    assert result[1]["scene_id"] == "field_camera_1_20260703_C2"
    assert result[0]["is_fall"] is False
    assert result[0]["label"] == "not_fall"


def test_build_field_rows_ignores_unreviewed_and_missing_clips(tmp_path) -> None:
    rows = [
        {"label": None, "review_status": "needs_review", "clip_path": "skip.mp4"},
        {
            "label": "non_fall",
            "review_status": "reviewed",
            "clip_path": "missing.mp4",
        },
    ]

    assert builder.build_field_rows(rows, clip_dir=tmp_path) == []


def test_limit_rows_per_scene_group_samples_evenly() -> None:
    rows = [{"scene_id": f"field_camera_1_20260703_C{index}"} for index in range(1, 11)]

    selected = builder.limit_rows_per_scene_group(rows, 3)

    assert [row["scene_id"] for row in selected] == [
        "field_camera_1_20260703_C1",
        "field_camera_1_20260703_C5",
        "field_camera_1_20260703_C10",
    ]
