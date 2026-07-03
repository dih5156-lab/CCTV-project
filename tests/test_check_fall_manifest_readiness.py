"""Fall manifest readiness checker tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "health"
    / "check_fall_manifest_readiness.py"
)

spec = importlib.util.spec_from_file_location("check_fall_manifest_readiness", SCRIPT_PATH)
check_fall_manifest_readiness = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["check_fall_manifest_readiness"] = check_fall_manifest_readiness
spec.loader.exec_module(check_fall_manifest_readiness)


def test_build_summary_passes_balanced_scene_groups(tmp_path) -> None:
    rows = [
        {"scene_id": "fall_a_C1", "is_fall": True, "video_path": str(tmp_path / "a.mp4")},
        {"scene_id": "fall_b_C1", "is_fall": True, "video_path": str(tmp_path / "b.mp4")},
        {"scene_id": "non_a_C1", "is_fall": False, "video_path": str(tmp_path / "c.mp4")},
        {"scene_id": "non_b_C1", "is_fall": False, "video_path": str(tmp_path / "d.mp4")},
    ]
    for row in rows:
        Path(row["video_path"]).write_bytes(b"video")

    summary = check_fall_manifest_readiness.build_summary(rows, min_class_groups=2)

    assert summary["passed"] is True
    assert summary["group_counts"] == {"fall": 2, "non_fall": 2}
    assert summary["needed_group_counts"] == {"fall": 0, "non_fall": 0}


def test_build_summary_fails_when_non_fall_group_is_too_small(tmp_path) -> None:
    rows = [
        {"scene_id": "fall_a_C1", "is_fall": True, "video_path": str(tmp_path / "a.mp4")},
        {"scene_id": "fall_b_C1", "is_fall": True, "video_path": str(tmp_path / "b.mp4")},
        {"scene_id": "non_a_C1", "is_fall": False, "video_path": str(tmp_path / "c.mp4")},
        {"scene_id": "non_a_C2", "is_fall": False, "video_path": str(tmp_path / "d.mp4")},
    ]
    for row in rows:
        Path(row["video_path"]).write_bytes(b"video")

    summary = check_fall_manifest_readiness.build_summary(rows, min_class_groups=2)

    assert summary["passed"] is False
    assert summary["group_counts"]["non_fall"] == 1
    assert summary["needed_group_counts"]["non_fall"] == 1


def test_main_reports_manifest_payload(tmp_path, capsys, monkeypatch) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"scene_id": "fall_a_C1", "is_fall": True, "video_path": str(video)})
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_fall_manifest_readiness.py",
            "--manifest",
            str(manifest),
            "--min-class-groups",
            "1",
        ],
    )

    exit_code = check_fall_manifest_readiness.main()
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["manifest"] == str(manifest)
    assert output["group_counts"]["fall"] == 1
    assert output["group_counts"]["non_fall"] == 0
