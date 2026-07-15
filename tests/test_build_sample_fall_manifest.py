"""Fall manifest builder tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "datasets"
    / "build_sample_fall_manifest.py"
)

spec = importlib.util.spec_from_file_location("build_sample_fall_manifest", SCRIPT_PATH)
build_sample_fall_manifest = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["build_sample_fall_manifest"] = build_sample_fall_manifest
spec.loader.exec_module(build_sample_fall_manifest)


def _write_label(path: Path, *, scene_id: str, scene_is_fall: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {"scene_id": scene_id},
                "scene_info": {
                    "scene_IsFall": scene_is_fall,
                    "scene_length": 600,
                    "cam_num": 3,
                    "scene_cat_name": scene_is_fall,
                },
                "sensordata": {
                    "fall_start_frame": 10 if scene_is_fall == "낙상" else 0,
                    "fall_end_frame": 20 if scene_is_fall == "낙상" else 0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_build_manifest_supports_extracted_open_dataset_roots(tmp_path) -> None:
    source_root = tmp_path / "Training/01.원천데이터/extracted_TS/영상"
    label_root = tmp_path / "Training/02.라벨링데이터/영상"
    label_path = label_root / "Y/BY/00001_H_A_BY_C3/00001_H_A_BY_C3.json"
    video_path = source_root / "Y/BY/00001_H_A_BY_C3/00001_H_A_BY_C3.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"video")
    _write_label(label_path, scene_id="00001_H_A_BY_C3", scene_is_fall="낙상")

    rows = build_sample_fall_manifest.build_manifest(
        tmp_path,
        source_video_root=source_root,
        label_video_root=label_root,
        split="train",
    )

    assert len(rows) == 1
    assert rows[0]["scene_id"] == "00001_H_A_BY_C3"
    assert rows[0]["label"] == "fall"
    assert rows[0]["is_fall"] is True
    assert rows[0]["video_path"] == str(video_path)
    assert rows[0]["video_exists"] is True
    assert rows[0]["split"] == "train"


def test_build_manifest_keeps_default_sample_layout(tmp_path) -> None:
    sample_root = tmp_path / "Sample"
    label_path = sample_root / "02.라벨링데이터/영상/N/N/00002_H_A_N_C1/00002_H_A_N_C1.json"
    video_path = sample_root / "01.원천데이터/영상/N/N/00002_H_A_N_C1/00002_H_A_N_C1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"video")
    _write_label(label_path, scene_id="00002_H_A_N_C1", scene_is_fall="비낙상")

    rows = build_sample_fall_manifest.build_manifest(sample_root)

    assert len(rows) == 1
    assert rows[0]["label"] == "not_fall"
    assert rows[0]["is_fall"] is False
    assert rows[0]["video_path"] == str(video_path)
