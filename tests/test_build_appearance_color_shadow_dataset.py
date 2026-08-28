"""Tests for building a track-separated shadow color dataset."""

from __future__ import annotations

import json

import pytest

from scripts.ops import build_appearance_color_shadow_dataset as builder


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_dataset_adds_reviewed_rois_with_track_split(tmp_path):
    base = tmp_path / "base"
    (base / "train" / "black").mkdir(parents=True)
    (base / "train" / "gray").mkdir(parents=True)
    (base / "val" / "black").mkdir(parents=True)
    (base / "val" / "gray").mkdir(parents=True)
    (base / "train" / "black" / "base.jpg").write_bytes(b"base")
    shadow_dir = tmp_path / "shadow"
    (shadow_dir / "rois").mkdir(parents=True)
    (shadow_dir / "rois" / "upper_1.jpg").write_bytes(b"gray")
    (shadow_dir / "rois" / "upper_2.jpg").write_bytes(b"black")
    comparison = _write_json(
        shadow_dir / "comparison.json",
        {
            "items": [
                {
                    "id": 1,
                    "camera_id": "cam",
                    "track_id": 10,
                    "roi_path": "rois/upper_1.jpg",
                },
                {
                    "id": 2,
                    "camera_id": "cam",
                    "track_id": 20,
                    "roi_path": "rois/upper_2.jpg",
                },
            ]
        },
    )
    labels = _write_json(
        tmp_path / "labels.json",
        {
            "items": [
                {"id": 1, "upper_color": "gray"},
                {"id": 2, "upper_color": "black"},
            ]
        },
    )
    output = tmp_path / "output"

    summary = builder.build_dataset(
        base_dir=base,
        comparison_path=comparison,
        labels_path=labels,
        output_dir=output,
        validation_groups={"cam:10"},
        review_repeat=3,
    )

    assert (output / "val" / "gray" / "human_upper_1.jpg").read_bytes() == b"gray"
    assert (output / "train" / "black" / "human_upper_2.jpg").read_bytes() == b"black"
    assert (output / "train" / "black" / "human_upper_2_r3.jpg").read_bytes() == b"black"
    assert summary["base_counts"] == {"train": 1}
    assert summary["review_added"] == {"train:black": 3, "val:gray": 1}
    assert summary["review_repeat"] == 3
    archived = json.loads((output / "human_labels.json").read_text())
    assert {item["split"] for item in archived["items"]} == {"train", "val"}


def test_build_dataset_rejects_unsupported_human_color(tmp_path):
    base = tmp_path / "base"
    (base / "train" / "black").mkdir(parents=True)
    (base / "val" / "black").mkdir(parents=True)
    comparison = _write_json(tmp_path / "comparison.json", {"items": []})
    labels = _write_json(
        tmp_path / "labels.json",
        {"items": [{"id": 1, "upper_color": "other"}]},
    )

    with pytest.raises(ValueError, match="unsupported color"):
        builder.build_dataset(
            base_dir=base,
            comparison_path=comparison,
            labels_path=labels,
            output_dir=tmp_path / "output",
            validation_groups=set(),
        )
