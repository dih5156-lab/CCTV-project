"""Tests for appearance-color model shadow comparison helpers."""

from __future__ import annotations

import json

import numpy as np

from scripts.ops import shadow_compare_appearance_color_models as shadow


def test_select_rows_limits_each_track_and_keeps_newest_first():
    rows = [
        {"id": 5, "camera_id": "cam", "track_id": 1},
        {"id": 4, "camera_id": "cam", "track_id": 1},
        {"id": 3, "camera_id": "cam", "track_id": 1},
        {"id": 2, "camera_id": "cam", "track_id": 2},
        {"id": 1, "camera_id": "cam", "track_id": 3},
    ]

    selected = shadow._select_rows(rows, limit=4, max_per_track=2)

    assert [item["id"] for item in selected] == [5, 4, 2, 1]


def test_resolve_crop_path_translates_container_app_path(tmp_path):
    resolved = shadow._resolve_crop_path(
        "/app/data/runtime/appearance_crops/person.jpg",
        tmp_path,
    )

    assert resolved == tmp_path / "data/runtime/appearance_crops/person.jpg"


def test_person_box_reverses_scaled_context_crop_geometry():
    image = np.zeros((424, 195, 3), dtype=np.uint8)

    box = shadow._person_box(
        image,
        164,
        654,
        133,
        351,
        bbox_frame_width=1920,
        bbox_frame_height=1080,
        saved_frame_width=1280,
        saved_frame_height=720,
        context_ratio=0.6,
    )

    assert box == (53, 140, 142, 374)


def test_person_box_rejects_unexpected_crop_geometry():
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    box = shadow._person_box(
        image,
        164,
        654,
        133,
        351,
        bbox_frame_width=1920,
        bbox_frame_height=1080,
        saved_frame_width=1280,
        saved_frame_height=720,
        context_ratio=0.6,
    )

    assert box is None


def test_summarize_reports_model_and_runtime_agreement():
    records = [
        {
            "camera_id": "cam",
            "track_id": 1,
            "runtime_upper_color": "black",
            "baseline_color": "brown",
            "candidate_color": "black",
            "models_disagree": True,
        },
        {
            "camera_id": "cam",
            "track_id": 2,
            "runtime_upper_color": "gray",
            "baseline_color": "gray",
            "candidate_color": "gray",
            "models_disagree": False,
        },
        {
            "camera_id": "cam",
            "track_id": 2,
            "runtime_upper_color": "unknown",
            "baseline_color": "black",
            "candidate_color": "brown",
            "models_disagree": True,
        },
    ]

    summary = shadow._summarize(records)

    assert summary["evaluated"] == 3
    assert summary["unique_tracks"] == 2
    assert summary["model_disagreements"] == 2
    assert summary["model_disagreement_rate"] == 0.6667
    assert summary["runtime_comparable"] == 2
    assert summary["baseline_runtime_agreement"] == 0.5
    assert summary["candidate_runtime_agreement"] == 1.0
    assert summary["candidate_changes_to_black"] == 1
    assert summary["candidate_changes_from_black"] == 1


def test_upper_color_observations_handles_valid_and_invalid_metadata():
    metadata = json.dumps({"color_observations": {"upper_color": 4}})

    assert shadow._upper_color_observations(metadata) == 4
    assert shadow._upper_color_observations("not-json") == 0
    assert shadow._upper_color_observations(None) == 0


def test_write_html_adds_human_review_download(tmp_path):
    output = tmp_path / "review.html"
    shadow._write_html(
        output,
        [
            {
                "id": 7,
                "camera_id": "cam",
                "track_id": 3,
                "source_path": "sources/source_7.jpg",
                "roi_path": "rois/upper_7.jpg",
                "runtime_upper_color": "gray",
                "baseline_color": "gray",
                "baseline_confidence": 0.9,
                "candidate_color": "black",
                "candidate_confidence": 0.8,
                "models_disagree": True,
            }
        ],
    )

    document = output.read_text(encoding="utf-8")
    assert "data-id='7'" in document
    assert "sources/source_7.jpg" in document
    assert "원본에서 대상 사람을 확인" in document
    assert "appearance_color_shadow_labels.json" in document
    assert "upper_color: element.value || null" in document
    assert "lower_color: 'exclude'" in document
