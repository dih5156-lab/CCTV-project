from scripts.datasets.build_rapv2_reviewed_training_manifest import (
    _sanitize_base_rows,
    build_quality_report,
    convert_rap_row,
)

SUPPORTED_COLORS = {
    "black",
    "white",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "purple",
    "navy",
    "orange",
}


def test_quality_report_measures_preserved_source_labels():
    items = [
        {"image_path": "a.png", "field": "lower_color", "current_label": "red", "review_label": None},
        {"image_path": "b.png", "field": "lower_color", "current_label": "red", "review_label": "red"},
        {"image_path": "c.png", "field": "lower_color", "current_label": "red", "review_label": "brown"},
        {"image_path": "d.png", "field": "lower_color", "current_label": "red", "review_label": "exclude"},
    ]

    report = build_quality_report(items)

    assert report["lower_color.red"]["reviewed"] == 4
    assert report["lower_color.red"]["preserved"] == 2
    assert report["lower_color.red"]["keep_rate"] == 0.5


def test_convert_rap_row_applies_review_and_masks_low_quality_source_label():
    source = {
        "image_path": "/source/a.png",
        "split": "train",
        "source_index": "7",
        "group_id": "track-7",
        "upper_color": "orange",
        "lower_color": "green",
    }
    reviewed = {
        ("/source/a.png", "upper_color"): {
            "current_label": "orange",
            "review_label": "red",
        }
    }
    quality = {
        "upper_color.orange": {"keep_rate": 0.2},
        "lower_color.green": {"keep_rate": 0.4},
    }

    result = convert_rap_row(
        source,
        reviewed_items=reviewed,
        quality_report=quality,
        supported_colors=SUPPORTED_COLORS,
        minimum_keep_rate=0.6,
        container_image_root="/app/data/datasets/rapv2/RAP_dataset",
    )

    assert result["upper_color"] == "red"
    assert result["upper_color_defined"] is True
    assert result["lower_color_defined"] is False
    assert result["person_id"] == "rapv2:track-7"
    assert result["image_path"].endswith("/a.png")
    assert result["human_reviewed"] is True


def test_convert_rap_row_keeps_unreviewed_label_only_when_quality_passes():
    source = {
        "image_path": "/source/b.png",
        "split": "val",
        "source_index": "8",
        "group_id": "track-8",
        "upper_color": "blue",
        "lower_color": "orange",
    }
    quality = {
        "upper_color.blue": {"keep_rate": 1.0},
        "lower_color.orange": {"keep_rate": 0.15},
    }

    result = convert_rap_row(
        source,
        reviewed_items={},
        quality_report=quality,
        supported_colors=SUPPORTED_COLORS,
        minimum_keep_rate=0.6,
        container_image_root="/app/data/datasets/rapv2/RAP_dataset",
    )

    assert result["upper_color"] == "blue"
    assert result["upper_color_defined"] is True
    assert result["lower_color_defined"] is False
    assert result["human_reviewed"] is False


def test_sanitize_base_rows_masks_colors_outside_runtime_schema():
    rows = [
        {
            "upper_color": "pink",
            "upper_color_defined": True,
            "lower_color": "blue",
            "lower_color_defined": True,
        }
    ]

    sanitized = _sanitize_base_rows(rows, SUPPORTED_COLORS)

    assert sanitized[0]["upper_color"] == ""
    assert sanitized[0]["upper_color_defined"] is False
    assert sanitized[0]["lower_color"] == "blue"
    assert sanitized[0]["lower_color_defined"] is True
