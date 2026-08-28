import re

import numpy as np

from scripts.datasets.prepare_rap_attribute_manifest import (
    _group_key_from_image_name,
    _pkl_payload_to_rows,
    _write_review_html,
    canonicalize_row,
)


def test_canonicalize_row_maps_common_attribute_columns():
    row = {
        "image_path": "000001.jpg",
        "Female": "1",
        "Upper-Black": "1",
        "Lower-Blue": "1",
        "Backpack": "1",
        "Hat": "1",
    }

    result = canonicalize_row(row, image_root="images")

    assert result["image_path"] == "images/000001.jpg"
    assert result["gender"] == "female"
    assert result["upper_color"] == "black"
    assert result["lower_color"] == "blue"
    assert result["bag"] == "yes"
    assert result["hat"] == "yes"


def test_canonicalize_row_keeps_unknown_when_attribute_is_missing():
    row = {
        "filename": "000002.jpg",
        "lower_gray": "1",
    }

    result = canonicalize_row(row)

    assert result["image_path"] == "000002.jpg"
    assert result["gender"] == "unknown"
    assert result["upper_color"] == "unknown"
    assert result["lower_color"] == "gray"
    assert result["bag"] == "unknown"
    assert result["hat"] == "unknown"


def test_canonicalize_row_accepts_attributes_list_column():
    row = {
        "name": "000003.jpg",
        "attributes": "male;upper red;lower black;handbag",
    }

    result = canonicalize_row(row)

    assert result["gender"] == "male"
    assert result["upper_color"] == "red"
    assert result["lower_color"] == "black"
    assert result["bag"] == "yes"


def test_canonicalize_row_maps_rapv2_typos_and_extended_colors():
    row = {
        "image_path": "sample.png",
        "Femal": "1",
        "up-ColorRed": "1",
        "lb-ColorOrange": "1",
    }

    result = canonicalize_row(row)

    assert result["gender"] == "female"
    assert result["upper_color"] == "red"
    assert result["lower_color"] == "orange"


def test_group_key_keeps_frames_from_the_same_track_together():
    first = "CAM01-clip-tarid7-frame10-line1.png"
    second = "CAM01-clip-tarid7-frame99-line1.png"

    assert _group_key_from_image_name(first) == _group_key_from_image_name(second)


def test_pkl_payload_to_rows_uses_all_images_and_group_split():
    payload = {
        "image_name": [
            "CAM01-clip-tarid7-frame10-line1.png",
            "CAM01-clip-tarid7-frame99-line1.png",
            "CAM02-clip-tarid8-frame20-line1.png",
        ],
        "attr_name": ["Femal", "up-ColorRed", "lb-ColorOrange"],
        "label": np.asarray([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=np.int32),
        "partition": {
            "train": np.asarray([0]),
            "val": np.asarray([1]),
            "test": np.asarray([2]),
        },
    }

    rows = _pkl_payload_to_rows(
        payload,
        image_root="images",
        split_mode="group-hash",
        split_seed="rapv2-test",
    )

    assert len(rows) == 3
    assert rows[0]["image_path"].endswith("images/CAM01-clip-tarid7-frame10-line1.png")
    assert rows[0]["upper_color"] == "red"
    assert rows[0]["lower_color"] == "orange"
    assert rows[0]["split"] == rows[1]["split"]
    assert rows[0]["source_split"] == "train"
    assert rows[1]["source_split"] == "val"
    assert rows[2]["gender"] == "male"


def test_canonicalize_row_preserves_multicolor_labels_as_mixture():
    result = canonicalize_row(
        {
            "image_path": "multicolor.png",
            "ub-ColorBlack": "1",
            "ub-ColorWhite": "1",
            "lb-ColorBlue": "1",
        }
    )

    assert result["upper_color"] == "mixture"
    assert result["upper_color_labels"] == "black;white"
    assert result["lower_color"] == "blue"
    assert result["lower_color_labels"] == "blue"


def test_review_html_interleaves_upper_and_lower_rows(tmp_path):
    rows = [
        {
            "image_path": str(tmp_path / "black.png"),
            "upper_color": "black",
            "lower_color": "black",
        },
        {
            "image_path": str(tmp_path / "white.png"),
            "upper_color": "white",
            "lower_color": "white",
        },
    ]
    output = tmp_path / "review.html"

    _write_review_html(output, rows, per_color=1)

    document = output.read_text(encoding="utf-8")
    fields = re.findall(r"data-field='([^']+)'", document)
    assert fields == ["upper_color", "lower_color", "upper_color", "lower_color"]
