from scripts.datasets.prepare_rap_attribute_manifest import canonicalize_row


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
