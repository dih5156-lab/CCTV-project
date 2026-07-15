import json

from scripts.datasets.prepare_ai4c_color_classification import prepare_dataset


def test_prepare_dataset_filters_licenses_and_ambiguous_labels(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("black.jpg", "gray.jpg", "ambiguous.jpg", "blocked.jpg"):
        (images / name).write_bytes(name.encode())

    categories = [
        {"id": "black", "label": "black"},
        {"id": "grey", "label": "grey"},
        {"id": "red", "label": "red"},
    ]
    payload = {
        "images": [
            {"id": "1", "file_name": "black.jpg", "license": "https://creativecommons.org/licenses/by/4.0/"},
            {"id": "2", "file_name": "gray.jpg", "license": "https://creativecommons.org/publicdomain/zero/1.0/"},
            {"id": "3", "file_name": "ambiguous.jpg", "license": "https://creativecommons.org/publicdomain/mark/1.0/"},
            {"id": "4", "file_name": "blocked.jpg", "license": "inCopyright"},
        ],
        "categories": categories,
        "annotations": [
            {"image_id": "1", "category_id": "black"},
            {"image_id": "2", "category_id": "grey"},
            {"image_id": "3", "category_id": "black"},
            {"image_id": "3", "category_id": "red"},
            {"image_id": "4", "category_id": "red"},
        ],
    }
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "dataset"
    summary = prepare_dataset(
        annotations,
        [images],
        output,
        val_ratio=0.2,
        seed=42,
    )

    assert (output / "train" / "black" / "black.jpg").exists()
    assert (output / "train" / "gray" / "gray.jpg").exists()
    assert summary["train_total"] == 2
    assert summary["val_total"] == 0
    assert summary["skipped"] == {"ambiguous_color": 1, "license": 1}
