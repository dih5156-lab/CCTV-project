import csv
import json

from scripts.datasets.build_appearance_training_lists import _split_rows, main


def test_build_appearance_training_lists_writes_split_files(tmp_path, monkeypatch):
    manifest = tmp_path / "appearance_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "gender", "upper_color", "lower_color", "bag", "hat"],
        )
        writer.writeheader()
        writer.writerow({
            "image_path": "images/001.jpg",
            "gender": "female",
            "upper_color": "black",
            "lower_color": "blue",
            "bag": "yes",
            "hat": "no",
        })
        writer.writerow({
            "image_path": "images/002.jpg",
            "gender": "male",
            "upper_color": "white",
            "lower_color": "gray",
            "bag": "no",
            "hat": "yes",
        })

    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_appearance_training_lists.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--val-ratio",
            "0.5",
            "--seed",
            "1",
        ],
    )

    assert main() == 0

    train_lines = (output_dir / "train_list.txt").read_text(encoding="utf-8").splitlines()
    val_lines = (output_dir / "val_list.txt").read_text(encoding="utf-8").splitlines()
    label_map = json.loads((output_dir / "appearance_label_map.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(train_lines) == 1
    assert len(val_lines) == 1
    assert train_lines[0].startswith("images/")
    assert len(train_lines[0].split()) == 1 + len(label_map["labels"])
    assert len(label_map["labels"]) == 25
    assert label_map["labels"][0] == {"index": 0, "field": "gender", "value": "female", "threshold": 0.5}
    assert any(label["field"] == "upper_color" and label["value"] == "navy" for label in label_map["labels"])
    assert any(label["field"] == "upper_color" and label["value"] == "orange" for label in label_map["labels"])
    assert summary["train_rows"] == 1
    assert summary["val_rows"] == 1
    assert summary["summary"]["gender_female"] == 1


def test_build_appearance_training_lists_preserves_manifest_split(tmp_path, monkeypatch):
    manifest = tmp_path / "appearance_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "person_id", "split", "gender", "upper_color", "lower_color", "bag", "hat"],
        )
        writer.writeheader()
        writer.writerow({"image_path": "images/train.png", "person_id": "H1", "split": "train", "gender": "male", "upper_color": "other", "lower_color": "black", "bag": "no", "hat": "no"})
        writer.writerow({"image_path": "images/validation.png", "person_id": "H2", "split": "validation", "gender": "female", "upper_color": "white", "lower_color": "blue", "bag": "no", "hat": "no"})

    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["build_appearance_training_lists.py", "--manifest", str(manifest), "--output-dir", str(output_dir)],
    )

    assert main() == 0
    train_lines = (output_dir / "train_list.txt").read_text(encoding="utf-8").splitlines()
    val_lines = (output_dir / "val_list.txt").read_text(encoding="utf-8").splitlines()

    assert train_lines[0].startswith("images/train.png ")
    assert val_lines[0].startswith("images/validation.png ")


def test_split_rows_uses_explicit_manifest_split():
    rows = [
        {"image_path": "validation.png", "split": "validation"},
        {"image_path": "train.png", "split": "train"},
    ]

    train_rows, validation_rows = _split_rows(rows, val_ratio=0.9, seed=999)

    assert [row["image_path"] for row in train_rows] == ["train.png"]
    assert [row["image_path"] for row in validation_rows] == ["validation.png"]
