import csv
import json

from scripts.datasets.build_appearance_training_lists import main


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
    assert label_map["labels"][0] == {"index": 0, "field": "gender", "value": "female", "threshold": 0.5}
    assert summary["train_rows"] == 1
    assert summary["val_rows"] == 1
    assert summary["summary"]["gender_female"] == 1
