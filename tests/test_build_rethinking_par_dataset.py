import csv
import pickle

from scripts.datasets.build_rethinking_par_dataset import ATTRIBUTES, main


def test_build_rethinking_par_dataset_writes_compatible_pkl(tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    image_root.mkdir()
    manifest = tmp_path / "appearance_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "gender", "upper_color", "lower_color", "bag", "hat"],
        )
        writer.writeheader()
        writer.writerow({
            "image_path": str(image_root / "001.jpg"),
            "gender": "female",
            "upper_color": "black",
            "lower_color": "blue",
            "bag": "yes",
            "hat": "no",
        })
        writer.writerow({
            "image_path": str(image_root / "002.jpg"),
            "gender": "male",
            "upper_color": "white",
            "lower_color": "gray",
            "bag": "no",
            "hat": "yes",
        })

    output_pkl = tmp_path / "RAP2" / "dataset_all.pkl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_rethinking_par_dataset.py",
            "--manifest",
            str(manifest),
            "--image-root",
            str(image_root),
            "--output-pkl",
            str(output_pkl),
            "--val-ratio",
            "0.5",
        ],
    )

    assert main() == 0

    with output_pkl.open("rb") as handle:
        dataset = pickle.load(handle)

    assert dataset.root == str(image_root)
    assert dataset.image_name == ["001.jpg", "002.jpg"]
    assert dataset.label.shape == (2, len(ATTRIBUTES))
    assert dataset.attr_name[0] == "gender_female"
    assert dataset.label_idx.eval == list(range(len(ATTRIBUTES)))
    assert len(dataset.partition.trainval[0]) == 1
    assert len(dataset.partition.test[0]) == 1
