import csv
import os
import pickle
import subprocess
from pathlib import Path

from scripts.datasets.build_rethinking_par_dataset import (
    ATTRIBUTES,
    build_dataset,
    main,
)


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

    assert dataset.root == str(image_root.resolve())
    assert dataset.image_name == ["001.jpg", "002.jpg"]
    assert dataset.label.shape == (2, len(ATTRIBUTES))
    assert dataset.attr_name[0] == "gender_female"
    assert dataset.label_idx.eval == list(range(len(ATTRIBUTES)))
    assert len(dataset.partition.trainval[0]) == 1
    assert len(dataset.partition.test[0]) == 1


def test_build_dataset_preserves_explicit_manifest_split():
    rows = [
        {"image_path": "validation.png", "split": "validation", "upper_color": "white", "lower_color": "black"},
        {"image_path": "train.png", "split": "train", "upper_color": "blue", "lower_color": "gray"},
    ]

    dataset = build_dataset(rows, image_root=None, val_ratio=0.9, seed=999)

    assert dataset.partition.train[0].tolist() == [1]
    assert dataset.partition.val[0].tolist() == [0]


def test_rethinking_loader_accepts_generated_namespace_dataset(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "data"
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "sample.png").write_bytes(b"not-read-during-init")
    rows = [{"image_path": "sample.png", "split": "train", "upper_color": "blue", "lower_color": "black"}]
    dataset = build_dataset(rows, image_root=image_root, val_ratio=0.2, seed=42)
    pkl_path = data_root / "RAP2/dataset_all.pkl"
    pkl_path.parent.mkdir(parents=True)
    with pkl_path.open("wb") as handle:
        pickle.dump(dataset, handle)

    environment = os.environ.copy()
    environment["PAR_DATA_ROOT"] = str(data_root)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(project_root / ".training_env/numpy1"),
            str(project_root),
            str(project_root / "Rethinking_of_PAR"),
        ]
    )
    code = "from configs import cfg; cfg.defrost(); cfg.DATASET.NAME='RAP2'; cfg.DATASET.LABEL='all'; cfg.freeze(); from dataset.pedes_attr.pedes import PedesAttr; print(len(PedesAttr(cfg, 'train')))"
    result = subprocess.run(
        [str(project_root / ".venv/bin/python"), "-c", code],
        cwd=project_root / "Rethinking_of_PAR",
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("1")
