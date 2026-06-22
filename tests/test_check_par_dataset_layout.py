import pickle
from types import SimpleNamespace

from scripts.datasets.check_par_dataset_layout import check_dataset


def test_check_par_dataset_layout_detects_pa100k_ready(tmp_path):
    data_dir = tmp_path / "PA100k" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "000001.jpg").write_bytes(b"fake")
    (tmp_path / "PA100k" / "annotation.mat").write_bytes(b"fake")

    dataset = SimpleNamespace()
    dataset.root = str(data_dir)
    dataset.image_name = ["000001.jpg"]
    dataset.attr_name = ["Female"]
    with (tmp_path / "PA100k" / "dataset_all.pkl").open("wb") as handle:
        pickle.dump(dataset, handle)

    result = check_dataset(tmp_path, "PA100K")

    assert result["image_count"] == 1
    assert result["annotation_exists"] is True
    assert result["pkl"]["images"] == 1
    assert result["pkl_first_image_exists"] is True
    assert result["ready"] is True
