import pickle
from types import SimpleNamespace

from scripts.datasets.patch_par_dataset_root import patch_dataset_root


def test_patch_par_dataset_root_updates_root_and_keeps_backup(tmp_path):
    pkl_path = tmp_path / "dataset_all.pkl"
    image_root = tmp_path / "images"
    image_root.mkdir()

    dataset = SimpleNamespace()
    dataset.root = "/old/root"
    dataset.image_name = ["001.jpg"]
    with pkl_path.open("wb") as handle:
        pickle.dump(dataset, handle)

    old_root, new_root = patch_dataset_root(pkl_path, image_root)

    with pkl_path.open("rb") as handle:
        patched = pickle.load(handle)

    assert old_root == "/old/root"
    assert new_root == str(image_root.resolve())
    assert patched.root == str(image_root.resolve())
    assert pkl_path.with_suffix(".pkl.bak").exists()
