from pathlib import Path

from scripts.ops.check_fall_test_dataset import check_dataset


def test_check_dataset_accepts_fall_and_notfall_files(tmp_path: Path):
    (tmp_path / "fall").mkdir()
    (tmp_path / "notfall").mkdir()
    (tmp_path / "fall" / "20260910_183000_forward_fall.mp4").touch()
    (tmp_path / "notfall" / "20260910_183500_walking.mp4").touch()

    result = check_dataset(tmp_path)

    assert result["valid"] is True
    assert result["file_count"] == 2


def test_check_dataset_rejects_invalid_pose_filename(tmp_path: Path):
    (tmp_path / "fall").mkdir()
    (tmp_path / "notfall").mkdir()
    (tmp_path / "fall" / "bad_name.mp4").touch()

    result = check_dataset(tmp_path)

    assert result["valid"] is False
    assert any("invalid filename" in error for error in result["errors"])
