import json

from src.bootstrap.runtime import load_camera_list


def test_load_camera_list_accepts_existing_file_uri(tmp_path):
    video_file = tmp_path / "sample video.mp4"
    video_file.write_bytes(b"")
    cameras_file = tmp_path / "cameras.json"
    cameras_file.write_text(
        json.dumps([{"id": "sample_eval", "source": video_file.as_uri()}]),
        encoding="utf-8",
    )

    cameras = load_camera_list(str(cameras_file))

    assert cameras == [{"id": "sample_eval", "source": video_file.as_uri()}]


def test_load_camera_list_skips_missing_file_uri(tmp_path):
    missing_video = tmp_path / "missing.mp4"
    cameras_file = tmp_path / "cameras.json"
    cameras_file.write_text(
        json.dumps([{"id": "sample_eval", "source": missing_video.as_uri()}]),
        encoding="utf-8",
    )

    cameras = load_camera_list(str(cameras_file))

    assert cameras == []
