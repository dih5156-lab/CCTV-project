import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "datasets"
    / "collect_fall_dataset.py"
)
spec = importlib.util.spec_from_file_location("collect_fall_dataset", SCRIPT_PATH)
collector = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["collect_fall_dataset"] = collector
spec.loader.exec_module(collector)


def test_initialize_dataset_creates_separated_directories(tmp_path) -> None:
    paths = collector.initialize_dataset(tmp_path)

    assert paths.review_log == tmp_path / "annotations" / "review.jsonl"
    assert paths.pending_dir.is_dir()
    assert paths.fall_dir.is_dir()
    assert paths.non_fall_dir.is_dir()
    assert paths.manifest_dir.is_dir()
    assert paths.review_log.is_file()


def test_collect_video_copies_to_pending_and_appends_review_record(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fall-video")
    dataset_root = tmp_path / "dataset"

    record = collector.collect_video(
        source,
        dataset_root=dataset_root,
        camera_id="gate/camera 1",
        source_name="manual",
        note="앉기 오탐 후보",
        created_at=datetime(2026, 7, 8, 1, 2, 3, tzinfo=timezone.utc),
    )

    saved_video = Path(record["clip_path"])
    assert saved_video.parent == dataset_root / "clips" / "pending"
    assert saved_video.read_bytes() == b"fall-video"
    assert record["event_id"].startswith("gate_camera_1_20260708T010203")
    assert record["label"] is None
    assert record["review_status"] == "unlabeled"
    review_rows = [
        json.loads(line)
        for line in (dataset_root / "annotations" / "review.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert review_rows == [record]


def test_collect_video_rejects_duplicate_content(tmp_path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"same-video")
    second.write_bytes(b"same-video")
    dataset_root = tmp_path / "dataset"

    collector.collect_video(first, dataset_root=dataset_root, camera_id="camera_1")

    try:
        collector.collect_video(second, dataset_root=dataset_root, camera_id="camera_1")
    except ValueError as exc:
        assert "duplicate video" in str(exc)
    else:
        raise AssertionError("duplicate content should be rejected")
