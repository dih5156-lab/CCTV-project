import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ops"
    / "label_fall_shadow_clips.py"
)
spec = importlib.util.spec_from_file_location("label_fall_shadow_clips", SCRIPT_PATH)
labeler = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["label_fall_shadow_clips"] = labeler
spec.loader.exec_module(labeler)


def _row(event_id: str, clip_path: str, camera_id: str = "camera_1") -> dict:
    return {
        "event_id": event_id,
        "camera_id": camera_id,
        "clip_path": clip_path,
        "label": None,
        "review_status": "unlabeled",
    }


def test_select_candidates_maps_container_path_and_filters_sample(tmp_path) -> None:
    clip = tmp_path / "camera.mp4"
    clip.write_bytes(b"video")
    sample = tmp_path / "sample.mp4"
    sample.write_bytes(b"video")
    rows = [
        _row("camera", "/app/data/fall_review_clips/camera.mp4"),
        _row("sample", "/app/data/fall_review_clips/sample.mp4", "sample_eval"),
    ]

    candidates = labeler.select_candidates(
        rows, camera="camera_1", include_sample_eval=False, clip_dir=tmp_path
    )

    assert [row["event_id"] for row in candidates] == ["camera"]
    assert candidates[0]["local_clip_path"] == str(clip)


def test_select_candidates_skips_needs_review(tmp_path) -> None:
    clip = tmp_path / "camera.mp4"
    clip.write_bytes(b"video")
    row = _row("camera", str(clip))
    row["review_status"] = "needs_review"

    candidates = labeler.select_candidates(
        [row], camera="camera_1", include_sample_eval=False, clip_dir=tmp_path
    )

    assert candidates == []


def test_apply_label_updates_only_matching_event(tmp_path) -> None:
    review_log = tmp_path / "review.jsonl"
    rows = [_row("first", "first.mp4"), _row("second", "second.mp4")]
    labeler.write_review_log_atomic(review_log, rows)

    labeler.apply_label(
        review_log, event_id="second", label="non_fall", review_status="reviewed"
    )

    saved = labeler.read_review_log(review_log)
    assert saved[0]["label"] is None
    assert saved[1]["label"] == "non_fall"
    assert saved[1]["review_status"] == "reviewed"


def test_read_review_log_reports_invalid_line(tmp_path) -> None:
    review_log = tmp_path / "review.jsonl"
    review_log.write_text(json.dumps({"event_id": "ok"}) + "\ninvalid\n")

    try:
        labeler.read_review_log(review_log)
    except ValueError as exc:
        assert "line 2" in str(exc)
    else:
        raise AssertionError("invalid JSON should fail")
