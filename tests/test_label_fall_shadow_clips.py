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


def test_classify_shadow_disagreement_detects_both_directions() -> None:
    primary_positive = _row("primary-positive", "positive.mp4")
    primary_positive.update(
        {
            "event_type": "fall_detected",
            "falldata_aux": {"status": "ok", "confirmed": False},
        }
    )
    shadow_positive = _row("shadow-positive", "negative.mp4")
    shadow_positive.update(
        {
            "event_type": "fall_shadow_window",
            "falldata_aux": {"status": "ok", "confirmed": True},
        }
    )

    assert labeler.classify_shadow_disagreement(primary_positive) == (
        "primary_fall_shadow_non_fall"
    )
    assert labeler.classify_shadow_disagreement(shadow_positive) == (
        "primary_non_fall_shadow_fall"
    )


def test_classify_shadow_disagreement_ignores_agreement_and_runtime_error() -> None:
    agreement = {
        "event_type": "fall_shadow_window",
        "falldata_aux": {"status": "ok", "confirmed": False},
    }
    runtime_error = {
        "event_type": "fall_detected",
        "falldata_aux": {"status": "error", "confirmed": False},
    }

    assert labeler.classify_shadow_disagreement(agreement) is None
    assert labeler.classify_shadow_disagreement(runtime_error) is None


def test_select_candidates_can_keep_only_shadow_disagreements(tmp_path) -> None:
    disagreement_clip = tmp_path / "disagreement.mp4"
    disagreement_clip.write_bytes(b"video")
    agreement_clip = tmp_path / "agreement.mp4"
    agreement_clip.write_bytes(b"video")
    disagreement = _row("disagreement", str(disagreement_clip))
    disagreement.update(
        {
            "event_type": "fall_shadow_window",
            "falldata_aux": {"status": "ok", "confirmed": True},
        }
    )
    agreement = _row("agreement", str(agreement_clip))
    agreement.update(
        {
            "event_type": "fall_shadow_window",
            "falldata_aux": {"status": "ok", "confirmed": False},
        }
    )

    candidates = labeler.select_candidates(
        [agreement, disagreement],
        camera="camera_1",
        include_sample_eval=False,
        only_disagreements=True,
        clip_dir=tmp_path,
    )

    assert [row["event_id"] for row in candidates] == ["disagreement"]
    assert candidates[0]["disagreement_type"] == "primary_non_fall_shadow_fall"


def test_select_candidates_can_include_threshold_boundary(tmp_path) -> None:
    boundary_clip = tmp_path / "boundary.mp4"
    boundary_clip.write_bytes(b"video")
    boundary = _row("boundary", str(boundary_clip))
    boundary.update(
        {
            "event_type": "fall_shadow_window",
            "falldata_aux": {
                "status": "ok",
                "confirmed": False,
                "fall_probability": 0.74,
                "threshold": 0.78,
            },
        }
    )

    candidates = labeler.select_candidates(
        [boundary],
        camera="camera_1",
        include_sample_eval=False,
        include_threshold_boundary=0.05,
        clip_dir=tmp_path,
    )

    assert [row["event_id"] for row in candidates] == ["boundary"]
    assert candidates[0]["review_reason"] == "threshold_boundary"


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


def test_apply_label_moves_clip_into_label_directory(tmp_path) -> None:
    pending_dir = tmp_path / "clips" / "pending"
    pending_dir.mkdir(parents=True)
    clip = pending_dir / "event.mp4"
    clip.write_bytes(b"video")
    review_log = tmp_path / "annotations" / "review.jsonl"
    labeler.write_review_log_atomic(review_log, [_row("event", str(clip))])

    labeler.apply_label(
        review_log,
        event_id="event",
        label="fall",
        review_status="reviewed",
        clip_dir=pending_dir,
        labeled_dir=tmp_path / "clips" / "labeled",
    )

    saved = labeler.read_review_log(review_log)[0]
    labeled_clip = tmp_path / "clips" / "labeled" / "fall" / "event.mp4"
    assert labeled_clip.read_bytes() == b"video"
    assert not clip.exists()
    assert saved["clip_path"] == str(labeled_clip)


def test_read_review_log_reports_invalid_line(tmp_path) -> None:
    review_log = tmp_path / "review.jsonl"
    review_log.write_text(json.dumps({"event_id": "ok"}) + "\ninvalid\n")

    try:
        labeler.read_review_log(review_log)
    except ValueError as exc:
        assert "line 2" in str(exc)
    else:
        raise AssertionError("invalid JSON should fail")


def test_build_review_document_contains_video_and_label_controls(tmp_path) -> None:
    clip = tmp_path / "candidate & one.mp4"
    clip.write_bytes(b"video")
    candidate = _row("event<&1", str(clip))
    candidate.update(
        {
            "local_clip_path": str(clip),
            "event_type": "fall_near_miss",
            "falldata_aux": {"fall_probability": 0.81234},
        }
    )

    document = labeler.build_review_document([candidate])

    assert "<video controls" in document
    assert clip.resolve().as_uri().replace("&", "&amp;") in document
    assert "event&lt;&amp;1" in document
    assert "0.812" in document
    assert "data-label='fall'" in document
    assert "data-label='non_fall'" in document
    assert "data-label='needs_review'" in document
    assert "fall_review_labels.json" in document
    assert "localStorage" in document
    assert "eventIds.has(eventId)" in document


def test_prepare_browser_clips_uses_h264_copy_without_changing_source(tmp_path) -> None:
    source = tmp_path / "pending" / "candidate.mp4"
    source.parent.mkdir()
    source.write_bytes(b"mpeg4-source")
    candidate = _row("event-1", str(source))
    candidate["local_clip_path"] = str(source)
    calls = []

    def convert(input_path: Path, output_path: Path) -> None:
        calls.append((input_path, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"h264-copy")

    prepared = labeler.prepare_browser_clips(
        [candidate], tmp_path / "browser", convert=convert
    )

    assert source.read_bytes() == b"mpeg4-source"
    assert calls == [(source, tmp_path / "browser" / "event-1.mp4")]
    assert prepared[0]["local_clip_path"] == str(
        tmp_path / "browser" / "event-1.mp4"
    )
    assert prepared[0]["source_clip_path"] == str(source)


def test_write_review_html_uses_relative_video_url_for_http(tmp_path) -> None:
    output = tmp_path / "review" / "fall_review.html"
    clip = output.parent / "browser_clips" / "event-1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"h264")
    candidate = _row("event-1", str(clip))
    candidate["local_clip_path"] = str(clip)

    labeler.write_review_html(output, [candidate])

    document = output.read_text(encoding="utf-8")
    assert "src='browser_clips/event-1.mp4'" in document
    assert "file://" not in document


def test_import_review_labels_applies_batch_and_creates_backup(tmp_path) -> None:
    pending_dir = tmp_path / "clips" / "pending"
    pending_dir.mkdir(parents=True)
    fall_clip = pending_dir / "fall.mp4"
    fall_clip.write_bytes(b"fall")
    review_clip = pending_dir / "review.mp4"
    review_clip.write_bytes(b"review")
    review_log = tmp_path / "annotations" / "review.jsonl"
    labeler.write_review_log_atomic(
        review_log,
        [_row("fall-event", str(fall_clip)), _row("review-event", str(review_clip))],
    )
    labels_path = tmp_path / "fall_review_labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "items": [
                    {"event_id": "fall-event", "label": "fall"},
                    {"event_id": "review-event", "label": "needs_review"},
                ],
            }
        )
    )

    summary = labeler.import_review_labels(
        review_log,
        labels_path,
        clip_dir=pending_dir,
        labeled_dir=tmp_path / "clips" / "labeled",
    )

    saved = {row["event_id"]: row for row in labeler.read_review_log(review_log)}
    labeled_clip = tmp_path / "clips" / "labeled" / "fall" / "fall.mp4"
    assert summary["updated"] == 2
    assert Path(summary["backup"]).is_file()
    assert labeled_clip.read_bytes() == b"fall"
    assert saved["fall-event"]["label"] == "fall"
    assert saved["fall-event"]["review_status"] == "reviewed"
    assert saved["fall-event"]["clip_path"] == str(labeled_clip)
    assert saved["review-event"]["label"] is None
    assert saved["review-event"]["review_status"] == "needs_review"
    assert review_clip.is_file()


def test_import_review_labels_rejects_unknown_event_before_backup(tmp_path) -> None:
    review_log = tmp_path / "review.jsonl"
    labeler.write_review_log_atomic(review_log, [_row("known", "known.mp4")])
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "items": [{"event_id": "unknown", "label": "non_fall"}],
            }
        )
    )

    try:
        labeler.import_review_labels(review_log, labels_path)
    except ValueError as exc:
        assert "unknown event_id" in str(exc)
    else:
        raise AssertionError("unknown event should fail")

    assert list(tmp_path.glob("*.bak")) == []


def test_import_review_labels_rejects_duplicate_review_event(tmp_path) -> None:
    review_log = tmp_path / "review.jsonl"
    labeler.write_review_log_atomic(
        review_log, [_row("duplicate", "one.mp4"), _row("duplicate", "two.mp4")]
    )
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "items": [{"event_id": "duplicate", "label": "needs_review"}],
            }
        )
    )

    try:
        labeler.import_review_labels(review_log, labels_path)
    except ValueError as exc:
        assert "duplicate event_id in review log" in str(exc)
    else:
        raise AssertionError("duplicate review event should fail")

    assert list(tmp_path.glob("*.bak")) == []
