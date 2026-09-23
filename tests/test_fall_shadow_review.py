from unittest.mock import MagicMock

from src.core._fall_shadow_review import (
    FallShadowReviewConfig,
    FallShadowReviewRecorder,
)


def test_periodic_shadow_window_uses_temporal_candidate(tmp_path):
    falldata_aux = MagicMock()
    falldata_aux.verify_temporal_shadow.return_value = {
        "status": "ok",
        "confirmed": False,
        "temporal_compare_model": {"frames_with_pose": 42},
    }
    recorder = FallShadowReviewRecorder(
        FallShadowReviewConfig(
            review_log_path=tmp_path / "review.jsonl",
            clip_dir=tmp_path / "clips",
        ),
        falldata_aux=falldata_aux,
    )

    recorder.verify_and_write_aux_record(
        "camera-1",
        {
            "type": "fall_shadow_window",
            "metadata": {"falldata_aux_shadow_window": True},
        },
    )

    falldata_aux.verify_temporal_shadow.assert_called_once_with()
    falldata_aux.verify.assert_not_called()


def test_regular_fall_event_keeps_inline_verification(tmp_path):
    falldata_aux = MagicMock()
    falldata_aux.verify.return_value = {"status": "ok", "confirmed": False}
    recorder = FallShadowReviewRecorder(
        FallShadowReviewConfig(
            review_log_path=tmp_path / "review.jsonl",
            clip_dir=tmp_path / "clips",
        ),
        falldata_aux=falldata_aux,
    )

    recorder.verify_and_write_aux_record(
        "camera-1",
        {
            "type": "fall_detected",
            "metadata": {
                "falldata_pose_scope": {
                    "camera_id": "camera-1",
                    "object_id": 12,
                    "generation": 2,
                    "frame_index": 10,
                }
            },
        },
    )

    falldata_aux.verify.assert_called_once_with(
        camera_name="camera-1",
        pose_scope={
            "camera_id": "camera-1",
            "object_id": 12,
            "generation": 2,
            "frame_index": 10,
        },
    )
    falldata_aux.verify_temporal_shadow.assert_not_called()


def test_periodic_shadow_saves_the_exact_verified_clip(tmp_path):
    falldata_aux = MagicMock()
    falldata_aux.verify_temporal_shadow.return_value = {
        "status": "ok",
        "confirmed": False,
        "temporal_compare_model": {"frames_with_pose": 42},
    }
    falldata_aux.save_last_verified_clip.return_value = 42
    recorder = FallShadowReviewRecorder(
        FallShadowReviewConfig(
            review_log_path=tmp_path / "review.jsonl",
            clip_dir=tmp_path / "clips",
            save_clips=True,
        ),
        falldata_aux=falldata_aux,
    )

    _, record = recorder.verify_and_write_aux_record(
        "camera-1",
        {
            "type": "fall_shadow_window",
            "metadata": {"falldata_aux_shadow_window": True},
        },
    )

    falldata_aux.save_last_verified_clip.assert_called_once()
    falldata_aux.save_buffered_clip.assert_not_called()
    assert record["clip_frames"] == 42


def test_periodic_shadow_does_not_save_zero_pose_clip(tmp_path):
    falldata_aux = MagicMock()
    falldata_aux.verify_temporal_shadow.return_value = {
        "status": "insufficient_pose_frames",
        "confirmed": False,
        "temporal_compare_model": {"frames_with_pose": 0},
    }
    recorder = FallShadowReviewRecorder(
        FallShadowReviewConfig(
            review_log_path=tmp_path / "review.jsonl",
            clip_dir=tmp_path / "clips",
            save_clips=True,
        ),
        falldata_aux=falldata_aux,
    )

    _, record = recorder.verify_and_write_aux_record(
        "camera-1",
        {
            "type": "fall_shadow_window",
            "metadata": {"falldata_aux_shadow_window": True},
        },
    )

    falldata_aux.save_last_verified_clip.assert_not_called()
    falldata_aux.discard_last_verified_clip.assert_called_once_with()
    assert record["clip_path"] is None
    assert record["clip_frames"] is None
