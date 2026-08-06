from types import SimpleNamespace

import numpy as np

from scripts.datasets.smoke_yolo_pose_fall_rf import (
    _candidate_window_frames,
    _fall_probability_from_classifier,
    _resolve_inference_config,
    _select_model_features,
    _tail_sampling_window,
)


def test_select_model_features_keeps_legacy_model_feature_order():
    summary = {
        "feature_names": ["legacy_b", "new_feature", "legacy_a"],
        "feature_vector": [2.0, 99.0, 1.0],
    }

    features = _select_model_features(
        summary=summary,
        model_feature_names=["legacy_a", "legacy_b"],
        expected_feature_count=2,
    )

    np.testing.assert_array_equal(features, np.asarray([[1.0, 2.0]], dtype=np.float32))


def test_resolve_inference_config_uses_bundle_defaults_and_cli_overrides():
    bundle = {
        "inference_config": {
            "max_frames": 48,
            "frame_stride": 3,
            "imgsz": 640,
            "confidence_threshold": 0.35,
        }
    }

    config = _resolve_inference_config(
        bundle,
        {
            "max_frames": None,
            "frame_stride": None,
            "imgsz": 320,
            "confidence_threshold": None,
        },
    )

    assert config == {
        "max_frames": 48,
        "frame_stride": 3,
        "imgsz": 320,
        "confidence_threshold": 0.35,
        "candidate_window_frames": 0,
        "candidate_window_seconds": 0.0,
    }


def test_fall_probability_uses_classifier_fall_class_label():
    classifier = SimpleNamespace(classes_=np.asarray([0, 1]))

    probability = _fall_probability_from_classifier(
        classifier,
        [[0.08, 0.92]],
    )

    assert probability == 0.92


def test_tail_sampling_window_limits_candidate_clip_to_recent_frames():
    assert _tail_sampling_window(total_frames=600, window_frames=181) == (420, 600)
    assert _tail_sampling_window(
        total_frames=600,
        window_frames=181,
        candidate_end_frame=405,
    ) == (225, 405)
    assert _tail_sampling_window(total_frames=120, window_frames=181) == (None, None)
    assert _tail_sampling_window(total_frames=600, window_frames=0) == (None, None)


def test_candidate_window_seconds_adapts_to_video_fps():
    config = {
        "candidate_window_frames": 181,
        "candidate_window_seconds": 3.0,
    }

    assert _candidate_window_frames(config, video_fps=59.94) == 181
    assert _candidate_window_frames(config, video_fps=30.0) == 91
