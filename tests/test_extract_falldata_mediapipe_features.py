import numpy as np

from scripts.datasets.extract_falldata_mediapipe_features import (
    FRAME_FEATURES,
    _pad_or_trim,
    _tail_start_frame,
)


def test_tail_start_frame_keeps_most_recent_limited_frames():
    assert _tail_start_frame(total_frames=600, max_frames=120) == 480


def test_tail_start_frame_starts_at_zero_when_limit_covers_video():
    assert _tail_start_frame(total_frames=100, max_frames=120) == 0
    assert _tail_start_frame(total_frames=100, max_frames=None) == 0


def test_pad_or_trim_stretches_short_sequence_to_600_frames():
    first = np.zeros(FRAME_FEATURES, dtype=np.float32)
    last = np.ones(FRAME_FEATURES, dtype=np.float32)

    sequence = _pad_or_trim(
        [first, last],
        sequence_transform="stretch",
    )

    assert sequence.shape == (600, FRAME_FEATURES)
    np.testing.assert_array_equal(sequence[0], first)
    np.testing.assert_array_equal(sequence[-1], last)
