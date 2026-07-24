from scripts.datasets.extract_falldata_mediapipe_features import _tail_start_frame


def test_tail_start_frame_keeps_most_recent_limited_frames():
    assert _tail_start_frame(total_frames=600, max_frames=120) == 480


def test_tail_start_frame_starts_at_zero_when_limit_covers_video():
    assert _tail_start_frame(total_frames=100, max_frames=120) == 0
    assert _tail_start_frame(total_frames=100, max_frames=None) == 0
