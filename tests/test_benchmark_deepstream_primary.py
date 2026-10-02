"""Safety and reporting contracts for the isolated, primary-only load test."""

import configparser
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def benchmark():
    # Missing implementation is an explicit RED assertion, not a collection error.
    path = Path("scripts/ops/benchmark_deepstream_primary.py")
    assert path.is_file(), "isolated benchmark implementation is missing"
    return importlib.import_module("scripts.ops.benchmark_deepstream_primary")


def test_networked_execution_is_rejected(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    with pytest.raises(ValueError, match="network"):
        benchmark().validate_run(video, 1, 5, 20, ["lo", "eth0"])


@pytest.mark.parametrize(
    "channels,warmup,seconds",
    [(0, 5, 20), (5, 5, 20), (1, -1, 20), (1, 5, 0), (1, 5, 121)],
)
def test_unbounded_or_invalid_load_is_rejected(tmp_path, channels, warmup, seconds):
    video = tmp_path / "video.mp4"
    video.touch()
    with pytest.raises(ValueError):
        benchmark().validate_run(video, channels, warmup, seconds, ["lo"])


def test_missing_video_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="video"):
        benchmark().validate_run(tmp_path / "missing.mp4", 1, 5, 20, ["lo"])


def test_isolated_local_replay_is_allowed(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    benchmark().validate_run(video, 4, 5, 20, ["lo"])


def test_config_does_not_modify_original_or_enable_engine_rebuild(tmp_path):
    source = tmp_path / "original.txt"
    original = "[property]\nmodel-engine-file=old.engine\nonnx-file=old.onnx\nbatch-size=8\nlabelfile-path=labels.txt\nnetwork-type=100\n"
    source.write_text(original)
    output = tmp_path / "private.txt"
    engine = tmp_path / "pose.engine"
    engine.touch()
    benchmark().prepare_config(source, output, engine, interval=1)
    actual = configparser.ConfigParser()
    actual.read(output)
    assert source.read_text() == original
    assert actual["property"]["model-engine-file"] == str(engine)
    assert actual["property"]["batch-size"] == "1"
    assert actual["property"]["interval"] == "1"
    assert actual["property"]["labelfile-path"] == str(tmp_path / "labels.txt")
    assert "onnx-file" not in actual["property"]


def test_report_uses_deltas_and_marks_partial_scope():
    before = {
        "replay_1": {
            "frames_seen": 300,
            "frames_inferred": 150,
            "frames_skipped": 150,
            "frames_inference_unknown": 0,
            "tensor_frames": 150,
        }
    }
    after = {
        "replay_1": {
            "frames_seen": 900,
            "frames_inferred": 450,
            "frames_skipped": 450,
            "frames_inference_unknown": 0,
            "tensor_frames": 450,
        }
    }
    report = benchmark().summarize(before, after, 20)
    assert report["cameras"]["replay_1"]["seen_fps"] == 30
    assert report["cameras"]["replay_1"]["inferred_fps"] == 15
    assert report["metric_evidence"] == "primary_tensor_meta"
    assert report["cameras"]["replay_1"]["tensor_fps"] == 15
    assert report["commercial_acceptance"] == "NOT_EVALUATED"
    assert report["scope"] == "decode_and_primary_only"


def test_counter_reset_is_not_reported_as_valid_throughput():
    before = {"replay_1": {"frames_seen": 100}}
    after = {"replay_1": {"frames_seen": 10}}
    with pytest.raises(ValueError, match="counter"):
        benchmark().summarize(before, after, 20)


def test_zero_measurement_window_is_rejected():
    with pytest.raises(ValueError, match="elapsed"):
        benchmark().summarize({}, {}, 0)


@pytest.mark.parametrize(
    "topology,expected",
    [
        ("independent", [{0: "replay_1"}, {0: "replay_2"}]),
        ("batched", [{0: "replay_1", 1: "replay_2"}]),
    ],
)
def test_branch_maps_preserve_camera_identity_in_both_topologies(topology, expected):
    assert benchmark().branch_camera_maps(2, topology) == expected


def test_unknown_topology_is_rejected():
    with pytest.raises(ValueError):
        benchmark().branch_camera_maps(2, "typo")


@pytest.mark.parametrize("batch_size", [0, 5])
def test_invalid_inference_batch_size_is_rejected(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch"):
        benchmark().prepare_config(None, None, None, interval=0, batch_size=batch_size)


def test_batched_config_uses_requested_size_without_modifying_source(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("[property]\nbatch-size=1\nmodel-engine-file=original.engine\n")
    engine = tmp_path / "candidate.engine"
    engine.touch()
    output = tmp_path / "private.txt"
    benchmark().prepare_config(source, output, engine, interval=0, batch_size=4)
    config = configparser.ConfigParser()
    config.read(output)
    assert config["property"]["batch-size"] == "4"
    assert "batch-size=1" in source.read_text()


def test_tensor_frames_are_routed_by_source_id_not_callback_branch():
    def node(data, next_node=None):
        return SimpleNamespace(data=data, next=next_node)

    tensor = SimpleNamespace(base_meta=SimpleNamespace(meta_type=10))
    frame_0 = SimpleNamespace(
        source_id=0, frame_user_meta_list=node(tensor, node(tensor))
    )
    frame_1 = SimpleNamespace(source_id=1, frame_user_meta_list=node(tensor))
    frame_2 = SimpleNamespace(source_id=1, frame_user_meta_list=None)
    batch = SimpleNamespace(frame_meta_list=node(frame_0, node(frame_1, node(frame_2))))
    pyds = SimpleNamespace(
        NvDsFrameMeta=SimpleNamespace(cast=lambda value: value),
        NvDsUserMeta=SimpleNamespace(cast=lambda value: value),
        NvDsMetaType=SimpleNamespace(NVDSINFER_TENSOR_OUTPUT_META=10),
    )
    assert benchmark().count_tensor_frames(batch, pyds, {0: "left", 1: "right"}) == {
        "left": 1,
        "right": 1,
    }


def test_replay_report_exposes_catchup_instead_of_implying_realtime_fps():
    before = {
        "replay_1": {
            "frames_inferred": 300,
            "tensor_frames": 300,
            "source_pts_ns": 10_000_000_000,
            "output_running_time_ns": 25_000_000_000,
        }
    }
    after = {
        "replay_1": {
            "frames_inferred": 1200,
            "tensor_frames": 1200,
            "source_pts_ns": 40_000_000_000,
            "output_running_time_ns": 45_000_000_000,
        }
    }
    camera = benchmark().summarize(before, after, 20)["cameras"]["replay_1"]
    assert camera["inferred_fps"] == 45
    assert camera.get("replay_timing") == {
        "source_timeline_seconds": 30.0,
        "source_to_wall_ratio": 1.5,
        "output_lag_start_seconds": 15.0,
        "output_lag_end_seconds": 5.0,
        "output_lag_change_seconds": -10.0,
        "scope": "source_pts_vs_pipeline_clock_not_end_to_end",
    }


@pytest.mark.parametrize("pts", [None, -1, 2**64 - 1])
def test_missing_or_invalid_pts_is_not_reported_as_zero_lag(pts):
    before = {"replay_1": {"source_pts_ns": pts, "output_running_time_ns": 0}}
    after = {
        "replay_1": {
            "source_pts_ns": 1_000_000_000,
            "output_running_time_ns": 1_000_000_000,
        }
    }
    camera = benchmark().summarize(before, after, 1)["cameras"]["replay_1"]
    assert "replay_timing" in camera
    assert camera["replay_timing"]["source_timeline_seconds"] is None
    assert camera["replay_timing"]["output_lag_start_seconds"] is None
    assert camera["replay_timing"]["output_lag_change_seconds"] is None


def test_replay_timing_keeps_valid_pts_zero_and_flags_backwards_timeline():
    before = {
        "replay_1": {
            "source_pts_ns": 1_000_000_000,
            "output_running_time_ns": 1_000_000_000,
        }
    }
    after = {"replay_1": {"source_pts_ns": 0, "output_running_time_ns": 2_000_000_000}}
    camera = benchmark().summarize(before, after, 1)["cameras"]["replay_1"]
    assert "replay_timing" in camera
    assert camera["replay_timing"]["source_timeline_seconds"] is None
    assert camera["replay_timing"]["output_lag_end_seconds"] == 2.0


def test_tensor_counter_captures_pts_separately_for_each_source():
    def node(data, next_node=None):
        return SimpleNamespace(data=data, next=next_node)

    tensor = SimpleNamespace(base_meta=SimpleNamespace(meta_type=10))
    frames = [
        SimpleNamespace(
            source_id=source_id, buf_pts=pts, frame_user_meta_list=node(tensor)
        )
        for source_id, pts in [(0, 0), (1, 1_000_000_000)]
    ]
    batch = SimpleNamespace(frame_meta_list=node(frames[0], node(frames[1])))
    pyds = SimpleNamespace(
        NvDsFrameMeta=SimpleNamespace(cast=lambda value: value),
        NvDsUserMeta=SimpleNamespace(cast=lambda value: value),
        NvDsMetaType=SimpleNamespace(NVDSINFER_TENSOR_OUTPUT_META=10),
    )
    # Catch missing instrumentation at the real batch traversal boundary.
    import inspect

    assert (
        "frame_timings" in inspect.signature(benchmark().count_tensor_frames).parameters
    )
    timings = {}
    counts = benchmark().count_tensor_frames(
        batch,
        pyds,
        {0: "left", 1: "right"},
        frame_timings=timings,
        running_time_ns=3_000_000_000,
    )
    assert counts == {"left": 1, "right": 1}
    assert timings == {
        "left": {"source_pts_ns": 0, "output_running_time_ns": 3_000_000_000},
        "right": {
            "source_pts_ns": 1_000_000_000,
            "output_running_time_ns": 3_000_000_000,
        },
    }
