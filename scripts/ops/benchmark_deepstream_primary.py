#!/usr/bin/env python3
"""Bounded offline decode + primary inference benchmark, NOT product acceptance.

Run ONLY in a Docker container with --network none, read-only model/config/input
mounts and a private writable output directory. No application runner, MQTT,
EdgeX, event publisher, tracker, secondary models, or UI is started here.
Independent mode uses one batch-1 mux/infer per decoder. Batched mode shares
one mux/infer across independently decoded sources with distinct source IDs.
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path


def validate_run(video, channels, warmup, seconds, network_names):
    if set(network_names) != {"lo"}:
        raise ValueError("network isolation required: use Docker --network none")
    if not Path(video).is_file():
        raise ValueError("video must be an existing local file")
    if channels not in (1, 2, 3, 4):
        raise ValueError("channels must be between 1 and 4")
    if not 0 <= warmup <= 30 or not 1 <= seconds <= 120:
        raise ValueError("warmup must be 0..30 and seconds must be 1..120")


def branch_camera_maps(channels, topology):
    if channels not in (1, 2, 3, 4):
        raise ValueError("channels must be between 1 and 4")
    cameras = [f"replay_{index + 1}" for index in range(channels)]
    if topology == "batched":
        return [dict(enumerate(cameras))]
    if topology == "independent":
        return [{0: camera_id} for camera_id in cameras]
    raise ValueError("unknown topology")


def prepare_config(source, output, engine, *, interval, batch_size=1):
    if batch_size not in (1, 2, 3, 4):
        raise ValueError("batch size must be between 1 and 4")
    source, output, engine = map(Path, (source, output, engine))
    if output.resolve() == source.resolve():
        raise ValueError("configuration output must be private")
    if not engine.is_file():
        raise ValueError("existing engine is required; automatic builds are disabled")
    config = configparser.ConfigParser(interpolation=None)
    with source.open() as handle:
        config.read_file(handle)
    properties = config["property"]
    for key in ("onnx-file", "model-file", "proto-file", "tlt-encoded-model"):
        properties.pop(key, None)
    for key in ("labelfile-path", "custom-lib-path"):
        if properties.get(key):
            properties[key] = str((source.parent / properties[key]).resolve())
    properties.update(
        {
            "model-engine-file": str(engine.resolve()),
            "batch-size": str(batch_size),
            "interval": str(interval),
            "output-tensor-meta": "1",
        }
    )
    with output.open("x") as handle:
        config.write(handle)


def count_tensor_frames(
    batch, pyds_module, pad_to_camera, *, frame_timings=None, running_time_ns=None
):
    counts = dict.fromkeys(pad_to_camera.values(), 0)
    node = batch.frame_meta_list
    while node is not None:
        frame = pyds_module.NvDsFrameMeta.cast(node.data)
        camera_id = pad_to_camera[frame.source_id]
        if frame_timings is not None:
            frame_timings[camera_id] = {
                "source_pts_ns": int(frame.buf_pts),
                "output_running_time_ns": running_time_ns,
            }
        meta_node = frame.frame_user_meta_list
        while meta_node is not None:
            meta = pyds_module.NvDsUserMeta.cast(meta_node.data)
            if (
                meta.base_meta.meta_type
                == pyds_module.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
            ):
                counts[camera_id] += 1
                break
            meta_node = meta_node.next
        node = node.next
    return counts


def summarize_replay_timing(start, end, elapsed):
    def seconds(value):
        # GST_CLOCK_TIME_NONE is uint64 max; zero is a valid source PTS.
        return (
            value / 1e9 if isinstance(value, int) and 0 <= value < 2**64 - 1 else None
        )

    start_pts, end_pts = (seconds(item.get("source_pts_ns")) for item in (start, end))
    start_clock, end_clock = (
        seconds(item.get("output_running_time_ns")) for item in (start, end)
    )
    timeline = (
        end_pts - start_pts
        if start_pts is not None and end_pts is not None and end_pts >= start_pts
        else None
    )
    start_lag = (
        start_clock - start_pts
        if start_clock is not None and start_pts is not None
        else None
    )
    end_lag = (
        end_clock - end_pts if end_clock is not None and end_pts is not None else None
    )
    return {
        "source_timeline_seconds": timeline,
        "source_to_wall_ratio": timeline / elapsed if timeline is not None else None,
        "output_lag_start_seconds": start_lag,
        "output_lag_end_seconds": end_lag,
        "output_lag_change_seconds": end_lag - start_lag
        if start_lag is not None and end_lag is not None
        else None,
        "scope": "source_pts_vs_pipeline_clock_not_end_to_end",
    }


def summarize(before, after, elapsed):
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("elapsed must be positive and finite")
    cameras = {}
    fields = (
        "frames_seen",
        "frames_inferred",
        "frames_skipped",
        "frames_inference_unknown",
        "tensor_frames",
    )
    for camera_id, end in after.items():
        start = before[camera_id]
        delta = {field: end.get(field, 0) - start.get(field, 0) for field in fields}
        if min(delta.values()) < 0:
            raise ValueError("counter reset during measurement")
        cameras[camera_id] = {
            **delta,
            "seen_fps": delta["frames_seen"] / elapsed,
            "inferred_fps": delta["frames_inferred"] / elapsed,
            "tensor_fps": delta["tensor_frames"] / elapsed,
            "last_frame_age_seconds": end.get("last_frame_age_seconds"),
            "replay_timing": summarize_replay_timing(start, end, elapsed),
        }
    return {
        "scope": "decode_and_primary_only",
        "commercial_acceptance": "NOT_EVALUATED",
        "metric_evidence": "primary_tensor_meta",
        "elapsed_seconds": elapsed,
        "cameras": cameras,
    }


def run(args):
    validate_run(
        args.video,
        args.channels,
        args.warmup,
        args.seconds,
        [path.name for path in Path("/sys/class/net").iterdir()],
    )
    if not Path("/.dockerenv").exists():
        raise ValueError("an isolated Docker container is required")
    args.output.mkdir(parents=True, exist_ok=False)
    camera_maps = branch_camera_maps(args.channels, args.topology)
    private_config = args.output / "primary.txt"
    prepare_config(
        args.config,
        private_config,
        args.engine,
        interval=args.interval,
        batch_size=len(camera_maps[0]),
    )

    import gi

    gi.require_version("Gst", "1.0")
    import pyds
    from gi.repository import GLib, Gst

    from src.core._deepstream_primary_metrics import PrimaryFrameMetrics

    Gst.init(None)
    loop = GLib.MainLoop()
    metrics = PrimaryFrameMetrics()
    camera_ids = [f"replay_{index + 1}" for index in range(args.channels)]
    metrics.reset(camera_ids)
    tensor_counts = dict.fromkeys(camera_ids, 0)
    frame_timings = {}
    lock = threading.Lock()
    errors, pipelines, samples = [], [], []
    observed_caps = {}
    state = {"ready_at": None, "baseline": None, "baseline_at": None, "finished": False}

    def stop(reason):
        errors.append(str(reason))
        loop.quit()

    def snapshot():
        with lock:
            values = metrics.snapshot()
            for camera_id, count in tensor_counts.items():
                values[camera_id]["tensor_frames"] = count
                values[camera_id].update(frame_timings.get(camera_id, {}))
            return values

    def probe(_pad, info, pad_to_camera):
        try:
            buffer = info.get_buffer()
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch is None:
                raise RuntimeError("primary output has no batch metadata")
            primary = _pad.get_parent_element()
            clock = primary.get_clock()
            running_time_ns = (
                int(clock.get_time() - primary.get_base_time())
                if clock is not None
                else None
            )
            with lock:
                metrics.record_batch(
                    batch,
                    pyds_module=pyds,
                    pad_to_camera=pad_to_camera,
                    primary_element=primary,
                )
                for camera_id, count in count_tensor_frames(
                    batch,
                    pyds,
                    pad_to_camera,
                    frame_timings=frame_timings,
                    running_time_ns=running_time_ns,
                ).items():
                    tensor_counts[camera_id] += count
        except Exception as exc:
            GLib.idle_add(stop, f"{pad_to_camera} probe: {exc}")
        return Gst.PadProbeReturn.OK

    def on_message(_bus, message, camera_id):
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            stop(f"{camera_id}: {error}: {debug}")
        elif message.type == Gst.MessageType.EOS:
            stop(f"{camera_id}: early end of input")

    def on_pad(_source, pad, target, camera_id):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps.get_structure(0).get_name().startswith("video/"):
            return
        observed_caps[camera_id] = caps.to_string()
        if pad.link(target.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            stop(f"{camera_id}: failed to link decoded video")

    def element(factory, name):
        value = Gst.ElementFactory.make(factory, name)
        if value is None:
            raise RuntimeError(f"unavailable element: {factory}")
        return value

    def tick():
        now = time.monotonic()
        values = snapshot()
        temperatures = []
        for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                temperatures.append(int(path.read_text()) / 1000)
            except (OSError, ValueError):
                continue
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name in ("MemAvailable", "SwapFree"):
                memory[name] = int(value.split()[0])
        sample = {
            "monotonic": now,
            "cameras": values,
            "temperature_max_c": max(temperatures, default=None),
            "memory_kib": memory,
        }
        samples.append(sample)
        print(json.dumps({"sample": sample}), flush=True)
        if temperatures and max(temperatures) >= 80:
            stop("precautionary stop: observed temperature >= 80C")
            return False
        if memory.get("MemAvailable", 0) < 8 * 1024 * 1024:
            stop("precautionary stop: available host memory < 8 GiB")
            return False
        if not all(value["frames_seen"] > 0 for value in values.values()):
            if now - started_at > 45:
                stop("no frames from every channel within 45 seconds")
                return False
            return True
        if state["ready_at"] is None:
            state["ready_at"] = now
        if any(value["last_frame_age_seconds"] > 5 for value in values.values()):
            stop("a replay branch stalled for over 5 seconds")
            return False
        if state["baseline"] is None and now - state["ready_at"] >= args.warmup:
            state["baseline"], state["baseline_at"] = values, now
        elif (
            state["baseline"] is not None and now - state["baseline_at"] >= args.seconds
        ):
            state["end"], state["end_at"] = values, now
            state["finished"] = True
            loop.quit()
            return False
        return True

    started_at = time.monotonic()
    try:
        for branch_index, camera_map in enumerate(camera_maps):
            branch_name = f"branch_{branch_index}"
            pipeline = Gst.Pipeline.new(branch_name)
            pipelines.append(pipeline)
            mux = element("nvstreammux", f"mux-{branch_name}")
            mux.set_property("batch-size", len(camera_map))
            mux.set_property("width", args.width)
            mux.set_property("height", args.height)
            mux.set_property("batched-push-timeout", 33333)
            mux.set_property("live-source", False)
            mux.set_property("enable-padding", True)
            infer = element("nvinfer", f"primary-{branch_name}")
            infer.set_property("config-file-path", str(private_config))
            sink = element("fakesink", f"sink-{branch_name}")
            sink.set_property("sync", False)
            sink.set_property("async", False)
            sink.set_property("qos", False)
            for item in (mux, infer, sink):
                pipeline.add(item)
            for source_id, camera_id in camera_map.items():
                source = element("nvurisrcbin", f"source-{camera_id}")
                source.set_property("uri", args.video.resolve().as_uri())
                source.set_property("disable-audio", True)
                source.set_property("drop-frame-interval", 0)
                source.set_property("file-loop", False)
                pacer = element("identity", f"pacer-{camera_id}")
                pacer.set_property("sync", True)
                pipeline.add(source)
                pipeline.add(pacer)
                source.connect("pad-added", on_pad, pacer, camera_id)
                mux_pad = mux.request_pad_simple(f"sink_{source_id}")
                if pacer.get_static_pad("src").link(mux_pad) != Gst.PadLinkReturn.OK:
                    raise RuntimeError("pacer to mux link failed")
            if not mux.link(infer) or not infer.link(sink):
                raise RuntimeError("inference path link failed")
            infer.get_static_pad("src").add_probe(
                Gst.PadProbeType.BUFFER, probe, camera_map
            )
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", on_message, branch_name)
        GLib.timeout_add(1000, tick)
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, stop, "terminated")
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, stop, "interrupted")
        for pipeline in pipelines:
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("pipeline could not start")
        loop.run()
    except Exception as exc:
        errors.append(str(exc))
    finally:
        for pipeline in pipelines:
            pipeline.set_state(Gst.State.NULL)

    report = {
        "scope": "decode_and_primary_only",
        "commercial_acceptance": "NOT_EVALUATED",
        "finished": state["finished"],
        "errors": errors,
        "channels": args.channels,
        "interval": args.interval,
        "mux_resolution": [args.width, args.height],
        "topology": args.topology,
        "inference_batch_size": len(camera_maps[0]),
        "observed_decode_caps": observed_caps,
        "samples": samples,
    }
    if state["finished"]:
        report.update(
            summarize(
                state["baseline"], state["end"], state["end_at"] - state["baseline_at"]
            )
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "samples"}),
        flush=True,
    )
    return 0 if state["finished"] and not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument(
        "--topology", choices=("independent", "batched"), default="independent"
    )
    parser.add_argument("--interval", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--width", type=int, choices=(1280, 1920), default=1920)
    parser.add_argument("--height", type=int, choices=(720, 1080), default=1080)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
