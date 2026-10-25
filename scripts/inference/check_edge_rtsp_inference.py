"""SSH loopback 영상의 실제 DeepStream 추론을 제한된 시간 동안 검증한다."""

from __future__ import annotations

import argparse
import configparser
import json
import signal
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from src.core._deepstream_tensor_utils import layer_to_numpy, select_yolo_output
from src.core._yolo_postprocess import filter_yolo_candidates, nms_detections


def analyze_tensor(output, task):
    """원래 후처리의 후보 필터·NMS로 검출 수와 관절 수만 집계한다."""
    values = np.asarray(output)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("유효한 유한 YOLO 출력 행렬이 필요합니다.")
    rows = values.T if values.shape[0] < values.shape[1] else values
    if rows.shape[1] < 5 or (task == "pose" and rows.shape[1] != 56):
        raise ValueError("모델 작업과 출력 차원이 다릅니다.")
    rows, classes, scores = filter_yolo_candidates(
        rows, task=task, confidence_threshold=0.35, class_ids_filter=None,
    )
    detections = [
        {"box": (float(row[0] - row[2] / 2), float(row[1] - row[3] / 2),
                 float(row[2]), float(row[3])),
         "class_id": int(class_id), "confidence": float(score),
         "keypoints": int(np.count_nonzero(row[7:56:3] >= 0.5)) if task == "pose" else 0}
        for row, class_id, score in zip(rows, classes, scores)
    ]
    kept = nms_detections(detections, iou_threshold=0.45, max_detections=100)
    return {"detections": len(kept), "keypoints": sum(d["keypoints"] for d in kept),
            "class_counts": dict(Counter(str(d["class_id"]) for d in kept))}


def prepare_config(source, output, engine):
    """검증된 엔진만 사용하고 운영 설정·자동 엔진 생성을 배제한다."""
    source, output, engine = map(Path, (source, output, engine))
    if not engine.is_file() or output.resolve() == source.resolve():
        raise ValueError("별도 설정 경로와 기존 엔진이 필요합니다.")
    config = configparser.ConfigParser(interpolation=None)
    with source.open() as stream:
        config.read_file(stream)
    props = config["property"]
    for key in ("onnx-file", "model-file", "proto-file", "tlt-encoded-model",
                "operate-on-gie-id", "operate-on-class-ids"):
        props.pop(key, None)
    if props.get("labelfile-path"):
        props["labelfile-path"] = str((source.parent / props["labelfile-path"]).resolve())
    # 이 시험은 개별 모델의 전체 프레임 추론이며 SGIE/이벤트 수명주기 검증은 별도다.
    props.update({"model-engine-file": str(engine.resolve()), "batch-size": "1",
                  "process-mode": "1", "interval": "0", "output-tensor-meta": "1"})
    with output.open("x") as stream:
        config.write(stream)


def run(args):
    camera = json.loads(args.camera.read_text())
    url = camera["source"]
    if urlsplit(url).hostname != "127.0.0.1" or not 1 <= args.seconds <= 1800:
        raise ValueError("SSH loopback 입력과 1..1800초 시험만 허용합니다.")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_config = args.output / "infer.txt"
    prepare_config(args.infer_config, private_config, args.engine)
    import gi

    gi.require_version("Gst", "1.0")
    import pyds
    from gi.repository import Gst

    Gst.init(None)
    result = {"started_at": time.time(), "frames": 0, "tensor_frames": 0,
              "detections": 0, "keypoints": 0, "class_counts": {}, "max_gap_seconds": 0,
              "errors": [], "warnings": [], "task": args.task,
              "scope": "single_model_full_frame_inference_without_events"}
    result["stage_buffers"] = {"parsed": 0, "decoded": 0, "batched": 0}
    result["stage_caps"] = {}
    interrupted = False

    def stop(*unused):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    pipeline = Gst.parse_launch(
        "rtspsrc name=source protocols=tcp latency=200 tcp-timeout=5000000 ! "
        "application/x-rtp,media=video ! rtph264depay ! h264parse name=parser config-interval=-1 ! "
        "nvv4l2decoder name=decoder ! mux.sink_0 nvstreammux name=mux batch-size=1 width=1920 "
        "height=1080 live-source=true batched-push-timeout=33333 compute-hw=1 ! "
        "nvinfer name=infer ! fakesink name=sink sync=false async=false qos=false"
    )
    pipeline.get_by_name("source").set_property("location", url)
    pipeline.get_by_name("infer").set_property("config-file-path", str(private_config))

    def count_stage(pad, info, stage):
        result["stage_buffers"][stage] += 1
        if stage not in result["stage_caps"]:
            result["stage_caps"][stage] = pad.get_current_caps().to_string()
        return Gst.PadProbeReturn.OK

    for name, stage in (("parser", "parsed"), ("decoder", "decoded"), ("mux", "batched")):
        pipeline.get_by_name(name).get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, count_stage, stage)

    def probe(pad, info):
        try:
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
            if batch is None:
                raise RuntimeError("추론 출력에 batch metadata가 없습니다.")
            node = batch.frame_meta_list
            while node:
                frame = pyds.NvDsFrameMeta.cast(node.data)
                result["frames"] += 1
                meta_node = frame.frame_user_meta_list
                found = False
                while meta_node:
                    meta = pyds.NvDsUserMeta.cast(meta_node.data)
                    if meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        tensor = pyds.NvDsInferTensorMeta.cast(meta.user_meta_data)
                        layer = select_yolo_output(tensor, pyds)
                        if layer is None:
                            raise RuntimeError("YOLO 출력 레이어가 없습니다.")
                        values = layer_to_numpy(layer, pyds)
                        counts = analyze_tensor(values, args.task)
                        result["output_shape"] = list(values.shape)
                        for key in ("detections", "keypoints"):
                            result[key] += counts[key]
                        for key, value in counts["class_counts"].items():
                            result["class_counts"][key] = result["class_counts"].get(key, 0) + value
                        found = True
                        break
                    meta_node = meta_node.next
                if not found:
                    raise RuntimeError("프레임의 실제 추론 출력이 없습니다.")
                now = time.monotonic()
                if result["tensor_frames"]:
                    result["max_gap_seconds"] = max(result["max_gap_seconds"], now - result["last"])
                else:
                    result["first"] = now
                    result["caps"] = pad.get_current_caps().to_string()
                result["last"] = now
                result["tensor_frames"] += 1
                node = node.next
        except Exception as exc:
            if not result["errors"]:
                result["errors"].append(str(exc).replace(url, "[RTSP]"))
        return Gst.PadProbeReturn.OK

    pipeline.get_by_name("infer").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)
    bus = pipeline.get_bus()
    started = time.monotonic()
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            result["errors"].append("PLAYING transition failed")
        ready_deadline = time.monotonic() + 45
        next_sample = 0
        with (args.output / "progress.jsonl").open("x", buffering=1) as progress:
            while not interrupted and not result["errors"]:
                span = result.get("last", 0) - result.get("first", 0)
                if span >= args.seconds:
                    break
                message = bus.timed_pop_filtered(Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS)
                if message:
                    if message.type == Gst.MessageType.EOS:
                        result["errors"].append("Unexpected EOS")
                    else:
                        error, _ = message.parse_error() if message.type == Gst.MessageType.ERROR else message.parse_warning()
                        key = "errors" if message.type == Gst.MessageType.ERROR else "warnings"
                        result[key].append(str(error).replace(url, "[RTSP]"))
                now = time.monotonic()
                if (result["tensor_frames"] and now - result["last"] > 5) or (not result["tensor_frames"] and now > ready_deadline):
                    result["errors"].append("Inference stalled or did not start")
                if now >= next_sample:
                    progress.write(json.dumps(result) + "\n")
                    next_sample = now + 5
    finally:
        # SDK 종료가 지연돼 외부 timeout으로 종료돼도 원인·단계 계측을 먼저 보존한다.
        result.update({"pass": False, "cleanup_complete": False})
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        cleanup_started = time.monotonic()
        state = pipeline.set_state(Gst.State.NULL)
        result.update(cleanup_seconds=time.monotonic() - cleanup_started,
                      cleanup_complete=state != Gst.StateChangeReturn.FAILURE)
        # NULL 전에는 probe가 frames와 tensor_frames 사이를 갱신 중일 수 있다.
        # 스트리밍 스레드가 종료된 뒤 기간·계수·최종 판정을 함께 확정한다.
        span = result.get("last", 0) - result.get("first", 0)
        result.update(ended_at=time.time(), elapsed_seconds=time.monotonic() - started,
                      inference_span_seconds=span, requested_seconds=args.seconds,
                      inference_fps=(result["tensor_frames"] - 1) / span if span else 0,
                      interrupted=interrupted)
        result["pass"] = bool(not interrupted and not result["errors"] and span >= args.seconds
                              and result["tensor_frames"] == result["frames"]
                              and result["max_gap_seconds"] <= 5 and result["cleanup_complete"])
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--task", choices=["pose", "detect"], required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
