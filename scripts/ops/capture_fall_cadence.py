"""격리된 한 채널의 실제 낙상 경로를 관측하고 두 채널의 재생 시작을 맞춘다."""

import hashlib
import json
import os
import runpy
import signal
import sys
import threading
import time
from pathlib import Path

assert Path("/.dockerenv").exists()
assert {p.name for p in Path("/sys/class/net").iterdir()} == {"lo"}
manifest = json.loads(Path("/input/manifest.json").read_text())
assert int(os.environ["DS_PRIMARY_INTERVAL"]) == manifest["interval"]
assert (
    manifest["fps"] == 24 and 24 / (manifest["interval"] + 1) == manifest["target_hz"]
)
assert os.environ["DS_OUTPUT_MODE"] == "fakesink"
assert os.environ["FALLDATA_AUX_ENABLED"] == "false"
for path, key in [
    ("/app/models/fall/yolov8n-pose.engine", "model_sha256"),
    ("/app/config/deepstream/config_infer_primary.txt", "config_sha256"),
]:
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == manifest[key]

from src.core import deepstream_processor as dp  # noqa: E402
from src.core.events import EventType  # noqa: E402

out = Path("/output")
camera_id = manifest["camera_id"]
source_rows, processed_rows, event_rows = [], [], []
outcome = {
    "all_frames_processed": False,
    "camera_id": camera_id,
    "fps": 24,
    "errors": [],
}
saved = False
original_source = dp.DeepStreamProcessor._on_source_pad_added


def wait_for_peer(stop_event):
    # 모델 준비 시간이 달라도 두 영상의 첫 프레임부터 동일한 시간대에 비교한다.
    shared = Path("/coordination")
    (shared / f"{camera_id}.ready").write_text("ready")
    deadline = time.monotonic() + 60
    while not all((shared / f"replay_{i}.ready").exists() for i in (1, 2)):
        if stop_event.wait(0.05) or time.monotonic() >= deadline:
            raise RuntimeError("다른 채널의 재생 준비 시간 초과")
    clock = shared / "start_at"
    try:
        with clock.open("x") as stream:
            stream.write(str(time.monotonic() + 1.0))
    except FileExistsError:
        pass
    while True:
        try:
            return float(clock.read_text())
        except ValueError:
            if stop_event.wait(0.01) or time.monotonic() >= deadline:
                raise RuntimeError("동시 재생 시각 읽기 실패")


def source_added(self, source, pad, sinkpad):
    caps = pad.get_current_caps() or pad.query_caps(None)
    if (
        sinkpad.is_linked()
        or not caps
        or not caps.get_structure(0).get_name().startswith("video/")
    ):
        return original_source(self, source, pad, sinkpad)
    outcome["decode_caps"] = caps.to_string()
    anchor = None

    def pace(pad, info):
        nonlocal anchor
        buffer = info.get_buffer()
        if buffer is not None:
            pts = buffer.pts / dp.Gst.SECOND
            try:
                if anchor is None:
                    anchor = wait_for_peer(self.stop_event) - pts
                self.stop_event.wait(max(0, anchor + pts - time.monotonic()))
                source_rows.append(
                    {
                        "frame_index": len(source_rows),
                        "pts": pts,
                        "monotonic": time.monotonic(),
                    }
                )
            except Exception as exc:
                outcome["errors"].append(str(exc))
                self.stop_event.set()
                return dp.Gst.PadProbeReturn.DROP
        return dp.Gst.PadProbeReturn.OK

    pad.add_probe(dp.Gst.PadProbeType.BUFFER, pace)
    return original_source(self, source, pad, sinkpad)


original_probe = dp.DeepStreamProcessor._on_pad_probe


def probe(self, pad, info, camera):
    result = original_probe(self, pad, info, camera)
    buffer = info.get_buffer()
    batch = (
        dp.pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if buffer is not None
        else None
    )
    node = batch.frame_meta_list if batch is not None else None
    while node is not None:
        frame = dp.pyds.NvDsFrameMeta.cast(node.data)
        assert self._pad_to_camera[frame.source_id] == camera_id
        processed_rows.append(
            {
                "frame_index": int(frame.frame_num),
                "pts": int(frame.buf_pts) / dp.Gst.SECOND,
                "monotonic": time.monotonic(),
            }
        )
        try:
            node = node.next
        except StopIteration:
            break
    return result


original_queue = dp.DeepStreamProcessor._enqueue_queue_item


def queue(self, item, camera):
    result = original_queue(self, item, camera)
    if getattr(item, "event_type", None) == EventType.FALL_DETECTED:
        event_rows.append(
            {
                "stage": "publish_queue",
                "accepted": bool(result),
                "camera_id": camera,
                "frame_index": (item.metadata or {}).get("frame_num"),
                "object_id": item.object_id,
                "monotonic": time.monotonic(),
                "event_timestamp": item.timestamp,
            }
        )
    return result


original_start = dp.DeepStreamProcessor.start


def start(self, *args, **kwargs):
    outcome["flags"] = {
        camera: self._feature_flags_for_camera(camera) for camera in self._cameras
    }
    assert not self._falldata_aux.enabled
    policy = self._debouncer._config.events
    outcome["policy"] = {
        key: getattr(policy, key)
        for key in (
            "debounce_enabled",
            "fall_sustained_seconds",
            "fall_gap_reset_seconds",
            "fall_resend_cooldown",
        )
    }
    result = original_start(self, *args, **kwargs)

    def finish():
        target = manifest["expected_frames"]
        deadline = time.monotonic() + 100
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if len(source_rows) >= target and len(processed_rows) >= target:
                outcome["all_frames_processed"] = (
                    len(source_rows) == len(processed_rows) == target
                )
                time.sleep(1)
                break
            time.sleep(0.1)
        outcome["sigterm_requested"] = time.monotonic()
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=finish, daemon=True).start()
    return result


original_stop = dp.DeepStreamProcessor.stop


def stop(self, *args, **kwargs):
    global saved
    try:
        return original_stop(self, *args, **kwargs)
    finally:
        if not saved:
            saved = True
            # EOS가 종료 감시보다 먼저 도착해도 실제 프레임 수로 완료를 판정한다.
            outcome["all_frames_processed"] = (
                len(source_rows) == len(processed_rows) == manifest["expected_frames"]
            )
            outcome.update(
                stats=self.get_stats(),
                source=source_rows,
                processed=processed_rows,
                events=event_rows,
            )
            (out / "capture.json").write_text(
                json.dumps(outcome, indent=2, default=str) + "\n"
            )


dp.DeepStreamProcessor._on_source_pad_added = source_added
dp.DeepStreamProcessor._on_pad_probe = probe
dp.DeepStreamProcessor._enqueue_queue_item = queue
dp.DeepStreamProcessor.start = start
dp.DeepStreamProcessor.stop = stop
# 외부 전송은 차단된 loopback으로 제한하고 DB·이벤트는 개인 출력 경로에만 남긴다.
sys.argv = [
    "main.py",
    "--cameras",
    "/app/cameras.json",
    "--mqtt-broker",
    "127.0.0.1",
    "--mqtt-port",
    "9",
    "--mqtt-topic-prefix",
    "isolated/cadence24",
    "--device",
    "cuda",
    "--confidence",
    "0.5",
    "--pose-confidence",
    "0.5",
    "--frame-skip",
    str(manifest["interval"] + 1),
    "--fps",
    "24",
    "--debounce",
    "3.0",
    "--zone-detection",
    "--zones-config",
    "/app/zones_config.json",
    "--api-port",
    "8765",
]
runpy.run_path("/app/main.py", run_name="__main__")
