"""격리 재현 시험에서만 사용하는 DeepStream 최초 오류·프레임 이력 기록기."""

import json
import os
import re
import runpy
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

CUDA_SIGNATURE = re.compile(
    # 최초719도 기록하되 메시지 원문을 남겨 원700의 재현으로 잘못 합산하지 않는다.
    r"cudaErrorIllegalAddress|cudaErrorLaunchFailure|status\s*=\s*(?:700|719)|"
    r"err_no\s*:\s*(?:700|719)|"
    r"NVDSINFER_CUDA_ERROR|Failed in mem copy", re.IGNORECASE
)
SOURCE_TIMEOUT_SIGNATURE = re.compile(
    r"No data from source since last \d+ sec\. Trying reconnection"
)


def configure_gpu_core_dump(output):
    """격리 진단에서 선택한 경우에만 최초 GPU 예외의 덤프 저장을 준비한다."""
    if os.environ.get("AUDIT_CUDA_COREDUMP") != "1":
        return None
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 재실행의 PID/초 단위 시각이 같아도 기존 근거를 덮어쓰지 않는다.
    directory = Path(tempfile.mkdtemp(prefix="gpu_core_", dir=output))
    environment = {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION": "0",
        "CUDA_ENABLE_LIGHTWEIGHT_COREDUMP": "1",
        # 수동 요청은 정상 실행도 중단할 수 있어 수동 트리거를 활성화하지 않는다.
        "CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "0",
        "CUDA_COREDUMP_FILE": str(directory / "gpu.%p.%t.nvcudmp"),
    }
    # 상속된 생성 옵션이 위의 경량/CPU 덤프 정책을 바꾸지 않도록 명시한다.
    os.environ.pop("CUDA_COREDUMP_GENERATION_FLAGS", None)
    os.environ.update(environment)
    configuration = {"directory": str(directory), "mode": "lightweight_exception_only",
                     "environment": environment}
    (directory / "capture_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    return configuration


class StartupTrace:
    def __init__(self, output, capacity=4096):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.recent = deque(maxlen=capacity)
        self.first = []
        self.stages = {}
        self.native = deque(maxlen=64)
        self.sequence = 0
        self.captured = set()
        self.lock = threading.RLock()

    def record(self, kind, **values):
        # Gst.Buffer/NvDs 메타 객체는 보관하지 않고 수명과 무관한 숫자·문자열만 받는다.
        row = dict(kind=kind, monotonic=time.monotonic(), **values)
        with self.lock:
            self.sequence += 1
            row["sequence"] = self.sequence
            self.recent.append(row)
            if len(self.first) < 128:
                self.first.append(row)
            if kind == "buffer":
                key = values["element"] + ":" + values["pad"]
                stage = self.stages.setdefault(key, {"count": 0})
                stage.setdefault("first", row)
                stage.update(count=stage["count"] + 1, last=row)
            elif kind == "caps":
                key = values["element"] + ":" + values["pad"]
                self.stages.setdefault(key, {"count": 0}).update(configuration=row)

    def snapshot(self):
        with self.lock:
            return dict(
                event_count=self.sequence, capacity=self.recent.maxlen,
                first=list(self.first), recent=list(self.recent),
                stages={key: dict(value) for key, value in self.stages.items()},
                native_lines=list(self.native), captured=sorted(self.captured),
            )

    def capture(self, category, reason):
        # 일반 RTSP 오류와 CUDA 오류를 분리해 후속 연쇄 오류가 최초 근거를 덮지 못한다.
        if category not in ("cuda", "bus", "source_timeout"):
            raise ValueError("unsupported capture category")
        with self.lock:
            if category in self.captured:
                return False
            document = dict(reason=reason, detected_monotonic=time.monotonic(),
                            **self.snapshot())
            with (self.output / f"first_{category}.json").open("x") as handle:
                json.dump(document, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            self.captured.add(category)
            return True

    def native_line(self, line):
        # 원문 전체는 app.log에 두고, 진단 묶음의 줄 길이와 개수를 제한한다.
        with self.lock:
            self.native.append(dict(monotonic=time.monotonic(), line=line[:4096]))
        if CUDA_SIGNATURE.search(line):
            self.capture("cuda", dict(origin="native_log", message=line[:4096]))
        elif SOURCE_TIMEOUT_SIGNATURE.search(line):
            # 입력 경고 당시 상하류의 마지막 버퍼를 CUDA 오류와 별도로 보존한다.
            self.capture("source_timeout", dict(origin="native_log", message=line[:4096]))

    def checkpoint(self):
        temporary = self.output / "trace_latest.tmp"
        temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False))
        temporary.replace(self.output / "trace_latest.json")


class NativeLogTail:
    def __init__(self, trace, path):
        self.trace, self.path = trace, Path(path)
        self.offset, self.pending = 0, ""
        self.lock = threading.Lock()

    def poll(self):
        # bus 콜백과 감시 스레드가 같은 로그 구간을 중복 소비하지 않는다.
        with self.lock:
            self._poll()

    def _poll(self):
        if not self.path.exists():
            return
        # 파일 전체를 매번 읽지 않으며, 줄 경계에서 나뉜 CUDA 서명도 재조합한다.
        with self.path.open("rb") as handle:
            if self.path.stat().st_size < self.offset:
                self.offset, self.pending = 0, ""
            handle.seek(self.offset)
            chunk = handle.read(65536)
            self.offset = handle.tell()
        if not chunk:
            return
        lines = (self.pending + chunk.decode(errors="replace")).split("\n")
        self.pending = lines.pop()
        for line in lines:
            self.trace.native_line(line)
        # 개행 없이 종료된 native 오류도 다음 flush를 기다리지 않고 보존한다.
        if CUDA_SIGNATURE.search(self.pending) or SOURCE_TIMEOUT_SIGNATURE.search(self.pending):
            self.trace.native_line(self.pending)
        self.pending = self.pending[-4096:]


def install_hooks(trace, tail):
    from gi.repository import Gst

    import src.core.deepstream_processor as ds

    original_make = ds.DeepStreamProcessor._make_element
    original_bus = ds.DeepStreamProcessor._on_bus_message

    observed_pads = set()

    def attach_pad(element, pad, name):
        # 동적 pad와 기존 pad 조회가 겹쳐도 같은 버퍼를 두 번 집계하지 않는다.
        key = (name, pad.get_name(), hash(pad))
        if key in observed_pads:
            return
        observed_pads.add(key)
        direction = pad.get_name()
        state = {"caps": None, "properties": None}

        def probe(pad, info):
            if info.type & Gst.PadProbeType.EVENT_DOWNSTREAM:
                event = info.get_event()
                if event and event.type in (Gst.EventType.STREAM_START, Gst.EventType.FLUSH_START, Gst.EventType.FLUSH_STOP, Gst.EventType.EOS):
                    trace.record("stream_event", element=name, pad=direction, event=event.type.value_nick)
                return Gst.PadProbeReturn.OK
            if info.type & Gst.PadProbeType.BUFFER_LIST:
                buffers = info.get_buffer_list()
                count = buffers.length() if buffers else 0
                buffer = buffers.get(0) if count else None
            else:
                buffer = info.get_buffer()
                count = 1 if buffer is not None else 0
            if buffer is None:
                return Gst.PadProbeReturn.OK
            caps = pad.get_current_caps()
            caps_text = caps.to_string() if caps else None
            if state["caps"] != caps_text or state["properties"] is None:
                properties = {}
                for prop in ("copy-hw", "compute-hw", "nvbuf-memory-type", "batch-size", "interval", "config-file-path", "rtsp-reconnect-interval", "latency", "max-size-buffers", "max-size-bytes", "max-size-time"):
                    if element.find_property(prop):
                        properties[prop] = element.get_property(prop)
                factory = element.get_factory()
                trace.record("caps", element=name, pad=direction, factory=factory.get_name() if factory else None,
                             caps=caps_text, properties=properties)
                state.update(caps=caps_text, properties=properties)
            # 픽셀 복사·CUDA 호출 없이 buffer-list도 관측한다. count는 영상 프레임 수와 구분한다.
            trace.record("buffer", element=name, pad=direction, pts=int(buffer.pts),
                         dts=int(buffer.dts), size=buffer.get_size(), flags=int(buffer.get_flags()),
                         list_items=count, **{
                             prop: element.get_property(prop)
                             for prop in ("current-level-buffers", "current-level-bytes")
                             if element.find_property(prop)
                         })
            return Gst.PadProbeReturn.OK

        mask = Gst.PadProbeType.BUFFER | Gst.PadProbeType.BUFFER_LIST | Gst.PadProbeType.EVENT_DOWNSTREAM | Gst.PadProbeType.EVENT_FLUSH
        pad.add_probe(mask, probe)

    def observe_element(element, name, dynamic=False):
        if dynamic:
            element.connect("pad-added", lambda owner, pad: attach_pad(owner, pad, name))
        iterator = element.iterate_pads()
        while True:
            result, pad = iterator.next()
            if result == Gst.IteratorResult.RESYNC:
                iterator.resync()
            elif result == Gst.IteratorResult.OK:
                attach_pad(element, pad, name)
            else:
                break

    def watch_source(element, name):
        # nvurisrcbin 내부에서 늦게 생성되는 RTP/파서/디코더를 함께 관측한다.
        observed_elements = set()

        def child_added(owner, parent, child):
            factory = child.get_factory()
            factory_name = factory.get_name() if factory else ""
            identity = (child.get_name(), hash(child))
            if identity in observed_elements or factory_name not in {
                "rtspsrc", "rtph264depay", "rtph265depay", "h264parse", "h265parse", "nvv4l2decoder", "queue"
            }:
                return
            observed_elements.add(identity)
            observe_element(child, name + "/" + child.get_name(), dynamic=True)

        element.connect("deep-element-added", child_added)
        observe_element(element, name, dynamic=True)
        iterator = element.iterate_recurse()
        while True:
            result, child = iterator.next()
            if result == Gst.IteratorResult.RESYNC:
                iterator.resync()
            elif result == Gst.IteratorResult.OK:
                child_added(element, None, child)
            else:
                break

    def make(processor, factory, name):
        element = original_make(processor, factory, name)
        if factory == "nvurisrcbin":
            watch_source(element, name)
        # 출력 NVENC 경계도 기록하여 디코딩/추론 후 어디서 진행이 멎는지 구분한다.
        elif factory in {"nvstreammux", "nvinfer", "nvvideoconvert", "nvdsosd", "nvtracker", "nvv4l2h264enc", "nvv4l2h265enc"}:
            observe_element(element, name)
        return element

    def bus_message(processor, bus, message):
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            values = dict(element=message.src.get_name() if message.src else None,
                          message=str(error), debug=str(debug))
            trace.record("bus_error", **values)
            # bus 전달 전에 native 로그를 먼저 확인해 더 이른 오류 문구를 보존한다.
            tail.poll()
            trace.capture("bus", dict(origin="gst_bus", **values))
            if CUDA_SIGNATURE.search(str(error) + str(debug)):
                trace.capture("cuda", dict(origin="gst_bus", **values))
        elif message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            if SOURCE_TIMEOUT_SIGNATURE.search(str(warning)):
                # 무수신 경고와 CUDA 실패가 서로의 최초 기록을 덮지 못하게 분리한다.
                values = dict(element=message.src.get_name() if message.src else None,
                              message=str(warning), debug=str(debug))
                trace.record("source_timeout_warning", **values)
                tail.poll()
                trace.capture("source_timeout", dict(origin="gst_bus", **values))
        return original_bus(processor, bus, message)

    ds.DeepStreamProcessor._make_element = make
    ds.DeepStreamProcessor._on_bus_message = bus_message


def main():
    # 운영 환경에서 실수로 실행하지 않도록 격리 네트워크와 명시적인 허용을 요구한다.
    if os.environ.get("AUDIT_STARTUP_TRACE") != "1" or {
        path.name for path in Path("/sys/class/net").iterdir()
    } != {"lo"}:
        raise RuntimeError("explicit opt-in and network none required")
    os.umask(0o077)
    trace = StartupTrace("/output")
    trace.record("observer_start")
    # DeepStream/TensorRT를 불러오기 전에 설정해야 CUDA 초기화에 반영된다.
    core_capture = configure_gpu_core_dump(trace.output)
    if core_capture:
        trace.record("gpu_core_capture_configured", **core_capture)
    tail = NativeLogTail(trace, trace.output / "app.log")
    stopped = threading.Event()

    def monitor():
        last_checkpoint = 0.0
        while not stopped.wait(0.05):
            tail.poll()
            if time.monotonic() - last_checkpoint >= 1:
                trace.checkpoint()
                last_checkpoint = time.monotonic()

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        install_hooks(trace, tail)
        runpy.run_path("/app/main.py", run_name="__main__")
    finally:
        stopped.set()
        watcher.join(timeout=2)
        tail.poll()
        trace.checkpoint()


if __name__ == "__main__":
    main()
