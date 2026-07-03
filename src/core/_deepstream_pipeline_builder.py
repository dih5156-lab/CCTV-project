"""Small helpers for assembling DeepStream GStreamer pipelines."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepStreamPipelineElements:
    streammux: Any
    nvinfer: Optional[Any]
    pphuman_infer: Optional[Any]
    helmet_infer: Optional[Any]
    tracker: Optional[Any]
    converter: Any
    osd: Any
    tee: Optional[Any]
    output_queue: Optional[Any]
    preview_elements: List[Any]
    output_elements: List[Any]

    def all_elements(self) -> List[Any]:
        elements = [self.streammux]
        elements.extend(
            element
            for element in (
                self.nvinfer,
                self.pphuman_infer,
                self.helmet_infer,
                self.tracker,
            )
            if element is not None
        )
        elements.extend([self.converter, self.osd])
        if self.tee is not None and self.output_queue is not None:
            elements.extend([self.tee, self.output_queue, *self.preview_elements])
        elements.extend(self.output_elements)
        return elements

    def topology(self) -> tuple[bool, bool, bool]:
        return (
            self.nvinfer is not None,
            self.helmet_infer is not None,
            self.pphuman_infer is not None,
        )


def link_deepstream_pipeline_path(
    elements: DeepStreamPipelineElements,
    *,
    link_or_raise: Callable[[Any, Any, Optional[str]], None],
    gst_module: Any,
    primary_probe_callback: Callable[[Any, Any, Any], Any],
    link_preview_branch: Callable[..., Any],
    pphuman_gie_id: int,
    pphuman_infer_config: Any,
) -> Any:
    """Link the main inference/output path and return the element to probe."""
    previous = elements.streammux
    probe_element = elements.streammux

    if elements.nvinfer is not None:
        link_or_raise(previous, elements.nvinfer, "nvstreammux -> nvinfer link 실패")
        previous = elements.nvinfer
        probe_element = elements.nvinfer
        if elements.pphuman_infer is not None:
            primary_srcpad = elements.nvinfer.get_static_pad("src")
            if primary_srcpad is None:
                raise RuntimeError("primary-infer src pad를 찾을 수 없습니다.")
            primary_srcpad.add_probe(
                gst_module.PadProbeType.BUFFER,
                primary_probe_callback,
                None,
            )

    if elements.pphuman_infer is not None:
        link_or_raise(
            previous,
            elements.pphuman_infer,
            f"{previous.get_name()} -> pphuman-infer link 실패",
        )
        previous = elements.pphuman_infer
        probe_element = elements.pphuman_infer
        logger.info(
            "PP-Human SGIE 파이프라인 연결 완료: gie_id=%d, config=%s",
            pphuman_gie_id,
            pphuman_infer_config,
        )

    if elements.helmet_infer is not None:
        link_or_raise(
            previous,
            elements.helmet_infer,
            f"{previous.get_name()} -> helmet-infer link 실패",
        )
        previous = elements.helmet_infer
        probe_element = elements.helmet_infer

    if elements.tracker is not None:
        link_or_raise(
            previous,
            elements.tracker,
            f"{previous.get_name()} -> nvtracker link 실패",
        )
        previous = elements.tracker

    link_or_raise(
        previous,
        elements.converter,
        f"{previous.get_name()} -> nvvideoconvert link 실패",
    )
    link_or_raise(elements.converter, elements.osd, "nvvideoconvert -> nvdsosd link 실패")
    previous = elements.osd

    if elements.tee is not None and elements.output_queue is not None:
        previous = link_preview_branch(
            osd=elements.osd,
            tee=elements.tee,
            output_queue=elements.output_queue,
            preview_elements=elements.preview_elements,
        )

    for element in elements.output_elements:
        link_or_raise(previous, element, None)
        previous = element

    return probe_element


def create_h264_encoder_elements(
    *,
    make_element: Callable[[str, str], Any],
    env_int: Callable[[str, int], int],
    set_optional_property: Callable[[Any, str, Any], None],
    gst_module: Any,
) -> List[Any]:
    converter = make_element("nvvideoconvert", "h264-nvvidconv")
    capsfilter = make_element("capsfilter", "h264-caps")
    encoder_name = os.environ.get("DS_H264_ENCODER", "nvv4l2h264enc").strip().lower()
    use_x264 = encoder_name in {"x264", "x264enc", "software"}
    encoder = make_element("x264enc" if use_x264 else "nvv4l2h264enc", "h264-encoder")
    parser = make_element("h264parse", "h264-parser")
    parsed_capsfilter = make_element("capsfilter", "h264-parsed-caps")

    width = env_int("DS_H264_WIDTH", 1280)
    height = env_int("DS_H264_HEIGHT", 720)
    memory = "" if use_x264 else "(memory:NVMM)"
    capsfilter.set_property(
        "caps",
        gst_module.Caps.from_string(
            f"video/x-raw{memory},format=NV12,width={width},height={height}"
        ),
    )

    bitrate = env_int("DS_H264_BITRATE", 6000000)
    iframe_interval = env_int("DS_H264_IFRAME_INTERVAL", 30)
    idr_interval = env_int("DS_H264_IDR_INTERVAL", iframe_interval)
    if use_x264:
        encoder.set_property("bitrate", max(1, bitrate // 1000))
        set_optional_property(encoder, "speed-preset", "ultrafast")
        set_optional_property(encoder, "tune", "zerolatency")
        set_optional_property(encoder, "key-int-max", iframe_interval)
        set_optional_property(encoder, "byte-stream", True)
        set_optional_property(encoder, "bframes", 0)
        set_optional_property(encoder, "b-adapt", False)
        set_optional_property(encoder, "ref", 1)
        set_optional_property(encoder, "cabac", False)
        set_optional_property(encoder, "aud", True)
        set_optional_property(encoder, "insert-vui", True)
        set_optional_property(encoder, "sliced-threads", True)
    else:
        encoder.set_property("bitrate", bitrate)
        set_optional_property(encoder, "maxperf-enable", True)
        set_optional_property(encoder, "insert-aud", True)
        set_optional_property(encoder, "insert-sps-pps", True)
        set_optional_property(encoder, "insert-vui", False)
        set_optional_property(encoder, "iframeinterval", iframe_interval)
        set_optional_property(encoder, "idrinterval", idr_interval)
        set_optional_property(encoder, "control-rate", 1)
        set_optional_property(encoder, "ratecontrol-enable", True)
        set_optional_property(encoder, "copy-timestamp", False)
        set_optional_property(encoder, "disable-cabac", True)
        set_optional_property(encoder, "num-B-Frames", 0)
        set_optional_property(encoder, "num-Ref-Frames", 1)
        set_optional_property(encoder, "profile", 0)
        # WebRTC does not support frame reordering. Jetson's encoder normally
        # advertises pic_order_cnt_type=2; the optional bitstream fixer needs
        # type 0 because it rewrites poc_lsb directly.
        poc_fix_enabled = os.environ.get(
            "DS_H264_POC_FIX_ENABLED", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        poc_type = 0 if poc_fix_enabled else env_int("DS_H264_POC_TYPE", 2)
        set_optional_property(encoder, "poc-type", poc_type)

    set_optional_property(parser, "disable-passthrough", True)
    set_optional_property(parser, "config-interval", -1)
    parsed_capsfilter.set_property(
        "caps",
        gst_module.Caps.from_string("video/x-h264,stream-format=byte-stream,alignment=au"),
    )
    return [converter, capsfilter, encoder, parser, parsed_capsfilter]


def create_preview_elements(
    *,
    make_element: Callable[[str, str], Any],
    env_int: Callable[[str, int], int],
    gst_module: Any,
    on_preview_sample: Callable[[Any], Any],
) -> List[Any]:
    queue = make_element("queue", "preview-queue")
    converter = make_element("nvvideoconvert", "preview-nvvidconv")
    capsfilter = make_element("capsfilter", "preview-caps")
    appsink = make_element("appsink", "preview-appsink")

    queue.set_property("leaky", 2)
    queue.set_property("max-size-buffers", 2)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)

    caps_parts = ["video/x-raw", "format=BGRx"]
    preview_width = env_int("DS_PREVIEW_WIDTH", 0)
    preview_height = env_int("DS_PREVIEW_HEIGHT", 0)
    if preview_width > 0 and preview_height > 0:
        caps_parts.extend([f"width={preview_width}", f"height={preview_height}"])
    capsfilter.set_property("caps", gst_module.Caps.from_string(",".join(caps_parts)))

    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1)
    appsink.set_property("drop", True)
    appsink.set_property("sync", False)
    appsink.connect("new-sample", on_preview_sample)
    return [queue, converter, capsfilter, appsink]


def create_output_elements(
    *,
    output_mode: str,
    make_element: Callable[[str, str], Any],
    set_optional_property: Callable[[Any, str, Any], None],
    env_int: Callable[[str, int], int],
    gst_module: Any,
    create_h264_encoder_elements_fn: Callable[[], List[Any]],
    poc_fixer_factory: Callable[[], Any],
) -> List[Any]:
    if output_mode in {"", "fake", "fakesink", "headless"}:
        sink = make_element("fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)
        return [sink]

    if output_mode in {
        "mpegts",
        "h264",
        "h264_mpegts",
        "h264-mpegts",
        "rtsp",
        "rtsp_publish",
        "rtsp-publish",
    }:
        h264_elements = create_h264_encoder_elements_fn()
        poc_fix_enabled = os.environ.get(
            "DS_H264_POC_FIX_ENABLED", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}

        poc_identity = None
        if poc_fix_enabled:
            poc_fixer = poc_fixer_factory()
            poc_identity = make_element("identity", "poc-fix-identity")
            poc_identity.set_property("signal-handoffs", True)
            poc_identity.set_property("silent", True)

            clock_time_none = getattr(gst_module, "CLOCK_TIME_NONE", -1)
            previous_pts: list = [clock_time_none]
            h264_fps = max(1, env_int("DS_H264_FPS", 30))
            poc_frame_ns = int(1_000_000_000 / h264_fps)

            def _poc_handoff(element: Any, buf: Any) -> None:
                size = buf.get_size()
                ok, minfo = buf.map(gst_module.MapFlags.READ)
                if not ok:
                    return
                try:
                    data = bytearray(minfo.data[:size])
                finally:
                    buf.unmap(minfo)
                poc_fixer.process_buffer(data)
                buf.fill(0, bytes(data))
                cur_pts = buf.pts
                if cur_pts != clock_time_none:
                    if previous_pts[0] == clock_time_none:
                        previous_pts[0] = cur_pts
                    elif cur_pts <= previous_pts[0]:
                        new_pts = previous_pts[0] + poc_frame_ns
                        buf.pts = new_pts
                        buf.dts = new_pts
                        previous_pts[0] = new_pts
                    else:
                        previous_pts[0] = cur_pts

            poc_identity.connect("handoff", _poc_handoff)

        if output_mode in {"rtsp", "rtsp_publish", "rtsp-publish"}:
            sink = make_element("rtspclientsink", "h264-rtsp-sink")
            sink.set_property(
                "location",
                os.environ.get(
                    "DS_RTSP_LOCATION",
                    "rtsp://cctv-media-server:8554/camera_1",
                ),
            )
            set_optional_property(sink, "protocols", "tcp")
            set_optional_property(sink, "latency", env_int("DS_RTSP_LATENCY_MS", 100))
            return [*h264_elements, *([poc_identity] if poc_identity else []), sink]

        mux = make_element("mpegtsmux", "mpegts-mux")
        sink = make_element("udpsink", "mpegts-udp-sink")
        set_optional_property(mux, "alignment", 7)
        set_optional_property(mux, "pcr-interval", 9000)
        set_optional_property(mux, "pat-interval", 9000)
        set_optional_property(mux, "pmt-interval", 9000)
        sink.set_property("host", os.environ.get("DS_H264_UDP_HOST", "cctv-media-server"))
        sink.set_property("port", env_int("DS_H264_UDP_PORT", 1234))
        sink.set_property("sync", False)
        sink.set_property("async", False)

        return [*h264_elements, *([poc_identity] if poc_identity else []), mux, sink]

    if output_mode in {"display", "egl", "ui"}:
        transform = make_element("nvegltransform", "egl-transform")
        sink = make_element("nveglglessink", "egl-sink")
        sink.set_property("sync", False)
        return [transform, sink]

    raise ValueError(
        "지원하지 않는 DS_OUTPUT_MODE 입니다: "
        f"{output_mode}. 사용 가능: fakesink, display, h264-mpegts, rtsp-publish"
    )


def register_pipeline_runtime_hooks(
    *,
    probe_element: Any,
    pipeline: Any,
    gst_module: Any,
    on_pad_probe: Callable[[Any, Any, Any], Any],
    on_bus_message: Callable[[Any, Any], bool],
) -> None:
    """파이프라인 실행에 필요한 src pad probe와 bus watch를 등록한다."""
    srcpad = probe_element.get_static_pad("src")
    if srcpad is None:
        raise RuntimeError(f"{probe_element.get_name()} src pad를 찾을 수 없습니다.")
    srcpad.add_probe(gst_module.PadProbeType.BUFFER, on_pad_probe, None)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message)


def start_pipeline_runtime(
    *,
    pipeline: Any,
    main_loop: Any,
    gst_module: Any,
    publish_loop_target: Callable[[], None],
    face_worker_loop_target: Callable[[], None],
) -> tuple[Any, Any, Any]:
    """DeepStream 런타임 스레드를 시작하고 파이프라인을 PLAYING으로 전환한다."""
    publish_thread = threading.Thread(
        target=publish_loop_target,
        daemon=True,
        name="ds-publish",
    )
    publish_thread.start()

    face_worker_thread = threading.Thread(
        target=face_worker_loop_target,
        daemon=True,
        name="ds-face-worker",
    )
    face_worker_thread.start()

    ret = pipeline.set_state(gst_module.State.PLAYING)
    if ret == gst_module.StateChangeReturn.FAILURE:
        raise RuntimeError("파이프라인을 PLAYING 상태로 전환하는 데 실패했습니다.")

    main_loop_thread = threading.Thread(
        target=main_loop.run,
        daemon=True,
        name="ds-main-loop",
    )
    main_loop_thread.start()

    return publish_thread, main_loop_thread, face_worker_thread


def stop_pipeline_runtime(
    *,
    pipeline: Any,
    main_loop: Any,
    publish_thread: Optional[Any],
    main_loop_thread: Optional[Any],
    face_worker_thread: Optional[Any],
    gst_module: Any,
    join_timeout_sec: float = 2.0,
) -> None:
    """DeepStream 런타임을 종료하고 관련 스레드를 조인한다."""
    if pipeline is not None:
        pipeline.set_state(gst_module.State.NULL)

    if main_loop is not None and main_loop.is_running():
        main_loop.quit()

    if publish_thread and publish_thread.is_alive():
        publish_thread.join(timeout=join_timeout_sec)
    if main_loop_thread and main_loop_thread.is_alive():
        main_loop_thread.join(timeout=join_timeout_sec)
    if face_worker_thread and face_worker_thread.is_alive():
        face_worker_thread.join(timeout=join_timeout_sec)


def create_pipeline_elements_bundle(
    *,
    make_element: Callable[[str, str], Any],
    preview_enabled: bool,
    primary_enabled: bool,
    helmet_enabled: bool,
    pphuman_enabled: bool,
    output_elements: List[Any],
    preview_elements: List[Any],
) -> DeepStreamPipelineElements:
    """DeepStream main path element 묶음을 생성한다."""
    streammux = make_element("nvstreammux", "streammux")
    nvinfer = make_element("nvinfer", "primary-infer") if primary_enabled else None
    pphuman_infer = (
        make_element("nvinfer", "pphuman-infer")
        if pphuman_enabled and nvinfer is not None
        else None
    )
    helmet_infer = make_element("nvinfer", "helmet-infer") if helmet_enabled else None
    tracker = make_element("nvtracker", "tracker") if (nvinfer or helmet_infer) else None
    converter = make_element("nvvideoconvert", "converter")
    osd = make_element("nvdsosd", "osd")
    tee = make_element("tee", "preview-tee") if preview_enabled else None
    output_queue = make_element("queue", "output-queue") if tee is not None else None

    return DeepStreamPipelineElements(
        streammux=streammux,
        nvinfer=nvinfer,
        pphuman_infer=pphuman_infer,
        helmet_infer=helmet_infer,
        tracker=tracker,
        converter=converter,
        osd=osd,
        tee=tee,
        output_queue=output_queue,
        preview_elements=preview_elements,
        output_elements=output_elements,
    )


def validate_pipeline_prerequisites(
    *,
    deepstream_loaded: bool,
    has_cameras: bool,
    infer_config_exists: bool,
    infer_config_path: Any,
) -> None:
    """DeepStream 파이프라인 빌드 전 필수 조건을 검증한다."""
    if not deepstream_loaded:
        raise RuntimeError(
            "DeepStreamProcessor 는 NVIDIA DeepStream SDK 와 pyds 바인딩이 "
            "설치된 환경(Jetson / Linux+GPU)에서만 실행할 수 있습니다."
        )
    if not has_cameras:
        raise RuntimeError("DeepStream 파이프라인을 만들 카메라가 없습니다.")
    if not infer_config_exists:
        raise FileNotFoundError(f"nvinfer 설정 파일 없음: {infer_config_path}")


def configure_pipeline_elements_bundle(
    *,
    elements: DeepStreamPipelineElements,
    n_cams: int,
    configure_streammux: Callable[[Any, int], None],
    configure_infer_elements: Callable[[Any, Any, Any, int], None],
    configure_tracker: Callable[[Any], None],
    configure_output_queue: Callable[[Any], None],
) -> None:
    """생성된 파이프라인 요소 묶음에 런타임 설정을 적용한다."""
    configure_streammux(elements.streammux, n_cams)
    configure_infer_elements(
        elements.nvinfer,
        elements.helmet_infer,
        elements.pphuman_infer,
        n_cams,
    )
    if elements.tracker is not None:
        configure_tracker(elements.tracker)
    if elements.output_queue is not None:
        configure_output_queue(elements.output_queue)


def add_pipeline_elements(pipeline: Any, elements: DeepStreamPipelineElements) -> None:
    """요소 묶음을 GStreamer pipeline에 추가한다."""
    for element in elements.all_elements():
        pipeline.add(element)
