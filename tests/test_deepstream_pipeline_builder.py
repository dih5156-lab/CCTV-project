import pytest

from src.core._deepstream_pipeline_builder import (
    DeepStreamPipelineElements,
    create_pipeline_elements_bundle,
    link_deepstream_pipeline_path,
)


@pytest.mark.parametrize("encoder_name", ["nvv4l2h264enc", "x264enc"])
@pytest.mark.parametrize("configured_fps,expected_fps", [(None, 30), (15, 15), (0, 1), (-5, 1)])
def test_h264_output_negotiates_positive_rate_with_bounded_resampling(
    monkeypatch, encoder_name, configured_fps, expected_fps
):
    import os
    from types import SimpleNamespace

    from src.core._deepstream_pipeline_builder import create_h264_encoder_elements

    monkeypatch.setenv("DS_H264_ENCODER", encoder_name)
    monkeypatch.delenv("DS_H264_FPS", raising=False)
    if configured_fps is not None:
        monkeypatch.setenv("DS_H264_FPS", str(configured_fps))

    def make_element(kind, name):
        properties = {}
        return SimpleNamespace(
            kind=kind, name=name, properties=properties,
            set_property=lambda key, value: properties.__setitem__(key, value),
        )

    elements = create_h264_encoder_elements(
        make_element=make_element,
        env_int=lambda key, default: int(os.environ.get(key, default)),
        set_optional_property=lambda element, key, value: element.set_property(key, value),
        gst_module=SimpleNamespace(Caps=SimpleNamespace(from_string=lambda value: value)),
    )
    kinds = [element.kind for element in elements]
    assert "videorate" in kinds, "FPS caps alone cannot negotiate streammux's 0/1 rate"
    rate_index = kinds.index("videorate")
    assert rate_index < kinds.index(encoder_name)
    rate = elements[rate_index]
    assert rate.properties["drop-only"] is False
    assert rate.properties["max-duplication-time"] == 100_000_000
    assert rate.properties["skip-to-first"] is True
    caps = elements[rate_index + 1].properties["caps"]
    assert f"framerate={expected_fps}/1" in caps
    assert "format=NV12" in caps
    assert ("memory:NVMM" in caps) == (encoder_name != "x264enc")
    # 양수 FPS 협상뿐 아니라 수신기가 읽는 H.264 시간 정보도 기록해야 한다.
    encoder = next(element for element in elements if element.kind == encoder_name)
    assert encoder.properties["insert-vui"] is True


class _Element:
    def __init__(self, name: str) -> None:
        self.name = name
        self.linked_to = []

    def link(self, other):
        self.linked_to.append(other)
        return True

    def get_name(self) -> str:
        return self.name

    def get_static_pad(self, name):
        return self

    def add_probe(self, *args):
        self.probe_args = args


def _link_or_raise(first, second, message=None) -> None:
    assert first.link(second)


def test_pipeline_elements_keep_gstreamer_add_order() -> None:
    elements = DeepStreamPipelineElements(
        streammux=_Element("streammux"),
        nvinfer=_Element("primary"),
        primary_tensor_queue=_Element("primary-tensor-queue"),
        metadata_probe_queue=_Element("metadata-probe-queue"),
        tracker=_Element("tracker"),
        pphuman_infer=_Element("pphuman"),
        helmet_infer=_Element("helmet"),
        converter=_Element("converter"),
        osd=_Element("osd"),
        tee=_Element("tee"),
        output_queue=_Element("output-queue"),
        preview_elements=[_Element("preview-queue")],
        output_elements=[_Element("sink")],
    )

    assert [element.get_name() for element in elements.all_elements()] == [
        "streammux",
        "primary",
        "primary-tensor-queue",
        "pphuman",
        "helmet",
        "tracker",
        "metadata-probe-queue",
        "converter",
        "osd",
        "tee",
        "output-queue",
        "preview-queue",
        "sink",
    ]
    assert elements.topology() == (True, True, True)


def test_link_pipeline_path_returns_last_inference_probe_element() -> None:
    elements = DeepStreamPipelineElements(
        streammux=_Element("streammux"),
        nvinfer=_Element("primary"),
        primary_tensor_queue=_Element("primary-tensor-queue"),
        metadata_probe_queue=_Element("metadata-probe-queue"),
        tracker=_Element("tracker"),
        pphuman_infer=_Element("pphuman"),
        helmet_infer=_Element("helmet"),
        converter=_Element("converter"),
        osd=_Element("osd"),
        tee=None,
        output_queue=None,
        preview_elements=[],
        output_elements=[_Element("sink")],
    )
    gst_module = type("Gst", (), {"PadProbeType": type("PadProbeType", (), {"BUFFER": "buffer"})})
    def primary_probe_callback(*args):
        return None

    probe_element = link_deepstream_pipeline_path(
        elements,
        link_or_raise=_link_or_raise,
        gst_module=gst_module,
        primary_probe_callback=primary_probe_callback,
        link_preview_branch=lambda **kwargs: kwargs["output_queue"],
        pphuman_gie_id=3,
        pphuman_infer_config="pphuman.txt",
    )

    assert probe_element.get_name() == "metadata-probe-queue"
    assert elements.primary_tensor_queue.probe_args == ("buffer", primary_probe_callback, None)
    assert [element.get_name() for element in elements.streammux.linked_to] == ["primary"]
    assert [element.get_name() for element in elements.nvinfer.linked_to] == ["primary-tensor-queue"]
    assert [element.get_name() for element in elements.primary_tensor_queue.linked_to] == ["pphuman"]
    assert [element.get_name() for element in elements.pphuman_infer.linked_to] == ["helmet"]
    assert [element.get_name() for element in elements.helmet_infer.linked_to] == ["tracker"]
    assert [element.get_name() for element in elements.tracker.linked_to] == ["metadata-probe-queue"]
    assert [element.get_name() for element in elements.metadata_probe_queue.linked_to] == ["converter"]


def test_crop_pose_bundle_adds_optional_pose_sgie_after_primary() -> None:
    created = []

    def make_element(kind, name):
        element = _Element(name)
        created.append((kind, name))
        return element

    elements = create_pipeline_elements_bundle(
        make_element=make_element,
        preview_enabled=False,
        primary_enabled=True,
        helmet_enabled=True,
        pphuman_enabled=True,
        output_elements=[_Element("sink")],
        preview_elements=[],
        pose_enabled=True,
    )

    assert elements.pose_infer.get_name() == "pose-infer"
    assert elements.primary_tensor_queue.get_name() == "primary-tensor-queue"
    assert elements.metadata_probe_queue.get_name() == "metadata-probe-queue"
    assert [element.get_name() for element in elements.all_elements()[:6]] == [
        "streammux", "primary-infer", "primary-tensor-queue", "pose-infer",
        "pphuman-infer", "helmet-infer"
    ]


def test_crop_pose_bundle_can_bypass_native_tracker() -> None:
    elements = create_pipeline_elements_bundle(
        make_element=lambda _kind, name: _Element(name),
        preview_enabled=False,
        primary_enabled=True,
        helmet_enabled=True,
        pphuman_enabled=True,
        output_elements=[_Element("sink")],
        preview_elements=[],
        pose_enabled=True,
        tracker_enabled=False,
    )

    assert elements.tracker is None


@pytest.mark.parametrize("mode,expected", [(None, None), ("", None), ("0", 0), ("1", 1), (" 0 ", 0)])
def test_osd_mode_is_opt_in_without_changing_inference(monkeypatch, mode, expected):
    from unittest.mock import Mock

    # 미설정은 SDK 기본값을 유지하고 명시한 경우에만 그리기 방식을 바꾼다.
    monkeypatch.delenv("DS_OSD_PROCESS_MODE", raising=False)
    if mode is not None:
        monkeypatch.setenv("DS_OSD_PROCESS_MODE", mode)
    make_element = Mock(side_effect=lambda kind, name: Mock(kind=kind, element_name=name))
    elements = create_pipeline_elements_bundle(
        make_element=make_element, preview_enabled=False, primary_enabled=True,
        helmet_enabled=True, pphuman_enabled=True, pose_enabled=True,
        output_elements=[], preview_elements=[],
    )
    if expected is None:
        elements.osd.set_property.assert_not_called()
    else:
        elements.osd.set_property.assert_called_once_with("process-mode", expected)
    for infer in (elements.nvinfer, elements.pose_infer, elements.helmet_infer, elements.pphuman_infer):
        assert infer.kind == "nvinfer"
        infer.set_property.assert_not_called()


@pytest.mark.parametrize("mode", ["2", "-1", "cpu", "0.0"])
def test_invalid_osd_mode_fails_before_allocating_pipeline(monkeypatch, mode):
    from unittest.mock import Mock

    # 오타를 GPU 기본값으로 숨기지 않고 시작 단계에서 원인을 알려준다.
    monkeypatch.setenv("DS_OSD_PROCESS_MODE", mode)
    make_element = Mock()
    with pytest.raises(ValueError, match="DS_OSD_PROCESS_MODE"):
        create_pipeline_elements_bundle(
            make_element=make_element, preview_enabled=False, primary_enabled=True,
            helmet_enabled=False, pphuman_enabled=False,
            output_elements=[], preview_elements=[],
        )
    make_element.assert_not_called()


def test_runtime_registers_reset_observer_on_same_pad_as_frames():
    from types import SimpleNamespace
    from unittest.mock import Mock, call

    from src.core._deepstream_pipeline_builder import register_pipeline_runtime_hooks
    pad, bus = Mock(), Mock()
    element = SimpleNamespace(get_static_pad=lambda name: pad)
    gst = SimpleNamespace(PadProbeType=SimpleNamespace(BUFFER=1, EVENT_DOWNSTREAM=2))
    frame, event, message = Mock(), Mock(), Mock()
    register_pipeline_runtime_hooks(probe_element=element,
        pipeline=SimpleNamespace(get_bus=lambda: bus), gst_module=gst,
        on_pad_probe=frame, on_bus_message=message, on_stream_event=event)
    assert pad.add_probe.call_args_list == [call(1, frame, None), call(2, event, None)]
    bus.connect.assert_called_once_with("message", message)


@pytest.mark.parametrize("helmet,attributes,pose", [
    (True, False, False), (False, True, False), (False, False, True),
    (True, True, False), (False, False, False),
])
def test_each_secondary_receives_primary_roi_probe(helmet, attributes, pose):
    elements = create_pipeline_elements_bundle(
        make_element=lambda kind, name: _Element(name), preview_enabled=False,
        primary_enabled=True, helmet_enabled=helmet, pphuman_enabled=attributes,
        pose_enabled=pose, output_elements=[_Element("sink")], preview_elements=[])
    needs_roi = helmet or attributes or pose
    assert (elements.primary_tensor_queue is not None) == needs_roi
    def callback(*args):
        return None
    gst = type("Gst", (), {"PadProbeType": type("PadProbeType", (), {"BUFFER": "buffer"})})
    link_deepstream_pipeline_path(elements, link_or_raise=_link_or_raise,
        gst_module=gst, primary_probe_callback=callback,
        link_preview_branch=lambda **kwargs: None, pphuman_gie_id=3,
        pphuman_infer_config="attributes.txt")
    if needs_roi:
        assert elements.primary_tensor_queue.probe_args == ("buffer", callback, None)
        assert elements.nvinfer.linked_to == [elements.primary_tensor_queue]


def test_preview_can_pause_without_waiting_for_first_frame():
    """미리보기 첫 버퍼가 없어도 다른 송출 분기의 상태 전환을 막지 않는다."""
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    from src.core._deepstream_pipeline_builder import create_preview_elements

    Gst.init(None)

    def make_element(factory, name):
        # GPU 변환 대신 identity를 쓰되 상태 전환은 실제 GStreamer appsink로 검증한다.
        element = Gst.ElementFactory.make(
            "identity" if factory == "nvvideoconvert" else factory, name
        )
        if element is None:
            pytest.skip(f"GStreamer {factory} plugin unavailable")
        return element

    elements = create_preview_elements(
        make_element=make_element,
        env_int=lambda _name, default: default,
        gst_module=Gst,
        on_preview_sample=lambda _sink: Gst.FlowReturn.OK,
    )
    sink = elements[-1]
    try:
        sink.set_state(Gst.State.PAUSED)
        result, state, pending = sink.get_state(100 * Gst.MSECOND)
        assert result == Gst.StateChangeReturn.SUCCESS
        assert state == Gst.State.PAUSED
        assert pending == Gst.State.VOID_PENDING
    finally:
        sink.set_state(Gst.State.NULL)


def test_shutdown_flushes_streaming_before_native_state_transition():
    """NULL 전환이 기다리는 streaming 작업을 먼저 깨우는 종료 순서를 보장한다."""
    from types import SimpleNamespace

    from src.core._deepstream_pipeline_builder import stop_pipeline_runtime

    calls = []
    flush = object()
    pipeline = SimpleNamespace(
        send_event=lambda event: calls.append(("flush", event)),
        # RTSP 출력이 없는 기존 경로도 flush 후 전체 종료 순서를 유지한다.
        get_by_name=lambda _name: None,
        set_state=lambda state: calls.append(("state", state)),
    )
    gst = SimpleNamespace(
        Event=SimpleNamespace(new_flush_start=lambda: flush),
        State=SimpleNamespace(NULL="NULL"),
    )
    stop_pipeline_runtime(
        pipeline=pipeline, main_loop=None, publish_thread=None,
        main_loop_thread=None, face_worker_thread=None, gst_module=gst,
    )
    assert calls == [("flush", flush), ("state", "NULL")]


def test_shutdown_does_not_reactivate_stopped_rtsp_bin():
    """먼저 종료한 출력 bin을 전체 종료 중 다시 여는 실제 Gst 상태 전환을 막는다."""
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    from src.core._deepstream_pipeline_builder import stop_pipeline_runtime

    Gst.init(None)

    class ObservedOutputBin(Gst.Bin):
        # 네트워크/GPU 없이 rtspclientsink와 같은 bin의 상태 전파를 확인한다.
        __gtype_name__ = "CctvShutdownObservedOutputBin"

        def __init__(self):
            super().__init__()
            self.transitions = []

        def do_change_state(self, transition):
            self.transitions.append(transition)
            return Gst.Bin.do_change_state(self, transition)

    pipeline = Gst.Pipeline.new("shutdown-regression")
    sink = ObservedOutputBin()
    sink.set_name("h264-rtsp-sink")
    pipeline.add(sink)
    try:
        assert pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.SUCCESS
        sink.transitions.clear()
        stop_pipeline_runtime(
            pipeline=pipeline, main_loop=None, publish_thread=None,
            main_loop_thread=None, face_worker_thread=None, gst_module=Gst,
        )
        assert Gst.StateChange.NULL_TO_READY not in sink.transitions
        assert sink.get_state(0).state == Gst.State.NULL
        assert pipeline.get_state(0).state == Gst.State.NULL
        # 전체 종료 후에는 상태 잠금을 해제하여 다음 명시적 시작을 방해하지 않는다.
        assert not sink.is_locked_state()
        assert pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.SUCCESS
        assert sink.get_state(0).state == Gst.State.PLAYING
    finally:
        pipeline.set_state(Gst.State.NULL)
