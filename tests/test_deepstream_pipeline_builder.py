from src.core._deepstream_pipeline_builder import (
    DeepStreamPipelineElements,
    link_deepstream_pipeline_path,
)
from src.core._deepstream_rtsp_output import RtspOutputBranch


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
        "pphuman",
        "helmet",
        "tracker",
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

    assert probe_element.get_name() == "helmet"
    assert elements.nvinfer.probe_args == ("buffer", primary_probe_callback, None)
    assert [element.get_name() for element in elements.streammux.linked_to] == ["primary"]
    assert [element.get_name() for element in elements.nvinfer.linked_to] == ["pphuman"]
    assert [element.get_name() for element in elements.pphuman_infer.linked_to] == ["helmet"]
    assert [element.get_name() for element in elements.helmet_infer.linked_to] == ["tracker"]
    assert [element.get_name() for element in elements.tracker.linked_to] == ["converter"]


def test_pipeline_elements_add_demux_and_camera_branches_in_order() -> None:
    elements = DeepStreamPipelineElements(
        streammux=_Element("streammux"),
        nvinfer=None,
        tracker=None,
        pphuman_infer=None,
        helmet_infer=None,
        converter=_Element("converter"),
        osd=_Element("osd"),
        tee=None,
        output_queue=None,
        preview_elements=[],
        output_elements=[],
        output_demux=_Element("output-demux"),
        rtsp_output_branches=[
            RtspOutputBranch(
                camera_id="camera_1",
                pad_id=0,
                elements=[_Element("queue-camera_1"), _Element("sink-camera_1")],
            ),
            RtspOutputBranch(
                camera_id="camera_2",
                pad_id=1,
                elements=[_Element("queue-camera_2"), _Element("sink-camera_2")],
            ),
        ],
    )

    assert [element.get_name() for element in elements.all_elements()] == [
        "streammux",
        "converter",
        "osd",
        "output-demux",
        "queue-camera_1",
        "sink-camera_1",
        "queue-camera_2",
        "sink-camera_2",
    ]


def test_link_pipeline_routes_annotated_output_through_demux() -> None:
    osd = _Element("osd")
    demux = _Element("output-demux")
    branch_calls = []
    elements = DeepStreamPipelineElements(
        streammux=_Element("streammux"),
        nvinfer=None,
        tracker=None,
        pphuman_infer=None,
        helmet_infer=None,
        converter=_Element("converter"),
        osd=osd,
        tee=None,
        output_queue=None,
        preview_elements=[],
        output_elements=[],
        output_demux=demux,
        rtsp_output_branches=[],
    )

    link_deepstream_pipeline_path(
        elements,
        link_or_raise=_link_or_raise,
        gst_module=type("Gst", (), {"PadProbeType": type("PadProbeType", (), {"BUFFER": "buffer"})}),
        primary_probe_callback=lambda *_args: None,
        link_preview_branch=lambda **kwargs: kwargs["output_queue"],
        link_rtsp_branches=lambda **kwargs: branch_calls.append(kwargs),
        pphuman_gie_id=3,
        pphuman_infer_config="pphuman.txt",
    )

    assert demux in osd.linked_to
    assert branch_calls == [{"demux": demux, "branches": []}]
