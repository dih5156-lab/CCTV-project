from src.core._deepstream_pipeline_builder import (
    DeepStreamPipelineElements,
    link_deepstream_pipeline_path,
)


class _Element:
    def __init__(self, name: str) -> None:
        self.name = name
        self.linked_to = []

    def link(self, other):
        self.linked_to.append(other)
        return True

    def get_name(self) -> str:
        return self.name


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
        "tracker",
        "pphuman",
        "helmet",
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
    primary_probe_calls = []
    pphuman_linked = []

    probe_element = link_deepstream_pipeline_path(
        elements,
        link_or_raise=_link_or_raise,
        add_primary_probe=primary_probe_calls.append,
        link_preview_branch=lambda **kwargs: kwargs["output_queue"],
        on_pphuman_linked=lambda: pphuman_linked.append(True),
    )

    assert probe_element.get_name() == "helmet"
    assert primary_probe_calls == [elements.nvinfer]
    assert pphuman_linked == [True]
    assert [element.get_name() for element in elements.streammux.linked_to] == ["primary"]
    assert [element.get_name() for element in elements.nvinfer.linked_to] == ["tracker"]
    assert [element.get_name() for element in elements.tracker.linked_to] == ["pphuman"]
    assert [element.get_name() for element in elements.pphuman_infer.linked_to] == ["helmet"]
    assert [element.get_name() for element in elements.helmet_infer.linked_to] == ["converter"]
