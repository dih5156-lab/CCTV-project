"""Small helpers for assembling DeepStream GStreamer pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional


@dataclass(frozen=True)
class DeepStreamPipelineElements:
    streammux: Any
    nvinfer: Optional[Any]
    tracker: Optional[Any]
    pphuman_infer: Optional[Any]
    helmet_infer: Optional[Any]
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
                self.tracker,
                self.pphuman_infer,
                self.helmet_infer,
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
    add_primary_probe: Callable[[Any], None],
    link_preview_branch: Callable[..., Any],
    on_pphuman_linked: Callable[[], None],
) -> Any:
    """Link the main inference/output path and return the element to probe."""
    previous = elements.streammux
    probe_element = elements.streammux

    if elements.nvinfer is not None:
        link_or_raise(previous, elements.nvinfer, "nvstreammux -> nvinfer link 실패")
        previous = elements.nvinfer
        probe_element = elements.nvinfer
        if elements.pphuman_infer is not None:
            add_primary_probe(elements.nvinfer)

    if elements.tracker is not None:
        link_or_raise(
            previous,
            elements.tracker,
            f"{previous.get_name()} -> nvtracker link 실패",
        )
        previous = elements.tracker

    if elements.pphuman_infer is not None:
        link_or_raise(
            previous,
            elements.pphuman_infer,
            f"{previous.get_name()} -> pphuman-infer link 실패",
        )
        previous = elements.pphuman_infer
        probe_element = elements.pphuman_infer
        on_pphuman_linked()

    if elements.helmet_infer is not None:
        link_or_raise(
            previous,
            elements.helmet_infer,
            f"{previous.get_name()} -> helmet-infer link 실패",
        )
        previous = elements.helmet_infer
        probe_element = elements.helmet_infer

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
