"""카메라별 DeepStream RTSP 출력 헬퍼 테스트."""

from __future__ import annotations

import pytest

from src.core._deepstream_rtsp_output import (
    create_rtsp_output_branches,
    link_rtsp_output_branches,
    resolve_rtsp_locations,
)


class _FakePad:
    def __init__(self, link_result: str = "ok") -> None:
        self.link_result = link_result
        self.linked_to = None

    def link(self, other):
        self.linked_to = other
        return self.link_result


class _FakeElement:
    def __init__(self, factory: str, name: str) -> None:
        self.factory = factory
        self.name = name
        self.properties = {}
        self.linked_to = []
        self.sink_pad = _FakePad()

    def set_property(self, name: str, value) -> None:
        self.properties[name] = value

    def get_name(self) -> str:
        return self.name

    def get_static_pad(self, name: str):
        return self.sink_pad if name == "sink" else None

    def link(self, other) -> bool:
        self.linked_to.append(other)
        return True


class _FakeDemux(_FakeElement):
    def __init__(self, pads: dict[str, _FakePad | None]) -> None:
        super().__init__("nvstreamdemux", "output-demux")
        self.pads = pads

    def get_request_pad(self, name: str):
        return self.pads.get(name)


def test_resolve_rtsp_locations_expands_camera_id_template() -> None:
    assert resolve_rtsp_locations(
        ["camera_1", "entrance-2"],
        location_template="rtsp://media:8554/{camera_id}",
        legacy_location=None,
    ) == {
        "camera_1": "rtsp://media:8554/camera_1",
        "entrance-2": "rtsp://media:8554/entrance-2",
    }


def test_resolve_rtsp_locations_keeps_single_camera_legacy_url() -> None:
    assert resolve_rtsp_locations(
        ["camera_1"],
        location_template=None,
        legacy_location="rtsp://media:8554/existing",
    ) == {"camera_1": "rtsp://media:8554/existing"}


def test_resolve_rtsp_locations_rejects_legacy_url_for_multiple_cameras() -> None:
    with pytest.raises(ValueError, match="DS_RTSP_LOCATION_TEMPLATE"):
        resolve_rtsp_locations(
            ["camera_1", "camera_2"],
            location_template=None,
            legacy_location="rtsp://media:8554/shared",
        )


@pytest.mark.parametrize("camera_id", ["camera/1", "camera 1", "카메라1", ""])
def test_resolve_rtsp_locations_rejects_unsafe_camera_id(camera_id: str) -> None:
    with pytest.raises(ValueError, match="camera ID"):
        resolve_rtsp_locations(
            [camera_id],
            location_template="rtsp://media:8554/{camera_id}",
            legacy_location=None,
        )


def test_resolve_rtsp_locations_uses_default_template() -> None:
    assert resolve_rtsp_locations(
        ["camera_1"],
        location_template=None,
        legacy_location=None,
    ) == {"camera_1": "rtsp://cctv-media-server:8554/camera_1"}


def test_resolve_rtsp_locations_requires_camera_id_placeholder() -> None:
    with pytest.raises(ValueError, match="camera_id"):
        resolve_rtsp_locations(
            ["camera_1"],
            location_template="rtsp://media:8554/static",
            legacy_location=None,
        )


def test_create_rtsp_output_branches_uses_unique_names_and_locations() -> None:
    created = []

    def make_element(factory: str, name: str):
        element = _FakeElement(factory, name)
        created.append(element)
        return element

    def create_output(camera_id: str, location: str):
        queue = make_element("queue", f"output-queue-{camera_id}")
        sink = make_element("rtspclientsink", f"h264-rtsp-sink-{camera_id}")
        sink.set_property("location", location)
        return [queue, sink]

    demux, branches = create_rtsp_output_branches(
        source_entries=[
            (0, "camera_1", {}, "rtsp://input/1"),
            (1, "camera_2", {}, "rtsp://input/2"),
        ],
        locations={
            "camera_1": "rtsp://media:8554/camera_1",
            "camera_2": "rtsp://media:8554/camera_2",
        },
        make_element=make_element,
        create_output_elements=create_output,
    )

    assert demux.factory == "nvstreamdemux"
    assert [branch.camera_id for branch in branches] == ["camera_1", "camera_2"]
    assert [branch.pad_id for branch in branches] == [0, 1]
    assert branches[0].elements[-1].properties["location"].endswith("/camera_1")
    assert branches[1].elements[-1].properties["location"].endswith("/camera_2")
    assert len({element.name for element in created}) == len(created)


def test_link_rtsp_output_branches_links_demux_pad_to_camera_queue() -> None:
    demux_pad = _FakePad()
    demux = _FakeDemux({"src_1": demux_pad})
    queue = _FakeElement("queue", "output-queue-camera_2")
    sink = _FakeElement("rtspclientsink", "h264-rtsp-sink-camera_2")
    _unused, branches = create_rtsp_output_branches(
        source_entries=[(1, "camera_2", {}, "rtsp://input/2")],
        locations={"camera_2": "rtsp://media:8554/camera_2"},
        make_element=lambda _factory, _name: demux,
        create_output_elements=lambda _camera_id, _location: [queue, sink],
    )

    link_rtsp_output_branches(
        demux=demux,
        branches=branches,
        gst_module=type("Gst", (), {"PadLinkReturn": type("Return", (), {"OK": "ok"})}),
        link_or_raise=lambda first, second, _message=None: first.link(second),
    )

    assert demux_pad.linked_to is queue.sink_pad
    assert sink in queue.linked_to


@pytest.mark.parametrize(
    ("demux_pad", "match"),
    [(None, "camera_2.*pad_id=1"), (_FakePad("failed"), "camera_2.*pad_id=1")],
)
def test_link_rtsp_output_branches_reports_camera_specific_pad_error(
    demux_pad,
    match: str,
) -> None:
    demux = _FakeDemux({"src_1": demux_pad})
    queue = _FakeElement("queue", "output-queue-camera_2")
    _unused, branches = create_rtsp_output_branches(
        source_entries=[(1, "camera_2", {}, "rtsp://input/2")],
        locations={"camera_2": "rtsp://media:8554/camera_2"},
        make_element=lambda _factory, _name: demux,
        create_output_elements=lambda _camera_id, _location: [queue],
    )

    with pytest.raises(RuntimeError, match=match):
        link_rtsp_output_branches(
            demux=demux,
            branches=branches,
            gst_module=type("Gst", (), {"PadLinkReturn": type("Return", (), {"OK": "ok"})}),
            link_or_raise=lambda first, second, _message=None: first.link(second),
        )
