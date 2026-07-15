"""카메라별 DeepStream RTSP 출력 구성 헬퍼."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_RTSP_LOCATION_TEMPLATE = "rtsp://cctv-media-server:8554/{camera_id}"
_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class RtspOutputBranch:
    """카메라 하나의 nvstreamdemux 출력 branch."""

    camera_id: str
    pad_id: int
    elements: list[Any]


def validate_camera_id(camera_id: str) -> None:
    """RTSP path와 GStreamer element 이름에 안전한 카메라 ID인지 확인한다."""
    if not isinstance(camera_id, str) or not _CAMERA_ID_PATTERN.fullmatch(camera_id):
        raise ValueError(
            f"유효하지 않은 camera ID: {camera_id!r}; "
            "영문자, 숫자, _, -만 사용할 수 있습니다."
        )


def resolve_rtsp_locations(
    camera_ids: Sequence[str],
    *,
    location_template: str | None,
    legacy_location: str | None,
) -> dict[str, str]:
    """활성 카메라 ID별 RTSP 게시 URL을 반환한다."""
    ids = list(camera_ids)
    for camera_id in ids:
        validate_camera_id(camera_id)

    template = location_template.strip() if location_template else None
    legacy = legacy_location.strip() if legacy_location else None
    if template:
        if "{camera_id}" not in template:
            raise ValueError("DS_RTSP_LOCATION_TEMPLATE에 {camera_id}가 필요합니다.")
        return {
            camera_id: template.replace("{camera_id}", camera_id)
            for camera_id in ids
        }

    if legacy:
        if len(ids) != 1:
            raise ValueError(
                "다중 카메라 RTSP 출력에는 DS_RTSP_LOCATION_TEMPLATE이 필요합니다."
            )
        return {ids[0]: legacy}

    return {
        camera_id: DEFAULT_RTSP_LOCATION_TEMPLATE.replace("{camera_id}", camera_id)
        for camera_id in ids
    }


def create_rtsp_output_branches(
    *,
    source_entries: list[tuple[int, str, dict[str, Any], str]],
    locations: dict[str, str],
    make_element: Callable[[str, str], Any],
    create_output_elements: Callable[[str, str], list[Any]],
) -> tuple[Any, list[RtspOutputBranch]]:
    """활성 source 순서대로 demux와 카메라별 출력 branch를 만든다."""
    demux = make_element("nvstreamdemux", "output-demux")
    branches = [
        RtspOutputBranch(
            camera_id=camera_id,
            pad_id=pad_id,
            elements=create_output_elements(camera_id, locations[camera_id]),
        )
        for pad_id, camera_id, _info, _source_uri in source_entries
    ]
    return demux, branches


def link_rtsp_output_branches(
    *,
    demux: Any,
    branches: list[RtspOutputBranch],
    gst_module: Any,
    link_or_raise: Callable[[Any, Any, str | None], None],
) -> None:
    """nvstreamdemux의 source pad를 카메라별 출력 branch에 연결한다."""
    for branch in branches:
        if not branch.elements:
            raise RuntimeError(
                f"RTSP 출력 branch가 비어 있습니다: "
                f"{branch.camera_id} pad_id={branch.pad_id}"
            )

        demux_pad = demux.get_request_pad(f"src_{branch.pad_id}")
        first = branch.elements[0]
        sink_pad = first.get_static_pad("sink")
        if demux_pad is None or sink_pad is None:
            raise RuntimeError(
                f"nvstreamdemux pad 요청 실패: "
                f"{branch.camera_id} pad_id={branch.pad_id}"
            )
        if demux_pad.link(sink_pad) != gst_module.PadLinkReturn.OK:
            raise RuntimeError(
                f"nvstreamdemux branch 연결 실패: "
                f"{branch.camera_id} pad_id={branch.pad_id}"
            )

        previous = first
        for element in branch.elements[1:]:
            link_or_raise(previous, element, None)
            previous = element
