"""카메라별 DeepStream RTSP 출력 헬퍼 테스트."""

from __future__ import annotations

import pytest

from src.core._deepstream_rtsp_output import resolve_rtsp_locations


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
