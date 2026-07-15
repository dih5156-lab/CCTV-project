"""카메라별 DeepStream RTSP 출력 구성 헬퍼."""

from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_RTSP_LOCATION_TEMPLATE = "rtsp://cctv-media-server:8554/{camera_id}"
_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


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
