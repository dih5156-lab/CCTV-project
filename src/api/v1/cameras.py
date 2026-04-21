"""GET /api/v1/cameras — 카메라 목록 조회 엔드포인트.

cameras.json을 읽어 등록된 카메라 목록을 반환한다.
비밀정보(RTSP 자격증명 등)는 응답에서 제거한다.
"""

from __future__ import annotations

import json
import logging
from typing import List
from urllib.parse import urlparse, urlunparse

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..dependencies._settings import CAMERAS_JSON as _CAMERAS_JSON
from ..dependencies.auth import verify_api_key
from ..dependencies.rate_limit import limiter
from ..schemas.common import BaseResponse, success_response
from ..schemas.site import CameraOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])


def _strip_credentials(url: str) -> str:
    """RTSP URL에서 사용자명/비밀번호를 제거한다."""
    try:
        parsed = urlparse(url)
        sanitized = parsed._replace(netloc=parsed.hostname or "")
        return urlunparse(sanitized)
    except Exception:
        return "[hidden]"


def _load_cameras() -> List[CameraOut]:
    if not _CAMERAS_JSON.exists():
        return []
    try:
        raw = json.loads(_CAMERAS_JSON.read_text(encoding="utf-8"))
        cameras = raw if isinstance(raw, list) else raw.get("cameras", [])
        result = []
        for cam in cameras:
            url = cam.get("url") or cam.get("rtsp_url") or ""
            result.append(
                CameraOut(
                    id=str(cam.get("id", cam.get("camera_id", ""))),
                    name=cam.get("name"),
                    url=_strip_credentials(url) if url else None,
                    zones=cam.get("zones"),
                )
            )
        return result
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("cameras.json 로드 실패: %s", exc)
        return []


@router.get(
    "",
    response_model=BaseResponse[List[CameraOut]],
    summary="카메라 목록 조회",
    description="등록된 CCTV 카메라 목록을 반환합니다. RTSP URL의 자격증명은 제거됩니다.",
)
@limiter.limit("60/minute")
def list_cameras(request: Request, _: None = Depends(verify_api_key)) -> BaseResponse[List[CameraOut]]:
    cameras = _load_cameras()
    return success_response(cameras)


@router.get(
    "/{camera_id}",
    response_model=BaseResponse[CameraOut],
    summary="카메라 단건 조회",
)
@limiter.limit("60/minute")
def get_camera(request: Request, camera_id: str, _: None = Depends(verify_api_key)) -> BaseResponse[CameraOut]:
    cameras = _load_cameras()
    for cam in cameras:
        if cam.id == camera_id:
            return success_response(cam)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="카메라를 찾을 수 없습니다.")
