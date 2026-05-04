"""사이트 관련 엔드포인트.

GET    /api/v1/sites              — 전체 사이트 목록
POST   /api/v1/sites              — 사이트 등록
DELETE /api/v1/sites/{site_id}    — 사이트 삭제

내부 cctv-action-layer의 REST 서버(포트 8080)로 요청을 프록시한다.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, Path, status

from .._action_proxy import proxy_action_request
from ..dependencies._settings import ACTION_LAYER_URL as _ACTION_URL
from ..dependencies.auth import verify_api_key
from ..schemas.common import BaseResponse, success_response
from ..schemas.site import SiteCreateIn, SiteOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sites", tags=["sites"])


@router.get(
    "",
    response_model=BaseResponse[List[SiteOut]],
    summary="사이트 목록 조회",
    description="Action Layer에 등록된 사이트 목록을 반환합니다. 사이트별 camera_ids, control_mode, alarm_devices를 포함합니다.",
)
async def list_sites(_: None = Depends(verify_api_key)) -> BaseResponse[List[SiteOut]]:
    raw = await proxy_action_request(_ACTION_URL, "get", "/sites")
    sites = [SiteOut(**s) for s in (raw if isinstance(raw, list) else [])]
    return success_response(sites)


@router.post(
    "",
    response_model=BaseResponse[dict],
    status_code=status.HTTP_201_CREATED,
    summary="사이트 등록",
    description="새 사이트를 Action Layer에 등록합니다. site_id는 중복되면 안 되며, camera_ids는 현장 카메라 ID와 맞아야 합니다.",
)
async def create_site(
    body: SiteCreateIn,
    _: None = Depends(verify_api_key),
) -> BaseResponse[dict]:
    result = await proxy_action_request(_ACTION_URL, "post", "/sites", body.model_dump())
    return success_response(result)


@router.delete(
    "/{site_id}",
    response_model=BaseResponse[dict],
    summary="사이트 삭제",
    description="기존 사이트 1건을 삭제합니다. 삭제 후 해당 사이트의 수동 승인/제어 설정도 함께 사라질 수 있습니다.",
)
async def delete_site(
    site_id: str = Path(min_length=1),
    _: None = Depends(verify_api_key),
) -> BaseResponse[dict]:
    result = await proxy_action_request(_ACTION_URL, "delete", f"/sites/{site_id}")
    return success_response(result)
