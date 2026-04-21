"""제어 관련 엔드포인트.

GET    /api/v1/control/mode              — 현재 제어 모드 조회
POST   /api/v1/control/mode              — 제어 모드 변경
GET    /api/v1/control/pending           — 수동 승인 대기 이벤트 목록
POST   /api/v1/control/approve/{eid}     — 이벤트 승인
POST   /api/v1/control/reject/{eid}      — 이벤트 거부

내부 cctv-action-layer의 REST 서버로 요청을 프록시한다.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, Path

from .._action_proxy import proxy_action_request
from ..dependencies._settings import ACTION_LAYER_URL as _ACTION_URL
from ..dependencies.auth import verify_api_key
from ..schemas.common import BaseResponse, success_response
from ..schemas.site import ApprovalOut, ModeOut, ModeSetIn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/control", tags=["control"])


@router.get(
    "/mode",
    response_model=BaseResponse[ModeOut],
    summary="현재 제어 모드 조회",
)
async def get_mode(_: None = Depends(verify_api_key)) -> BaseResponse[ModeOut]:
    raw = await proxy_action_request(_ACTION_URL, "get", "/mode")
    return success_response(ModeOut(mode=raw.get("mode", "auto")))


@router.post(
    "/mode",
    response_model=BaseResponse[ModeOut],
    summary="제어 모드 변경",
    description=(
        "site_id를 지정하면 해당 사이트만, 생략하면 전체 사이트의 모드를 변경합니다.\n\n"
        "- `auto`: 탐지 즉시 자동 실행\n"
        "- `manual`: 관리자 승인 후 실행"
    ),
)
async def set_mode(
    body: ModeSetIn,
    _: None = Depends(verify_api_key),
) -> BaseResponse[ModeOut]:
    payload: dict = {"mode": body.mode.value}
    if body.site_id:
        payload["site_id"] = body.site_id
    raw = await proxy_action_request(_ACTION_URL, "post", "/mode", payload)
    return success_response(
        ModeOut(mode=raw.get("mode", body.mode.value), site_id=body.site_id)
    )


@router.get(
    "/pending",
    response_model=BaseResponse[List[dict]],
    summary="수동 승인 대기 이벤트 목록",
    description="control_mode가 manual인 사이트에서 대기 중인 이벤트를 반환합니다.",
)
async def list_pending(_: None = Depends(verify_api_key)) -> BaseResponse[List[dict]]:
    raw = await proxy_action_request(_ACTION_URL, "get", "/pending")
    items = raw if isinstance(raw, list) else []
    return success_response(items)


@router.post(
    "/approve/{event_id}",
    response_model=BaseResponse[ApprovalOut],
    summary="이벤트 승인",
    description="수동 모드에서 대기 중인 이벤트를 승인하여 알람 장치를 실행합니다.",
)
async def approve_event(
    event_id: str = Path(min_length=1),
    _: None = Depends(verify_api_key),
) -> BaseResponse[ApprovalOut]:
    raw = await proxy_action_request(_ACTION_URL, "post", f"/approve/{event_id}", {})
    return success_response(
        ApprovalOut(
            event_id=event_id,
            status="approved",
            message=raw.get("message", "승인 완료"),
        )
    )


@router.post(
    "/reject/{event_id}",
    response_model=BaseResponse[ApprovalOut],
    summary="이벤트 거부",
    description="수동 모드에서 대기 중인 이벤트를 거부합니다.",
)
async def reject_event(
    event_id: str = Path(min_length=1),
    _: None = Depends(verify_api_key),
) -> BaseResponse[ApprovalOut]:
    raw = await proxy_action_request(_ACTION_URL, "post", f"/reject/{event_id}", {})
    return success_response(
        ApprovalOut(
            event_id=event_id,
            status="rejected",
            message=raw.get("message", "거부 완료"),
        )
    )
