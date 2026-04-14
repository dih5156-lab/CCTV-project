"""사이트 관련 엔드포인트.

GET    /api/v1/sites              — 전체 사이트 목록
POST   /api/v1/sites              — 사이트 등록
DELETE /api/v1/sites/{site_id}    — 사이트 삭제

내부 cctv-action-layer의 REST 서버(포트 8080)로 요청을 프록시한다.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, status

from ..dependencies.auth import verify_api_key
from ..schemas.common import BaseResponse
from ..schemas.site import SiteCreateIn, SiteOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sites", tags=["sites"])

_ACTION_URL = os.environ.get("ACTION_LAYER_URL", "http://cctv-action-layer:8080")


async def _proxy(method: str, path: str, payload: dict | None = None) -> dict:
    """Action Layer REST 서버로 요청을 프록시한다."""
    url = f"{_ACTION_URL.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            call = getattr(client, method)
            resp = await (call(url, json=payload) if payload is not None else call(url))
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Action Layer 연결 실패: {exc}",
        ) from exc


@router.get(
    "",
    response_model=BaseResponse[List[SiteOut]],
    summary="사이트 목록 조회",
)
async def list_sites(_: None = Depends(verify_api_key)) -> BaseResponse[List[SiteOut]]:
    raw = await _proxy("get", "/sites")
    sites = [SiteOut(**s) for s in (raw if isinstance(raw, list) else [])]
    return BaseResponse(success=True, data=sites)


@router.post(
    "",
    response_model=BaseResponse[dict],
    status_code=status.HTTP_201_CREATED,
    summary="사이트 등록",
)
async def create_site(
    body: SiteCreateIn,
    _: None = Depends(verify_api_key),
) -> BaseResponse[dict]:
    result = await _proxy("post", "/sites", body.model_dump())
    return BaseResponse(success=True, data=result)


@router.delete(
    "/{site_id}",
    response_model=BaseResponse[dict],
    summary="사이트 삭제",
)
async def delete_site(
    site_id: str = Path(min_length=1),
    _: None = Depends(verify_api_key),
) -> BaseResponse[dict]:
    result = await _proxy("delete", f"/sites/{site_id}")
    return BaseResponse(success=True, data=result)
