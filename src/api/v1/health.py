"""GET /api/v1/health — 시스템 상태 엔드포인트."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..dependencies._settings import ACTION_LAYER_URL, ALERT_API_URL
from ..schemas.common import BaseResponse, success_response

router = APIRouter(tags=["health"])


@router.get("/health", summary="시스템 상태 확인")
async def get_health() -> BaseResponse[dict]:
    """서버팀이 서비스 가용성을 확인하는 엔드포인트."""
    return success_response(
        {
            "status": "up",
            "service": "cctv-public-api",
            "version": "1.0.0",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "action_layer_url": ACTION_LAYER_URL,
            "alert_api_url": ALERT_API_URL,
        }
    )


async def _check_dependency(name: str, url: str) -> dict:
    """하위 서비스 health endpoint를 짧은 timeout으로 확인한다."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url)
        ok = 200 <= response.status_code < 300
        return {
            "name": name,
            "url": url,
            "status": "up" if ok else "down",
            "status_code": response.status_code,
        }
    except httpx.TimeoutException:
        return {
            "name": name,
            "url": url,
            "status": "down",
            "error": "timeout",
        }
    except httpx.HTTPError as exc:
        return {
            "name": name,
            "url": url,
            "status": "down",
            "error": type(exc).__name__,
        }


@router.get("/readiness", summary="Public API 의존 서비스 준비 상태 확인")
async def get_readiness() -> JSONResponse:
    """Action Layer와 Alert API까지 포함해 운영 준비 상태를 확인한다."""
    checked_at = datetime.now(timezone.utc).isoformat()
    dependencies = [
        await _check_dependency("action-layer", f"{ACTION_LAYER_URL.rstrip('/')}/health"),
        await _check_dependency("alert-api", f"{ALERT_API_URL.rstrip('/')}/health"),
    ]
    ready = all(dep["status"] == "up" for dep in dependencies)
    payload = {
        "success": ready,
        "data": {
            "status": "ready" if ready else "degraded",
            "service": "cctv-public-api",
            "checked_at": checked_at,
            "dependencies": dependencies,
        },
        "error": None if ready else "하위 서비스 준비 상태 확인에 실패했습니다.",
        "timestamp": checked_at,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)
