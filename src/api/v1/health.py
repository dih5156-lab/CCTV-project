"""GET /api/v1/health — 시스템 상태 엔드포인트."""

from __future__ import annotations

import asyncio
import os
import resource
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..dependencies._settings import ACTION_LAYER_URL, ALERT_API_URL
from ..schemas.common import BaseResponse, success_response

router = APIRouter(tags=["health"])

# 공유 httpx 클라이언트 — 매 요청마다 SSL 컨텍스트를 새로 여는 fd 누출 방지
_http_client: httpx.AsyncClient | None = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=2.0,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


def _fd_usage() -> dict:
    """현재 프로세스의 file descriptor 사용량을 반환한다."""
    try:
        open_fds = len(os.listdir("/proc/self/fd"))
    except OSError:
        open_fds = None

    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        soft_limit, hard_limit = None, None

    if soft_limit == resource.RLIM_INFINITY:
        soft_limit = None
    if hard_limit == resource.RLIM_INFINITY:
        hard_limit = None

    status = "unknown"
    usage_ratio = None
    remaining = None
    if open_fds is not None and soft_limit:
        usage_ratio = round(open_fds / soft_limit, 4)
        remaining = soft_limit - open_fds
        status = "critical" if usage_ratio >= 0.9 or remaining <= 32 else "ok"
    elif open_fds is not None:
        status = "ok"

    return {
        "status": status,
        "open": open_fds,
        "soft_limit": soft_limit,
        "hard_limit": hard_limit,
        "usage_ratio": usage_ratio,
        "remaining": remaining,
    }


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
            "resources": {
                "file_descriptors": _fd_usage(),
            },
        }
    )


async def _check_dependency(name: str, url: str) -> dict:
    """하위 서비스 health endpoint를 짧은 timeout으로 확인한다."""
    try:
        client = await get_http_client()
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
    fd_usage = _fd_usage()
    dependencies = await asyncio.gather(
        _check_dependency("action-layer", f"{ACTION_LAYER_URL.rstrip('/')}/health"),
        _check_dependency("alert-api", f"{ALERT_API_URL.rstrip('/')}/health"),
    )
    ready = all(dep["status"] == "up" for dep in dependencies) and fd_usage["status"] != "critical"
    payload = {
        "success": ready,
        "data": {
            "status": "ready" if ready else "degraded",
            "service": "cctv-public-api",
            "checked_at": checked_at,
            "dependencies": dependencies,
            "resources": {
                "file_descriptors": fd_usage,
            },
        },
        "error": None if ready else "하위 서비스 또는 리소스 준비 상태 확인에 실패했습니다.",
        "timestamp": checked_at,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)
