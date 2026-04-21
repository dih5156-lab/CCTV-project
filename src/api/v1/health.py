"""GET /api/v1/health — 시스템 상태 엔드포인트."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from ..dependencies._settings import ACTION_LAYER_URL, ALERT_API_URL
from ..schemas.common import BaseResponse, success_response

router = APIRouter(tags=["health"])


@router.get("/health", summary="시스템 상태 확인")
def get_health() -> BaseResponse[dict]:
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
