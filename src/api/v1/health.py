"""GET /api/v1/health — 시스템 상태 엔드포인트."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="시스템 상태 확인")
def get_health() -> dict:
    """서버팀이 서비스 가용성을 확인하는 엔드포인트."""
    return {
        "status": "up",
        "service": "cctv-public-api",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action_layer_url": os.environ.get("ACTION_LAYER_URL", "http://cctv-action-layer:8080"),
        "alert_api_url": os.environ.get("ALERT_API_URL", "http://cctv-alert-api:8000"),
    }
