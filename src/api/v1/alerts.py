"""POST /api/v1/alerts — 외부 알림 수신 엔드포인트.

서버팀(또는 외부 시스템)에서 CCTV 탐지 이벤트를 push할 때 사용한다.
payload를 검증 후 내부 cctv-alert-api로 중계하고 JSONL에 저장한다.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, status

from .._event_forwarding import close_alert_forwarding_client, forward_alert_event
from ..dependencies._settings import (
    ACTION_LAYER_URL as _ACTION_LAYER_URL,
)
from ..dependencies._settings import (
    ALERT_API_URL as _ALERT_API_URL,
)
from ..dependencies._settings import (
    ALERT_FALLBACK_LOG as _FALLBACK_LOG,
)
from ..dependencies.auth import verify_api_key
from ..dependencies.rate_limit import limiter
from ..schemas.common import BaseResponse, success_response
from ..schemas.event import AlertAccepted, AlertIn


async def close_alert_client() -> None:
    """Public API 종료 시 내부 alert/action 중계 HTTP 클라이언트를 닫는다."""
    await close_alert_forwarding_client()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])



@router.post(
    "",
    response_model=BaseResponse[AlertAccepted],
    status_code=status.HTTP_202_ACCEPTED,
    summary="탐지 이벤트 알림 수신",
    description=(
        "CCTV AI 엔진 또는 외부 시스템에서 탐지된 이벤트를 수신합니다. "
        "내부 cctv-alert-api로 중계되며 감사 로그에 기록됩니다."
    ),
)
@limiter.limit("30/minute")
async def receive_alert(
    request: Request,
    body: AlertIn,
    _: None = Depends(verify_api_key),
) -> BaseResponse[AlertAccepted]:
    await forward_alert_event(
        body,
        alert_api_url=_ALERT_API_URL,
        action_layer_url=_ACTION_LAYER_URL,
        fallback_log=_FALLBACK_LOG,
        logger=logger,
    )

    return success_response(
        AlertAccepted(
            accepted=True,
            event_type=body.event_type.value,
            camera_id=body.camera_id,
        )
    )
