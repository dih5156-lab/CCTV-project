"""POST /api/v1/alerts — 외부 알림 수신 엔드포인트.

서버팀(또는 외부 시스템)에서 CCTV 탐지 이벤트를 push할 때 사용한다.
payload를 검증 후 내부 cctv-alert-api로 중계하고 JSONL에 저장한다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, Request, status

from ..dependencies._settings import ALERT_API_URL as _ALERT_API_URL, ALERT_FALLBACK_LOG as _FALLBACK_LOG
from ..dependencies.auth import verify_api_key
from ..dependencies.rate_limit import limiter
from ..dependencies._settings import INTERNAL_SERVICE_TOKEN as _INTERNAL_TOKEN
from ..schemas.common import BaseResponse, success_response
from ..schemas.event import AlertAccepted, AlertIn

_INTERNAL_HEADERS: dict[str, str] = (
    {"X-Internal-Token": _INTERNAL_TOKEN} if _INTERNAL_TOKEN else {}
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])


def _save_fallback(payload: dict) -> None:
    """내부 API 전달 실패 시 로컬 JSONL에 백업 저장한다."""
    try:
        _FALLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {"received_at": datetime.now(timezone.utc).isoformat(), "payload": payload}
        with _FALLBACK_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Fallback 저장 실패: %s", exc)


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
    payload = body.model_dump()

    # 내부 alert-api로 중계
    target = f"{_ALERT_API_URL.rstrip('/')}/api/alerts"
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=_INTERNAL_HEADERS) as client:
            resp = await client.post(target, json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("내부 alert-api 중계 실패 (%s) → fallback 저장", exc)
        _save_fallback(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("예상치 못한 오류: %s", exc)
        _save_fallback(payload)

    return success_response(
        AlertAccepted(
            accepted=True,
            event_type=body.event_type.value,
            camera_id=body.camera_id,
        )
    )
