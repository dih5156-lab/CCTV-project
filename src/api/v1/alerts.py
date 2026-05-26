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

from ..dependencies._settings import (
    ACTION_LAYER_URL as _ACTION_LAYER_URL,
    ALERT_API_URL as _ALERT_API_URL,
    ALERT_FALLBACK_LOG as _FALLBACK_LOG,
    INTERNAL_SERVICE_TOKEN as _INTERNAL_TOKEN,
)
from ..dependencies.auth import verify_api_key
from ..dependencies.rate_limit import limiter
from ..schemas.common import BaseResponse, success_response
from ..schemas.event import AlertAccepted, AlertIn

_INTERNAL_HEADERS: dict[str, str] = (
    {"X-Internal-Token": _INTERNAL_TOKEN} if _INTERNAL_TOKEN else {}
)

# 요청마다 새 TCP 연결을 만들지 않도록 모듈 레벨 공유 클라이언트를 사용한다.
_shared_alert_client: httpx.AsyncClient | None = None


def _get_alert_client() -> httpx.AsyncClient:
    global _shared_alert_client
    if _shared_alert_client is None or _shared_alert_client.is_closed:
        _shared_alert_client = httpx.AsyncClient(
            timeout=5.0,
            headers=_INTERNAL_HEADERS,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
    return _shared_alert_client


async def close_alert_client() -> None:
    """Public API 종료 시 내부 alert/action 중계 HTTP 클라이언트를 닫는다."""
    global _shared_alert_client
    if _shared_alert_client is not None and not _shared_alert_client.is_closed:
        await _shared_alert_client.aclose()
    _shared_alert_client = None

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

    # 1) alert-api로 JSONL 저장 중계
    target = f"{_ALERT_API_URL.rstrip('/')}/api/alerts"
    try:
        client = _get_alert_client()
        resp = await client.post(target, json=payload)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("내부 alert-api 중계 실패 (%s) → fallback 저장", exc)
        _save_fallback(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("예상치 못한 오류: %s", exc)
        _save_fallback(payload)

    # 2) action layer로 전달 → 전광판/스피커/사이렌 실행
    action_payload = {
        "camera_id": body.camera_id,
        "type": body.event_type.value,
        "severity": body.severity.value if body.severity else "",
        "confidence": body.confidence,
    }
    action_target = f"{_ACTION_LAYER_URL.rstrip('/')}/events"
    try:
        resp2 = await client.post(action_target, json=action_payload)
        if resp2.status_code not in (200, 202):
            logger.warning("action layer 전달 실패 (status=%s)", resp2.status_code)
        else:
            logger.info("action layer 전달 완료: %s/%s", body.camera_id, body.event_type.value)
    except httpx.HTTPError as exc:
        logger.warning("action layer 전달 실패 (%s)", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("action layer 전달 오류: %s", exc)

    return success_response(
        AlertAccepted(
            accepted=True,
            event_type=body.event_type.value,
            camera_id=body.camera_id,
        )
    )
