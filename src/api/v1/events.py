"""GET /api/v1/events — 탐지 이벤트 조회 엔드포인트.

내부 cctv-alert-api의 JSONL 파일을 읽어 페이지네이션 응답으로 제공한다.
서버팀 대시보드나 모니터링 시스템에서 폴링할 때 사용한다.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request

from ..dependencies._settings import ALERT_LOG_PATH as _ALERT_LOG
from ..dependencies.auth import verify_api_key
from ..dependencies.rate_limit import limiter
from ..schemas.common import PaginatedResponse
from ..schemas.event import EventOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["events"])


def _read_events(
    limit: int,
    offset: int,
    camera_id: Optional[str],
    time_from: Optional[float] = None,
    time_to: Optional[float] = None,
    event_type: Optional[str] = None,
) -> tuple[List[EventOut], int]:
    """JSONL 파일에서 이벤트를 읽어 필터링·페이지네이션한다."""
    if not _ALERT_LOG.exists():
        return [], 0

    all_items: list[dict] = []
    try:
        lines = _ALERT_LOG.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):  # 최신 이벤트 우선
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                payload = entry.get("payload", entry)
                ts = payload.get("timestamp", 0.0)
                if time_from is not None and ts < time_from:
                    continue
                if time_to is not None and ts > time_to:
                    continue
                etype = payload.get("type", payload.get("event_type", "other"))
                if event_type is not None and etype != event_type:
                    continue
                all_items.append(
                    {
                        "camera_id": payload.get("camera_id"),
                        "event_type": etype,
                        "severity": payload.get("severity", "normal"),
                        "confidence": payload.get("confidence", 0.0),
                        "timestamp": ts,
                        "bbox": payload.get("bbox"),
                        "object_id": payload.get("object_id"),
                        "metadata": payload.get("metadata"),
                        "received_at": entry.get("receivedAt"),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
    except OSError as exc:
        logger.error("이벤트 로그 읽기 실패: %s", exc)
        return [], 0

    if camera_id:
        all_items = [e for e in all_items if e.get("camera_id") == camera_id]

    total = len(all_items)
    paged = all_items[offset : offset + limit]
    return [EventOut(**item) for item in paged], total


@router.get(
    "",
    response_model=PaginatedResponse[EventOut],
    summary="탐지 이벤트 목록 조회",
    description=(
        "최신 순으로 탐지 이벤트를 반환합니다. "
        "camera_id, event_type, 시간 범위(time_from/time_to)로 필터링 가능합니다."
    ),
)
@limiter.limit("60/minute")
def list_events(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500, description="페이지 크기"),
    offset: int = Query(default=0, ge=0, description="시작 오프셋"),
    camera_id: Optional[str] = Query(default=None, description="특정 카메라 필터"),
    event_type: Optional[str] = Query(default=None, description="이벤트 타입 필터 (예: fall_detected, helmet)"),
    time_from: Optional[float] = Query(default=None, description="시작 Unix timestamp (초 단위)"),
    time_to: Optional[float] = Query(default=None, description="종료 Unix timestamp (초 단위)"),
    _: None = Depends(verify_api_key),
) -> PaginatedResponse[EventOut]:
    items, total = _read_events(
        limit, offset, camera_id,
        time_from=time_from,
        time_to=time_to,
        event_type=event_type,
    )
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)
