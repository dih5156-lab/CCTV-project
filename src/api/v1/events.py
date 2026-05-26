"""GET /api/v1/events — 탐지 이벤트 조회 엔드포인트.

내부 cctv-alert-api의 JSONL 파일을 읽어 페이지네이션 응답으로 제공한다.
서버팀 대시보드나 모니터링 시스템에서 폴링할 때 사용한다.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request

from ..dependencies._settings import ALERT_LOG_PATH as _ALERT_LOG
from ..dependencies.auth import verify_api_key
from ..dependencies.rate_limit import limiter
from ..schemas.common import PaginatedResponse
from ..schemas.event import EventOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["events"])


def _coerce_timestamp(value: object, fallback: object = None) -> float:
    """로그 timestamp를 API 응답용 Unix seconds로 정규화한다."""
    for candidate in (value, fallback):
        if candidate in (None, ""):
            continue
        if isinstance(candidate, (int, float)):
            return float(candidate)
        if isinstance(candidate, str):
            try:
                return float(candidate)
            except ValueError:
                try:
                    normalized = candidate.replace("Z", "+00:00")
                    return datetime.fromisoformat(normalized).timestamp()
                except ValueError:
                    continue
    return 0.0


def _event_payload(entry: dict) -> dict:
    """Alert API JSONL 엔트리에서 실제 이벤트 본문을 꺼낸다."""
    payload = entry.get("payload", entry)
    if isinstance(payload, dict) and isinstance(payload.get("event"), dict):
        return payload["event"]
    return payload if isinstance(payload, dict) else {}


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

    # 최대 (limit + offset) * 10 또는 5000라인 중 큰 값 tail 읽기
    _TAIL_MAX = max(5000, (limit + offset) * 10)

    all_items: list[dict] = []
    try:
        with _ALERT_LOG.open("r", encoding="utf-8") as fh:
            last_lines = deque(fh, maxlen=_TAIL_MAX)

        for line in reversed(last_lines):  # 최신 이벤트 우선
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                payload = _event_payload(entry)
                raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
                event_meta = payload.get("event") if isinstance(payload.get("event"), dict) else {}
                ts = _coerce_timestamp(
                    payload.get("timestamp"),
                    entry.get("receivedAt"),
                )
                if time_from is not None and ts < time_from:
                    continue
                if time_to is not None and ts > time_to:
                    continue
                etype = (
                    payload.get("type")
                    or payload.get("event_type")
                    or event_meta.get("event_type")
                    or "other"
                )
                if event_type is not None and etype != event_type:
                    continue
                cam = payload.get("camera_id")
                if camera_id is not None and cam != camera_id:
                    continue
                all_items.append(
                    {
                        "camera_id": cam,
                        "event_type": etype,
                        "severity": payload.get("severity", event_meta.get("severity", "normal")),
                        "confidence": payload.get("confidence", event_meta.get("confidence", 0.0)),
                        "timestamp": ts,
                        "bbox": payload.get("bbox") or raw.get("bbox"),
                        "object_id": payload.get("object_id"),
                        "metadata": payload.get("metadata") or raw.get("metadata"),
                        "received_at": entry.get("receivedAt"),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
    except OSError as exc:
        logger.error("이벤트 로그 읽기 실패: %s", exc)
        return [], 0

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
async def list_events(
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
        limit,
        offset,
        camera_id,
        time_from=time_from,
        time_to=time_to,
        event_type=event_type,
    )
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)
