"""GET /api/v1/events — 탐지 이벤트 조회 엔드포인트.

내부 cctv-alert-api의 JSONL 파일을 읽어 페이지네이션 응답으로 제공한다.
서버팀 대시보드나 모니터링 시스템에서 폴링할 때 사용한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request

from ...canonical_event import (
    get_payload_camera_id,
    get_payload_confidence,
    get_payload_event_id,
    get_payload_event_type,
    get_payload_metadata,
    get_payload_severity,
)
from ...event_priority import event_priority, event_risk_level, event_risk_score
from ...services.event_review import EventReviewStore
from ...time_utils import coerce_timestamp_seconds
from ..dependencies._settings import ALERT_LOG_PATH as _ALERT_LOG
from ..dependencies.auth import verify_api_key
from ..dependencies.rate_limit import limiter
from ..schemas.common import PaginatedResponse
from ..schemas.event import EventOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["events"])
_ROTATED_LOG_KEEP = 5


def _event_payload(entry: dict) -> dict:
    """Alert API JSONL 엔트리에서 실제 이벤트 본문을 꺼낸다.

    저장 로그에는 두 형태가 섞일 수 있다.
    - 표준 payload: top-level ``event``는 이벤트 메타 정보
    - 래퍼 payload: ``{"topic": "...", "event": {...실제 이벤트...}}``
    """
    payload = entry.get("payload", entry)
    if not isinstance(payload, dict):
        return {}
    nested_event = payload.get("event")
    if (
        isinstance(nested_event, dict)
        and "topic" in payload
        and (
            "camera_id" in nested_event
            or "cameraId" in nested_event
            or "type" in nested_event
            or "event_type" in nested_event
            or "eventType" in nested_event
            or "raw" in nested_event
        )
    ):
        return nested_event
    return payload


def _event_item_from_entry(entry: dict) -> dict:
    payload = _event_payload(entry)
    raw_value = payload.get("raw")
    raw = raw_value if isinstance(raw_value, dict) else {}
    confidence = get_payload_confidence(payload)
    priority = event_priority(payload)
    return {
        "event_id": get_payload_event_id(payload),
        "camera_id": get_payload_camera_id(payload),
        "event_type": get_payload_event_type(payload),
        "severity": get_payload_severity(payload) or "normal",
        "confidence": confidence if confidence is not None else 0.0,
        "timestamp": coerce_timestamp_seconds(
            payload.get("timestamp") or payload.get("occurred_at"),
            entry.get("receivedAt"),
        ),
        "bbox": payload.get("bbox") or raw.get("bbox"),
        "object_id": payload.get("object_id", raw.get("object_id")),
        "metadata": get_payload_metadata(payload) or None,
        "received_at": entry.get("receivedAt"),
        "priority": priority,
        "risk_level": event_risk_level(payload),
        "risk_score": event_risk_score(payload),
    }


def _read_recent_log_lines(max_lines: int) -> list[str]:
    """활성 로그와 회전 로그를 합쳐 최신 라인부터 반환한다."""
    remaining = max_lines
    recent_lines: list[str] = []
    log_paths = [
        _ALERT_LOG,
        *[
            type(_ALERT_LOG)(f"{_ALERT_LOG}.{index}")
            for index in range(1, _ROTATED_LOG_KEEP + 1)
        ],
    ]

    for log_path in log_paths:
        if remaining <= 0:
            break
        if not log_path.exists():
            continue
        try:
            with log_path.open("r", encoding="utf-8") as fh:
                lines = deque(fh, maxlen=remaining)
        except OSError as exc:
            logger.warning("이벤트 로그 읽기 실패, 다음 파일 계속 확인: %s (%s)", log_path, exc)
            continue
        recent_lines.extend(reversed(lines))
        remaining -= len(lines)

    return recent_lines


def _matches_event_filters(
    item: dict,
    *,
    camera_id: Optional[str],
    event_type: Optional[str],
    time_from: Optional[float],
    time_to: Optional[float],
    fall_direction: Optional[str],
) -> bool:
    """단일 이벤트가 목록 조회 조건을 만족하는지 판단한다."""
    if time_from is not None and item["timestamp"] < time_from:
        return False
    if time_to is not None and item["timestamp"] > time_to:
        return False
    if event_type is not None and item["event_type"] != event_type:
        return False
    if camera_id is not None and item["camera_id"] != camera_id:
        return False
    if fall_direction is None:
        return True

    metadata = item.get("metadata") or {}
    direction = str(metadata.get("fall_direction") or "").lower()
    category = str(
        metadata.get("scene_cat_name") or metadata.get("fall_category") or ""
    ).lower()
    direction_alias = {"전면": "front", "후면": "back", "측면": "side"}.get(
        fall_direction, fall_direction
    )
    if direction_alias == "unclassified":
        return metadata.get("fall_detail_status") == "unclassified"
    return direction_alias == direction or (
        (direction_alias == "front" and "전면" in category)
        or (direction_alias == "back" and "후면" in category)
        or (direction_alias == "side" and "측면" in category)
    )


def _attach_review_annotations(items: list[dict]) -> None:
    """페이지에 반환할 이벤트에 검수 상태와 재계산 점수를 붙인다."""
    reviews = EventReviewStore().get_many(
        [str(item.get("event_id") or "") for item in items]
    )
    for item in items:
        review = reviews.get(str(item.get("event_id") or ""))
        if not review:
            continue
        item["review_status"] = review.get("status")
        item["reviewed_at"] = review.get("reviewed_at")
        item["risk_score"] = event_risk_score(
            item, review_status=str(review.get("status") or "")
        )


def _read_events(
    limit: int,
    offset: int,
    camera_id: Optional[str],
    time_from: Optional[float] = None,
    time_to: Optional[float] = None,
    event_type: Optional[str] = None,
    fall_direction: Optional[str] = None,
) -> tuple[List[EventOut], int]:
    """JSONL 파일에서 이벤트를 읽어 필터링·페이지네이션한다."""
    # 최대 (limit + offset) * 10 또는 5000라인 중 큰 값 tail 읽기
    _TAIL_MAX = max(5000, (limit + offset) * 10)

    all_items: list[dict] = []
    for line in _read_recent_log_lines(_TAIL_MAX):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            item = _event_item_from_entry(entry)
            if not _matches_event_filters(
                item,
                camera_id=camera_id,
                event_type=event_type,
                time_from=time_from,
                time_to=time_to,
                fall_direction=fall_direction,
            ):
                continue
            all_items.append(item)
        except (json.JSONDecodeError, KeyError):
            continue

    total = len(all_items)
    paged = all_items[offset : offset + limit]
    _attach_review_annotations(paged)
    return [EventOut(**item) for item in paged], total


@router.get(
    "",
    response_model=PaginatedResponse[EventOut],
    summary="탐지 이벤트 목록 조회",
    description=(
        "최신 순으로 탐지 이벤트를 반환합니다. "
        "camera_id, event_type, fall_direction, 시간 범위(time_from/time_to)로 필터링 가능합니다."
    ),
)
@limiter.limit("60/minute")
async def list_events(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500, description="페이지 크기"),
    offset: int = Query(default=0, ge=0, description="시작 오프셋"),
    camera_id: Optional[str] = Query(default=None, description="특정 카메라 필터"),
    event_type: Optional[str] = Query(default=None, description="이벤트 타입 필터 (예: fall_detected, helmet)"),
    fall_direction: Optional[str] = Query(
        default=None,
        pattern="^(front|back|side|전면|후면|측면|unclassified)$",
        description="낙상 방향 필터 (front/back/side 또는 전면/후면/측면)",
    ),
    time_from: Optional[float] = Query(default=None, description="시작 Unix timestamp (초 단위)"),
    time_to: Optional[float] = Query(default=None, description="종료 Unix timestamp (초 단위)"),
    _: None = Depends(verify_api_key),
) -> PaginatedResponse[EventOut]:
    items, total = await asyncio.to_thread(
        _read_events,
        limit,
        offset,
        camera_id,
        time_from=time_from,
        time_to=time_to,
        event_type=event_type,
        fall_direction=fall_direction,
    )
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)
