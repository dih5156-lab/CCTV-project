"""Event review endpoints for operator false-positive labeling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...services.event_review import EventReviewStore
from ..dependencies.auth import verify_api_key
from ..schemas.common import BaseResponse, success_response

router = APIRouter(prefix="/event-reviews", tags=["event-reviews"])


class EventReviewIn(BaseModel):
    event_id: str = Field(min_length=1, max_length=160)
    status: str = Field(description="true_positive, false_positive, uncertain")
    reviewer: Optional[str] = Field(default=None, max_length=80)
    category: Optional[str] = Field(default=None, max_length=80)
    note: Optional[str] = Field(default=None, max_length=500)
    event: Optional[Dict[str, Any]] = None


class EventReviewOut(BaseModel):
    id: int
    event_id: str
    reviewed_at: str
    reviewer: Optional[str] = None
    status: str
    category: Optional[str] = None
    note: Optional[str] = None
    camera_id: Optional[str] = None
    event_type: Optional[str] = None
    event_timestamp: Optional[float] = None
    object_id: Optional[str] = None


class EventReviewSummary(BaseModel):
    total: int
    by_status: Dict[str, int]
    by_event_type: List[Dict[str, Any]]
    recent: List[EventReviewOut]


_store: Optional[EventReviewStore] = None


def _get_store() -> EventReviewStore:
    global _store
    if _store is None:
        _store = EventReviewStore()
    return _store


@router.post(
    "",
    response_model=BaseResponse[EventReviewOut],
    summary="이벤트 검수 결과 저장",
    description="탐지 이벤트를 맞음/오탐/애매함으로 검수하고 별도 SQLite DB에 저장합니다.",
)
async def upsert_event_review(
    body: EventReviewIn,
    _: None = Depends(verify_api_key),
) -> BaseResponse[EventReviewOut]:
    try:
        review = _get_store().upsert(
            event_id=body.event_id,
            status=body.status,
            reviewer=body.reviewer,
            category=body.category,
            note=body.note,
            event=body.event,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return success_response(EventReviewOut(**review))


@router.get(
    "/summary",
    response_model=BaseResponse[EventReviewSummary],
    summary="이벤트 검수 요약",
    description="검수 누적 건수와 상태별/이벤트 타입별 집계를 반환합니다.",
)
async def get_event_review_summary(
    _: None = Depends(verify_api_key),
) -> BaseResponse[EventReviewSummary]:
    return success_response(EventReviewSummary(**_get_store().summary()))
