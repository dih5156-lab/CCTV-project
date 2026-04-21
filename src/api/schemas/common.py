"""공통 응답 스키마.

서버팀과 공유하는 모든 엔드포인트에서 동일한 외형을 보장한다.

BaseResponse:  성공/실패 통합 래퍼
ErrorResponse: 오류 상세 포함 응답
PaginatedResponse: 페이지네이션 포함 목록 응답
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BaseResponse(BaseModel, Generic[T]):
    """전체 API 공통 응답 래퍼."""

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True}


class PaginatedResponse(BaseModel, Generic[T]):
    """페이지네이션 목록 응답."""

    success: bool = True
    items: List[T]
    total: int
    limit: int
    offset: int
    timestamp: datetime = Field(default_factory=_utcnow)


def success_response(data: T | None = None) -> BaseResponse[T]:
    """성공 응답 래퍼를 생성한다."""
    return BaseResponse(success=True, data=data)


def error_response(message: str) -> BaseResponse[Any]:
    """오류 응답 래퍼를 생성한다."""
    return BaseResponse(success=False, data=None, error=message)
