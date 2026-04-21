"""이벤트/알림 관련 Pydantic 스키마."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# EventType 을 str,Enum 으로 직접 재사용 — 중복 정의 제거
from ...core.events import EventType as EventTypeOut  # noqa: F401  (re-export)


# ---------------------------------------------------------------------------
# Enum re-exports (서버팀은 이 값들을 사용한다)
# ---------------------------------------------------------------------------


class SeverityOut(str, Enum):
    CRITICAL = "critical"
    NORMAL = "normal"


# ---------------------------------------------------------------------------
# 요청 스키마
# ---------------------------------------------------------------------------


class BboxIn(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class AlertIn(BaseModel):
    """POST /api/v1/alerts 요청 바디."""

    camera_id: str = Field(min_length=1, max_length=128)
    event_type: EventTypeOut
    severity: SeverityOut = SeverityOut.NORMAL
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: float = Field(description="Unix epoch (초)")
    bbox: Optional[BboxIn] = None
    object_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 응답 스키마
# ---------------------------------------------------------------------------


class BboxOut(BaseModel):
    x: int
    y: int
    width: int
    height: int


class EventOut(BaseModel):
    """단일 탐지 이벤트 응답."""

    id: Optional[int] = None
    camera_id: Optional[str] = None
    event_type: str
    severity: str
    confidence: float
    timestamp: float
    bbox: Optional[BboxOut] = None
    object_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    received_at: Optional[datetime] = None


class AlertAccepted(BaseModel):
    """POST /api/v1/alerts 성공 응답."""

    accepted: bool = True
    event_type: str
    camera_id: str
