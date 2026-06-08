"""AIoT parser 공통 한국 표준시(KST) 유틸리티."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9), name="KST")


def now_kst() -> datetime:
    """현재 시각을 KST 기준 timezone-aware datetime으로 반환한다."""
    return datetime.now(KST)


def timestamp_ms_to_kst(timestamp_ms: int) -> datetime:
    """Unix epoch 밀리초를 KST datetime으로 변환한다."""
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=KST)
