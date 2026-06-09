"""프로젝트 공통 한국 표준시(KST) 유틸리티."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite

KST = timezone(timedelta(hours=9), name="KST")


def now_kst() -> datetime:
    """현재 시각을 KST 기준 timezone-aware datetime으로 반환한다."""
    return datetime.now(KST)


def now_kst_iso() -> str:
    """현재 시각을 KST 오프셋이 포함된 ISO 8601 문자열로 반환한다."""
    return now_kst().isoformat()


def timestamp_to_kst_iso(timestamp: float) -> str:
    """Unix epoch 초를 KST ISO 8601 문자열로 변환한다."""
    return datetime.fromtimestamp(timestamp, tz=KST).isoformat()


def _parse_single_candidate(candidate: object) -> float | None:
    """단일 후보값을 Unix epoch 초로 파싱 시도한다."""
    if candidate in (None, ""):
        return None
    if isinstance(candidate, (int, float)):
        return _normalize_epoch_number(float(candidate))
    if isinstance(candidate, str):
        try:
            return _normalize_epoch_number(float(candidate))
        except ValueError:
            try:
                # Python 3.11+ 에서는 "Z"를 기본 지원하지만, 하위 호환성을 위해 유지
                return datetime.fromisoformat(candidate.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
    return None


def coerce_timestamp_seconds(value: object, fallback: object = None) -> float:
    """숫자/문자열/ISO timestamp를 Unix epoch 초로 정규화한다.

    센서 로그처럼 밀리초 epoch가 들어오는 경우도 초 단위로 변환한다.
    변환할 수 없으면 0.0을 반환한다.
    """
    for candidate in (value, fallback):
        result = _parse_single_candidate(candidate)
        if result is not None:
            return result
    return 0.0


def _normalize_epoch_number(value: float) -> float | None:
    if not isfinite(value):
        return None
    return value / 1000.0 if abs(value) >= 1e11 else value
