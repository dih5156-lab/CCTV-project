"""저장·발행 경계에서 재사용하는 이벤트 식별/시간 필드 검증."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping

from .time_utils import timestamp_to_kst_iso


class EventValidationError(ValueError):
    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        super().__init__(
            ", ".join(f"{item['field']}:{item['code']}" for item in errors)
        )


def _timestamp_error(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return "invalid_timestamp"
    text = str(value).strip()
    if not text:
        return "invalid_timestamp"
    try:
        numeric = float(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(
                text[:-1] + "+00:00" if text.endswith("Z") else text
            )
        except ValueError:
            return "invalid_timestamp"
        return None if parsed.utcoffset() is not None else "timezone_required"

    if not math.isfinite(numeric):
        return "invalid_timestamp"
    # 기존 canonical 변환과 같은 초/ms/μs 기준으로 변환 가능 여부를 검사한다.
    if abs(numeric) >= 1e14:
        numeric /= 1_000_000
    elif abs(numeric) >= 1e11:
        numeric /= 1_000
    try:
        timestamp_to_kst_iso(numeric)
    except (ValueError, OverflowError, OSError):
        return "invalid_timestamp"
    return None


def validate_event_identity(
    payload: Mapping[str, Any], *, device_id: Any, occurred_at: Any
) -> None:
    """생산자 필드를 수정하지 않고 오류 필드·사유를 예외로 전달한다."""
    errors = []
    valid_device = isinstance(device_id, (str, int)) and not isinstance(device_id, bool)
    if (
        not valid_device
        or not str(device_id).strip()
        or str(device_id).strip().lower() == "unknown"
    ):
        errors.append({"field": "device_id", "code": "invalid_device_id"})
    if "event_id" in payload:
        event_id = payload["event_id"]
        if not isinstance(event_id, str) or not event_id.strip():
            errors.append({"field": "event_id", "code": "invalid_event_id"})
    if "schema_version" in payload and payload["schema_version"] != "1.0":
        errors.append({"field": "schema_version", "code": "unsupported_schema_version"})
    timestamp_error = _timestamp_error(occurred_at)
    if timestamp_error:
        errors.append({"field": "occurred_at", "code": timestamp_error})
    if errors:
        raise EventValidationError(errors)
