from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional, Tuple
from urllib.parse import urlparse

SUPPORTED_FILTERS = frozenset(
    {
        "camera_id",
        "upper_color",
        "lower_color",
        "has_helmet",
        "helmet_color",
        "has_backpack",
        "has_handbag",
        "has_suitcase",
        "gender",
        "age_group",
        "face_name",
    }
)
VALID_STATUSES = frozenset(
    {"accepted", "running", "completed", "failed", "expired", "rate_limited"}
)


class CommandValidationError(ValueError):
    """AIoT 명령 계약이 유효하지 않을 때 발생한다."""


@dataclass(frozen=True)
class AiQueryRequest:
    request_id: str
    jetson_id: str
    camera_ids: Tuple[str, ...]
    search_mode: Literal["live", "history", "both"]
    filters: Mapping[str, Any]
    query_text: Optional[str]
    time_from: Optional[float]
    time_to: Optional[float]
    limit: int
    expires_at: datetime


@dataclass(frozen=True)
class FetchMediaRequest:
    request_id: str
    parent_request_id: str
    match_ids: Tuple[str, ...]
    media_kind: Literal["snapshot", "clip"]
    upload_url: str
    max_bytes: int
    expires_at: datetime


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise CommandValidationError(f"missing {key}")
    return value


def _parse_expiry(value: Any, now: datetime) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise CommandValidationError("missing expires_at")
    try:
        expires_at = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CommandValidationError("invalid expires_at") from exc
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= now:
        raise CommandValidationError("command expired")
    return expires_at


def _validate_envelope(payload: Mapping[str, Any], message_type: str) -> None:
    if payload.get("schema_version") != "1.0":
        raise CommandValidationError("unsupported schema_version")
    if payload.get("message_type") != message_type:
        raise CommandValidationError("invalid message_type")


def parse_ai_query_request(
    payload: Mapping[str, Any], *, now: datetime, max_results: int = 20
) -> AiQueryRequest:
    _validate_envelope(payload, "ai_query_request")
    target = payload.get("target")
    if not isinstance(target, Mapping):
        raise CommandValidationError("missing target")
    mode = str(payload.get("search_mode") or "")
    if mode not in {"live", "history", "both"}:
        raise CommandValidationError("invalid search_mode")
    filters = payload.get("filters") or {}
    if not isinstance(filters, Mapping):
        raise CommandValidationError("filters must be an object")
    unknown = sorted(set(filters) - SUPPORTED_FILTERS)
    if unknown:
        raise CommandValidationError(f"unsupported filters: {', '.join(unknown)}")
    limit = int(payload.get("limit", max_results))
    if not 1 <= limit <= max_results:
        raise CommandValidationError(f"limit must be between 1 and {max_results}")
    camera_ids = tuple(str(value) for value in target.get("camera_ids", ("*",)))
    return AiQueryRequest(
        request_id=_required_text(payload, "request_id"),
        jetson_id=_required_text(target, "jetson_id"),
        camera_ids=camera_ids,
        search_mode=mode,  # type: ignore[arg-type]
        filters=dict(filters),
        query_text=str(payload["query_text"]) if payload.get("query_text") else None,
        time_from=payload.get("time_range", {}).get("from"),
        time_to=payload.get("time_range", {}).get("to"),
        limit=limit,
        expires_at=_parse_expiry(payload.get("expires_at"), now),
    )


def parse_fetch_media_request(
    payload: Mapping[str, Any], *, now: datetime
) -> FetchMediaRequest:
    _validate_envelope(payload, "fetch_media_request")
    upload_url = _required_text(payload, "upload_url")
    if urlparse(upload_url).scheme.lower() != "https":
        raise CommandValidationError("upload_url must use HTTPS")
    match_ids = tuple(str(value) for value in payload.get("match_ids") or ())
    if not match_ids:
        raise CommandValidationError("match_ids must not be empty")
    if len(match_ids) != 1:
        raise CommandValidationError("exactly one match_id is required per upload_url")
    media_kind = str(payload.get("media_kind") or "")
    if media_kind not in {"snapshot", "clip"}:
        raise CommandValidationError("invalid media_kind")
    max_bytes = int(payload.get("max_bytes") or 0)
    if max_bytes <= 0:
        raise CommandValidationError("max_bytes must be positive")
    return FetchMediaRequest(
        request_id=_required_text(payload, "request_id"),
        parent_request_id=_required_text(payload, "parent_request_id"),
        match_ids=match_ids,
        media_kind=media_kind,  # type: ignore[arg-type]
        upload_url=upload_url,
        max_bytes=max_bytes,
        expires_at=_parse_expiry(payload.get("expires_at"), now),
    )


def build_command_result(
    request_id: str, status: str, **fields: Any
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise CommandValidationError("invalid command status")
    return {
        "schema_version": "1.0",
        "message_type": "ai_command_result",
        "request_id": request_id,
        "status": status,
        **fields,
    }
