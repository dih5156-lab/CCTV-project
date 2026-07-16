from datetime import datetime, timedelta, timezone

import pytest

from src.aiot.contracts import (
    CommandValidationError,
    build_command_result,
    parse_ai_query_request,
    parse_fetch_media_request,
)

NOW = datetime(2026, 7, 16, tzinfo=timezone.utc)
FUTURE = (NOW + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")


def test_parse_ai_query_request_accepts_both_mode():
    request = parse_ai_query_request(
        {
            "schema_version": "1.0",
            "message_type": "ai_query_request",
            "request_id": "q-1",
            "target": {"jetson_id": "edge-01", "camera_ids": ["camera-1"]},
            "search_mode": "both",
            "filters": {"gender": "female", "has_handbag": True},
            "limit": 20,
            "expires_at": FUTURE,
        },
        now=NOW,
    )
    assert request.request_id == "q-1"
    assert request.search_mode == "both"
    assert request.camera_ids == ("camera-1",)


def test_parse_ai_query_request_rejects_expired_command():
    with pytest.raises(CommandValidationError, match="expired"):
        parse_ai_query_request(
            {
                "schema_version": "1.0",
                "message_type": "ai_query_request",
                "request_id": "q-1",
                "target": {"jetson_id": "edge-01"},
                "search_mode": "history",
                "expires_at": (NOW - timedelta(seconds=1)).isoformat(),
            },
            now=NOW,
        )


def test_parse_ai_query_request_rejects_unknown_filter():
    with pytest.raises(CommandValidationError, match="unsupported filters"):
        parse_ai_query_request(
            {
                "schema_version": "1.0",
                "message_type": "ai_query_request",
                "request_id": "q-1",
                "target": {"jetson_id": "edge-01"},
                "search_mode": "history",
                "filters": {"secret_model_knob": 1},
                "expires_at": FUTURE,
            },
            now=NOW,
        )


def test_parse_fetch_media_request_requires_https():
    with pytest.raises(CommandValidationError, match="HTTPS"):
        parse_fetch_media_request(
            {
                "schema_version": "1.0",
                "message_type": "fetch_media_request",
                "request_id": "m-1",
                "parent_request_id": "q-1",
                "match_ids": ["match-1"],
                "media_kind": "snapshot",
                "upload_url": "http://server/upload",
                "max_bytes": 1024,
                "expires_at": FUTURE,
            },
            now=NOW,
        )


def test_parse_fetch_media_request_rejects_multiple_matches_for_one_url():
    with pytest.raises(CommandValidationError, match="exactly one"):
        parse_fetch_media_request(
            {
                "schema_version": "1.0",
                "message_type": "fetch_media_request",
                "request_id": "m-1",
                "parent_request_id": "q-1",
                "match_ids": ["match-1", "match-2"],
                "media_kind": "snapshot",
                "upload_url": "https://server/upload",
                "max_bytes": 1024,
                "expires_at": FUTURE,
            },
            now=NOW,
        )


def test_build_command_result_keeps_correlation_fields():
    result = build_command_result(
        "q-1", "completed", parent_request_id="parent-1", matches=[]
    )
    assert result["request_id"] == "q-1"
    assert result["parent_request_id"] == "parent-1"
    assert result["status"] == "completed"
