"""control API 단위 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.api.v1.control import _normalize_pending_item, list_pending


def test_normalize_pending_item_supports_payload_fallbacks() -> None:
    item = _normalize_pending_item(
        {
            "payload": {
                "eventId": "evt-100",
                "cameraId": "cam-01",
                "type": "helmet",
                "severity": "high",
            },
            "queuedAt": "2026-04-24T10:00:00Z",
            "siteId": "site-a",
            "topic": "cctv/ai/events/cam-01/helmet",
        }
    )

    assert item.event_id == "evt-100"
    assert item.camera_id == "cam-01"
    assert item.event_type == "helmet"
    assert item.severity == "high"
    assert item.priority == 20
    assert item.risk_level == "normal"
    assert item.queued_at == "2026-04-24T10:00:00Z"
    assert item.site_id == "site-a"
    assert item.topic == "cctv/ai/events/cam-01/helmet"


def test_normalize_pending_item_supports_nested_event_confidence() -> None:
    item = _normalize_pending_item(
        {
            "payload": {
                "eventId": "evt-101",
                "cameraId": "cam-01",
                "type": "head",
                "event": {
                    "confidence": 0.87,
                    "severity": "warning",
                    "display_message": "안전모 미착용 확인",
                    "tts_message": "안전모를 착용해 주세요.",
                },
            },
        }
    )

    assert item.confidence == 0.87
    assert item.severity == "warning"
    assert item.priority == 2
    assert item.risk_level == "warning"
    assert item.display_message == "안전모 미착용 확인"
    assert item.tts_message == "안전모를 착용해 주세요."


@pytest.mark.asyncio
async def test_list_pending_returns_normalized_schema() -> None:
    raw_items = [
        {
            "event_id": "evt-1",
            "camera_id": "cam-01",
            "event_type": "head",
            "severity": "critical",
            "queued_at": "2026-04-24T09:00:00Z",
            "site_id": "site-01",
            "topic": "cctv/ai/events/cam-01/head",
        },
        {
            "payload": {
                "eventId": "evt-2",
                "cameraId": "cam-02",
                "type": "fall_detected",
                "severity": "critical",
            },
            "queuedAt": "2026-04-24T09:05:00Z",
        },
        "ignore-me",
    ]

    with patch(
        "src.api.v1.control.proxy_action_request",
        new=AsyncMock(return_value=raw_items),
    ):
        response = await list_pending(None)

    assert response.success is True
    assert response.data is not None
    assert len(response.data) == 2

    first = response.data[0]
    assert first.event_id == "evt-1"
    assert first.camera_id == "cam-01"
    assert first.event_type == "head"

    second = response.data[1]
    assert second.event_id == "evt-2"
    assert second.camera_id == "cam-02"
    assert second.event_type == "fall_detected"
    assert second.priority == 0
    assert second.risk_level == "critical"
    assert second.queued_at == "2026-04-24T09:05:00Z"
