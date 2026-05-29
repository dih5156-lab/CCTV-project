"""Internal event forwarding helpers for Public API endpoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .dependencies._settings import INTERNAL_SERVICE_TOKEN as _INTERNAL_TOKEN
from .schemas.event import AlertIn

_INTERNAL_HEADERS: dict[str, str] = (
    {"X-Internal-Token": _INTERNAL_TOKEN} if _INTERNAL_TOKEN else {}
)

_shared_alert_client: httpx.AsyncClient | None = None


def get_alert_forwarding_client() -> httpx.AsyncClient:
    """Return the shared HTTP client used for alert storage/action forwarding."""
    global _shared_alert_client
    if _shared_alert_client is None or _shared_alert_client.is_closed:
        _shared_alert_client = httpx.AsyncClient(
            timeout=5.0,
            headers=_INTERNAL_HEADERS,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
    return _shared_alert_client


async def close_alert_forwarding_client() -> None:
    """Close the shared alert forwarding HTTP client."""
    global _shared_alert_client
    if _shared_alert_client is not None and not _shared_alert_client.is_closed:
        await _shared_alert_client.aclose()
    _shared_alert_client = None


def save_alert_fallback(payload: dict[str, Any], fallback_log: Path, logger: logging.Logger) -> None:
    """Save an alert payload to a local JSONL fallback file."""
    try:
        fallback_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {"received_at": datetime.now(timezone.utc).isoformat(), "payload": payload}
        with fallback_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Fallback 저장 실패: %s", exc)


def build_alert_action_payload(alert: AlertIn) -> dict[str, Any]:
    """Build the Action Layer payload for a validated alert request."""
    action_payload: dict[str, Any] = {
        "camera_id": alert.camera_id,
        "type": alert.event_type.value,
        "severity": alert.severity.value if alert.severity else "",
        "confidence": alert.confidence,
        "timestamp": alert.timestamp,
    }
    if alert.bbox is not None:
        action_payload["bbox"] = alert.bbox.model_dump()
    if alert.object_id is not None:
        action_payload["object_id"] = alert.object_id
    if alert.metadata is not None:
        action_payload["metadata"] = alert.metadata
    return action_payload


async def forward_alert_event(
    alert: AlertIn,
    *,
    alert_api_url: str,
    action_layer_url: str,
    fallback_log: Path,
    logger: logging.Logger,
) -> None:
    """Store a public alert and dispatch it to the Action Layer when possible."""
    payload = alert.model_dump()
    client = get_alert_forwarding_client()

    storage_target = f"{alert_api_url.rstrip('/')}/api/alerts"
    try:
        response = await client.post(storage_target, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("내부 alert-api 중계 실패 (%s) -> fallback 저장", exc)
        save_alert_fallback(payload, fallback_log, logger)
    except Exception as exc:  # noqa: BLE001
        logger.error("예상치 못한 오류: %s", exc)
        save_alert_fallback(payload, fallback_log, logger)

    action_target = f"{action_layer_url.rstrip('/')}/events"
    action_payload = build_alert_action_payload(alert)
    try:
        response = await client.post(action_target, json=action_payload)
        if response.status_code not in (200, 202):
            logger.warning("action layer 전달 실패 (status=%s)", response.status_code)
        else:
            logger.info("action layer 전달 완료: %s/%s", alert.camera_id, alert.event_type.value)
    except httpx.HTTPError as exc:
        logger.warning("action layer 전달 실패 (%s)", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("action layer 전달 오류: %s", exc)
