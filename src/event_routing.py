"""이벤트 저장/전달 경로 정책."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical_event import SKIP_ALERT_FORWARD_METADATA_KEY, should_skip_alert_forward

ALERT_STORAGE_OWNER_METADATA_KEY = "alert_storage_owner"
PUBLIC_API_ALERT_STORAGE_OWNER = "public_api"


@dataclass(frozen=True)
class AlertForwardDecision:
    """Action Layer가 Alert API로 재전송할지 결정한 결과."""

    should_forward: bool
    http_sent: bool
    reason: str


def mark_alert_stored_by_public_api(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Public API가 이미 Alert API 저장을 담당한 이벤트임을 표시한다."""
    marked = dict(metadata or {})
    marked[SKIP_ALERT_FORWARD_METADATA_KEY] = True
    marked[ALERT_STORAGE_OWNER_METADATA_KEY] = PUBLIC_API_ALERT_STORAGE_OWNER
    return marked


def decide_alert_forward(payload: Mapping[str, Any], *, has_targets: bool) -> AlertForwardDecision:
    """Action Layer에서 Alert API 재전송 여부를 결정한다."""
    if should_skip_alert_forward(payload):
        return AlertForwardDecision(
            should_forward=False,
            http_sent=False,
            reason="already_stored",
        )
    return AlertForwardDecision(
        should_forward=has_targets,
        http_sent=has_targets,
        reason="forward_targets_configured" if has_targets else "no_forward_targets",
    )
