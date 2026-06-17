from src.canonical_event import SKIP_ALERT_FORWARD_METADATA_KEY
from src.event_routing import (
    ALERT_STORAGE_OWNER_METADATA_KEY,
    PUBLIC_API_ALERT_STORAGE_OWNER,
    decide_alert_forward,
    mark_alert_stored_by_public_api,
)


def test_mark_alert_stored_by_public_api_preserves_metadata():
    metadata = mark_alert_stored_by_public_api({"zone_id": "zone-A"})

    assert metadata["zone_id"] == "zone-A"
    assert metadata[SKIP_ALERT_FORWARD_METADATA_KEY] is True
    assert metadata[ALERT_STORAGE_OWNER_METADATA_KEY] == PUBLIC_API_ALERT_STORAGE_OWNER


def test_decide_alert_forward_skips_already_stored_payload():
    decision = decide_alert_forward(
        {"metadata": {SKIP_ALERT_FORWARD_METADATA_KEY: True}},
        has_targets=True,
    )

    assert decision.should_forward is False
    assert decision.http_sent is False
    assert decision.reason == "already_stored"


def test_decide_alert_forward_uses_configured_targets():
    decision = decide_alert_forward({"type": "head"}, has_targets=True)

    assert decision.should_forward is True
    assert decision.http_sent is True
    assert decision.reason == "forward_targets_configured"


def test_decide_alert_forward_handles_missing_targets():
    decision = decide_alert_forward({"type": "head"}, has_targets=False)

    assert decision.should_forward is False
    assert decision.http_sent is False
    assert decision.reason == "no_forward_targets"
