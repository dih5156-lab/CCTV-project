"""Arrival time must never turn an old or untrusted sensor value into fresh data."""

import copy

import pytest

from src.monitoring_freshness import SensorFreshness


def item(observed_at="1970-01-01T00:01:40Z", *, quality="valid"):
    return {
        "received_at": "1970-01-01T00:03:20Z",
        "record": {
            "message_type": "sensor_observation",
            "observed_at": observed_at,
            "observed_at_quality": quality,
            "measurements": {"temperature": {"value": 0, "quality": "valid"}},
        },
        "validation_error": None,
    }


def test_delayed_arrival_keeps_measurement_age_and_original_value():
    original = item()
    before = copy.deepcopy(original)
    view = SensorFreshness(60, clock_sync="confirmed").describe(original, now=200)
    assert view["status"] == "stale"
    assert view["age_seconds"] == 100
    assert view["on_arrival"]["status"] == "stale"
    assert view["on_arrival"]["age_seconds"] == 100
    assert original == before


@pytest.mark.parametrize(
    "now,status", [(100, "fresh"), (160, "fresh"), (160.01, "stale")]
)
def test_threshold_and_utc_offset(now, status):
    value = item("1970-01-01T09:01:40+09:00")
    view = SensorFreshness(60, clock_sync="confirmed").describe(value, now=now)
    assert view["status"] == status
    assert view["age_seconds"] == pytest.approx(now - 100)


@pytest.mark.parametrize(
    "observed,quality,reason",
    [
        (None, "missing", "observed_at_missing"),
        (None, "invalid", "observed_at_invalid"),
        ("garbage", "valid", "observed_at_invalid"),
        ("1970-01-01T00:01:40", "valid", "observed_at_invalid"),
        ("1970-01-01T00:01:40Z", "missing", "observed_at_missing"),
        ("1970-01-01T00:05:00Z", "valid", "future_observed_at"),
    ],
)
def test_bad_or_untrusted_time_is_not_replaced_by_arrival(observed, quality, reason):
    view = SensorFreshness(60, clock_sync="confirmed").describe(
        item(observed, quality=quality), now=200
    )
    assert view["status"] == "unknown"
    assert view["age_seconds"] is None
    assert view["reason"] == reason


def test_no_policy_or_unconfirmed_clock_never_claims_freshness():
    view = SensorFreshness(clock_sync="confirmed").describe(item(), now=200)
    assert view["status"] == "unknown" and view["reason"] == "threshold_unset"
    view = SensorFreshness(60).describe(item(), now=200)
    assert view["status"] == "unknown" and view["reason"] == "clock_sync_unknown"
    assert view["age_seconds"] is None


@pytest.mark.parametrize("threshold", [0, -1, True, "60", float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        SensorFreshness(threshold)


def test_zero_epoch_and_invalid_arrival_are_independent_of_current_age():
    value = item("1970-01-01T00:00:00Z")
    value["received_at"] = "invalid"
    view = SensorFreshness(60, clock_sync="confirmed").describe(value, now=100)
    assert view["status"] == "stale" and view["age_seconds"] == 100
    assert view["on_arrival"]["status"] == "unknown"
    assert view["on_arrival"]["reason"] == "reference_time_invalid"


@pytest.mark.parametrize("now", [True, float("nan"), float("inf"), "200"])
def test_invalid_query_time_is_rejected(now):
    with pytest.raises(ValueError):
        SensorFreshness(60).describe(item(), now=now)


def test_bad_clock_setting_and_non_sensor_or_corrupt_records():
    with pytest.raises(ValueError):
        SensorFreshness(60, clock_sync="assumed")
    policy = SensorFreshness(60, clock_sync="confirmed")
    value = item()
    value["record"] = {"event_id": "video-1"}
    assert policy.describe(value, now=200)["status"] == "not_applicable"
    value = item()
    value["validation_error"] = "payload_hash_mismatch"
    view = policy.describe(value, now=200)
    assert view["status"] == "unknown" and view["reason"] == "unverifiable_record"


def test_sensor_observation_inside_event_uses_observation_time():
    value = item()
    value["record"] = {
        "event_id": "sensor-event",
        "occurred_at": value["received_at"],
        "observation": value["record"],
    }
    view = SensorFreshness(60, clock_sync="confirmed").describe(value, now=200)
    assert view["status"] == "stale" and view["age_seconds"] == 100
