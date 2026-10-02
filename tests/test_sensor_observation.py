import json
from copy import deepcopy

import pytest


def _build(payload, **overrides):
    from src.sensor_observation import build_sensor_observation

    context = {
        "namespace": "external_mqtt",
        "source_record_id": "record-1",
        "device_id": "sensor-1",
        "received_at": "2026-09-18T11:00:00+09:00",
    }
    context.update(overrides)
    return build_sensor_observation(payload, **context)


def test_observation_distinguishes_zero_missing_and_invalid_without_mutation():
    payload = {
        "table": "t34957",
        "data": {"temperature_c": 0, "angle_x_deg": None, "angle_y_deg": "bad"},
    }
    original = deepcopy(payload)
    observation = _build(payload)

    assert observation["measurements"]["temperature"] == {
        "value": 0.0,
        "unit": "degC",
        "quality": "valid",
        "source_field": "temperature_c",
    }
    assert observation["measurements"]["angle_x"]["value"] is None
    assert observation["measurements"]["angle_x"]["quality"] == "missing"
    assert observation["measurements"]["angle_y"]["value"] is None
    assert observation["measurements"]["angle_y"]["quality"] == "invalid"
    assert payload == original


@pytest.mark.parametrize(
    "value", [True, False, "25.0", float("nan"), float("inf"), [], {}]
)
def test_invalid_measurement_never_becomes_valid_zero(value):
    observation = _build({"table": "t34954", "data": {"temperature_c": value}})
    reading = observation["measurements"]["temperature"]
    assert reading["value"] is None
    assert reading["quality"] == "invalid"
    assert observation["measurements"]["humidity"]["quality"] == "missing"
    json.dumps(observation, allow_nan=False)


@pytest.mark.parametrize(
    "profile,fields",
    [
        ("t34954", {"temperature": "degC", "humidity": "%"}),
        ("t34955", {"angle_x": "deg", "angle_y": "deg"}),
        ("t34957", {"temperature": "degC", "angle_x": "deg", "angle_y": "deg"}),
    ],
)
def test_profile_has_expected_units_even_when_measurements_are_missing(profile, fields):
    observation = _build({"table": profile, "data": {}})
    assert {
        key: value["unit"] for key, value in observation["measurements"].items()
    } == fields
    assert all(
        value["quality"] == "missing" for value in observation["measurements"].values()
    )


def test_unknown_profile_is_not_assigned_guessed_units():
    assert _build({"table": "vendor-other", "data": {"temperature": 25}}) is None


def test_missing_observation_time_is_not_replaced_with_receipt_time():
    observation = _build({"table": "t34954", "data": {}})
    assert observation["observed_at"] is None
    assert observation["observed_at_quality"] == "missing"
    assert observation["received_at"] == "2026-09-18T11:00:00+09:00"


def test_sensor_epoch_milliseconds_and_source_are_explicit():
    observation = _build({"table": "t34955", "decoded": {"created_at": 0}})
    assert observation["observed_at"] == "1970-01-01T09:00:00+09:00"
    assert observation["observed_at_source"] == "decoded.created_at"
    assert observation["observed_at_quality"] == "valid"


@pytest.mark.parametrize("timestamp", ["2026-09-18T00:00:00Z", True, 1.5, 10**100])
def test_invalid_sensor_time_keeps_observation_and_marks_quality(timestamp):
    observation = _build({"table": "t34954", "data": {"created_at": timestamp}})
    assert observation["observed_at"] is None
    assert observation["observed_at_quality"] == "invalid"


def test_observation_id_is_stable_for_the_same_source_record():
    payload = {"table": "t34954", "data": {"humidity_pct": 40, "temperature_c": 20}}
    first = _build(payload)
    second = _build(
        {"data": {"temperature_c": 20, "humidity_pct": 40}, "table": "t34954"},
        received_at="2026-09-18T12:00:00+09:00",
    )
    assert first["observation_id"] == second["observation_id"]
    assert first["source_record_id"] == "record-1"
    for field in ("namespace", "source_record_id", "device_id"):
        assert (
            _build(payload, **{field: "other"})["observation_id"]
            != first["observation_id"]
        )
    assert (
        _build({"table": "t34957", "data": payload["data"]})["observation_id"]
        != first["observation_id"]
    )


@pytest.mark.parametrize("field", ["namespace", "source_record_id", "device_id"])
def test_source_identity_is_required(field):
    with pytest.raises(ValueError):
        _build({"table": "t34954", "data": {}}, **{field: ""})


def test_invalid_data_container_is_not_mislabeled_as_missing():
    observation = _build({"table": "t34955", "data": []})
    assert all(
        value["quality"] == "invalid" for value in observation["measurements"].values()
    )


@pytest.mark.parametrize("received_at", ["", "bad-time", "2026-09-18T11:00:00"])
def test_receipt_time_requires_an_explicit_timezone(received_at):
    with pytest.raises(ValueError):
        _build({"table": "t34954", "data": {}}, received_at=received_at)
