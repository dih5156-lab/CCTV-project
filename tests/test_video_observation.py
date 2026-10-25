import json
from copy import deepcopy

import pytest

from src.video_observation import build_video_observation


def _build(payload, **overrides):
    context = {
        "namespace": "video_detection",
        "source_record_id": "record-1",
        "device_id": "cam01",
        "received_at": "2026-09-18T12:00:00+09:00",
    }
    context.update(overrides)
    return build_video_observation(payload, **context)


def test_attributes_keep_false_unknown_missing_and_invalid_separate():
    payload = {
        "object_id": 0,
        "timestamp": 1700000000,
        "metadata": {
            "has_helmet": False,
            "has_backpack": True,
            "upper_color": "unknown",
            "lower_color": "black",
            "gender": None,
            "has_handbag": 0,
            "backend": "deepstream_tensor",
            "attribute_backend": "hsv",
            "frame_num": 0,
            "model_task": "pose",
            "gie_id": 0,
            "model_version": "fixture-v1",
        },
    }
    original = deepcopy(payload)
    observation = _build(payload)
    measurements = observation["measurements"]
    assert measurements["has_helmet"] == {
        "value": False,
        "unit": None,
        "quality": "valid",
        "source_field": "metadata.has_helmet",
    }
    assert measurements["has_backpack"]["value"] is True
    assert measurements["upper_color"]["value"] is None
    assert measurements["upper_color"]["quality"] == "unknown"
    assert measurements["lower_color"]["value"] == "black"
    assert measurements["gender"]["quality"] == "missing"
    assert measurements["has_suitcase"]["quality"] == "missing"
    assert measurements["has_handbag"]["quality"] == "invalid"
    assert observation["track_id"]["value"] == "0"
    assert observation["frame_num"] == 0
    assert observation["model"] == {
        "backend": "deepstream_tensor",
        "attribute_backend": "hsv",
        "task": "pose",
        "gie_id": 0,
        "version": "fixture-v1",
    }
    assert payload == original


@pytest.mark.parametrize("value", [0, 1, "false", "true", "", [], {}, float("nan")])
def test_invalid_boolean_is_never_coerced_to_false(value):
    observation = _build({"metadata": {"has_helmet": value}})
    assert observation["measurements"]["has_helmet"]["quality"] == "invalid"
    assert observation["measurements"]["has_helmet"]["value"] is None
    json.dumps(observation, allow_nan=False)


def test_explicit_unknown_boolean_and_null_attribute_override_metadata():
    observation = _build(
        {
            "attributes": {"has_helmet": "unknown", "upper_color": None},
            "metadata": {"has_helmet": False, "upper_color": "red"},
        }
    )
    assert observation["measurements"]["has_helmet"]["quality"] == "unknown"
    assert observation["measurements"]["upper_color"]["quality"] == "missing"
    assert (
        observation["measurements"]["upper_color"]["source_field"]
        == "attributes.upper_color"
    )


@pytest.mark.parametrize("value", [False, 17, "", [], {}])
def test_invalid_category_is_not_a_label(value):
    reading = _build({"metadata": {"upper_color": value}})["measurements"][
        "upper_color"
    ]
    assert reading["value"] is None
    assert reading["quality"] == "invalid"


@pytest.mark.parametrize(
    "value,quality",
    [
        (None, "missing"),
        (-1, "unknown"),
        (2**64 - 1, "unknown"),
        (True, "invalid"),
        ("07", "invalid"),
        (1.5, "invalid"),
        (2**64, "invalid"),
    ],
)
def test_track_missing_untracked_and_malformed_ids(value, quality):
    track = _build({"object_id": value})["track_id"]
    assert track["value"] is None
    assert track["quality"] == quality


def test_processing_time_is_seconds_and_is_not_capture_time():
    observation = _build({"timestamp": 0, "occurred_at": "2026-09-18T00:00:00Z"})
    assert observation["processed_at"] == "1970-01-01T09:00:00+09:00"
    assert observation["processed_at_source"] == "timestamp"
    assert observation["processed_at_quality"] == "valid"
    assert observation["observed_at"] is None
    assert observation["observed_at_quality"] == "missing"


@pytest.mark.parametrize(
    "value", [True, "1700000000", float("nan"), float("inf"), 10**100]
)
def test_invalid_processing_time_is_marked_without_using_receipt_time(value):
    observation = _build({"timestamp": value})
    assert observation["processed_at"] is None
    assert observation["processed_at_quality"] == "invalid"


def test_missing_time_model_and_attributes_are_not_invented():
    observation = _build({})
    assert observation["processed_at"] is None
    assert observation["processed_at_quality"] == "missing"
    assert observation["processed_at_source"] is None
    assert all(value is None for value in observation["model"].values())
    assert all(
        reading["quality"] == "missing"
        for reading in observation["measurements"].values()
    )


def test_invalid_attribute_container_is_distinct_from_missing():
    observation = _build({"metadata": []})
    assert all(
        reading["quality"] == "invalid"
        for reading in observation["measurements"].values()
    )


def test_id_reuses_source_identity_not_receipt_time_or_dictionary_order():
    first = _build({"object_id": 7, "timestamp": 1700000000})
    repeated = _build(
        {"timestamp": 1700000000, "object_id": 7},
        received_at="2026-09-18T12:01:00+09:00",
    )
    assert first["observation_id"] == repeated["observation_id"]
    for field in ("namespace", "source_record_id", "device_id"):
        assert (
            _build({}, **{field: "other"})["observation_id"] != first["observation_id"]
        )


@pytest.mark.parametrize("field", ["namespace", "source_record_id", "device_id"])
def test_observation_requires_nonempty_source_identity(field):
    with pytest.raises(ValueError):
        _build({}, **{field: ""})


@pytest.mark.parametrize("value", ["bad", "2026-09-18T12:00:00", None])
def test_receipt_requires_timezone(value):
    with pytest.raises(ValueError):
        _build({}, received_at=value)
