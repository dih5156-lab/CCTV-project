"""Wire/ACK boundaries must preserve IDs and reject false storage success."""

import importlib
import json

import pytest


def module():
    return importlib.import_module("src.monitoring_contract")


def wire(record_id="evt_1", **changes):
    payload = {
        "schema_version": "1.0",
        "delivery_id": "job_1",
        "source_namespace": "video_detection",
        "event_id": record_id,
        "record": {"event_id": record_id, "label": "낙상", "value": 0},
    }
    payload.update(changes)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()


def test_wire_hash_uses_original_bytes_and_preserves_zero():
    contract = module()
    parsed = contract.parse_record(wire())
    assert parsed.record_id == "evt_1"
    assert json.loads(parsed.record_json)["value"] == 0
    assert parsed.payload_sha256 != contract.parse_record(wire() + b" ").payload_sha256


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"[]",
        b'{"x":1,"x":2}',
        wire(record={"event_id": "evt_1", "value": float("nan")}),
        wire().replace(b'"value": 0', b'"value": 1e9999'),
        wire(schema_version="2.0"),
        wire(observation_id="obs_wrong"),
        wire(record={"event_id": "wrong"}),
        wire(record={"event_id": "evt_1", "source_namespace": "wrong"}),
        wire().replace("낙상".encode(), b"\\ud800"),
    ],
)
def test_invalid_wire_is_rejected(payload):
    contract = module()
    with pytest.raises(contract.MonitoringContractError):
        contract.parse_record(payload)


def test_sensor_source_record_id_is_not_the_observation_id():
    contract = module()
    payload = {
        "schema_version": "1.0",
        "delivery_id": "job_sensor",
        "source_namespace": "tlv_mqtt",
        "observation_id": "obs_1",
        "record": {
            "message_type": "sensor_observation",
            "observation_id": "obs_1",
            "source_record_id": "ing_1",
            "observed_at": None,
        },
    }
    parsed = contract.parse_record(json.dumps(payload).encode())
    assert parsed.record_id == "obs_1"
    assert parsed.record_type == "sensor_observation"


def test_matching_ack_validation_and_wrong_binding():
    contract = module()
    binding = contract.MonitoringBinding(
        "origin_1", "consumer_1", "record-mqtt:cctv/records/v1/lab"
    )
    record = contract.parse_record(wire())
    ack = {
        "ack_schema_version": "1.0",
        "ack_kind": "consumer_committed",
        **binding.as_dict(),
        "delivery_id": record.delivery_id,
        "source_namespace": record.source_namespace,
        "record_type": record.record_type,
        "record_id": record.record_id,
        "payload_sha256": record.payload_sha256,
        "status": "stored",
        "receiver_record_id": "rec_1",
        "stored_at": "2026-09-21T10:00:00+00:00",
        "ack_emitted_at": "2026-09-21T10:00:01+00:00",
    }
    assert contract.validate_ack(ack, binding, record) == ack
    for field in ack:
        invalid = dict(ack)
        invalid.pop(field)
        with pytest.raises(contract.MonitoringContractError):
            contract.validate_ack(invalid, binding, record)
    for field, value in (
        ("consumer_id", "other"),
        ("ack_kind", "broker_puback"),
        ("payload_sha256", "0" * 64),
        ("status", "accepted"),
        ("stored_at", "2026-09-21T10:00:00"),
        ("ack_emitted_at", 123),
    ):
        with pytest.raises(contract.MonitoringContractError):
            contract.validate_ack({**ack, field: value}, binding, record)
