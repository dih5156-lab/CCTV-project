"""Record-only monitoring identity and durable-storage acknowledgement contract."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime

from .record_delivery import record_topic


class MonitoringContractError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def require_string(value, code="invalid"):
    if not isinstance(value, str) or not value.strip():
        raise MonitoringContractError(code)
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise MonitoringContractError(code) from error
    return value


@dataclass(frozen=True)
class MonitoringBinding:
    origin_id: str
    consumer_id: str
    destination: str

    def __post_init__(self):
        for value in (self.origin_id, self.consumer_id, self.destination):
            require_string(value, "binding_mismatch")
        record_topic(self.destination)

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MonitoringRecord:
    delivery_id: str
    source_namespace: str
    record_type: str
    record_id: str
    payload_sha256: str
    record_json: str


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MonitoringContractError("invalid")
        result[key] = value
    return result


def _reject_constant(value):
    raise MonitoringContractError("invalid")


def parse_record(wire: bytes) -> MonitoringRecord:
    if not isinstance(wire, bytes):
        raise MonitoringContractError("invalid")
    try:
        envelope = json.loads(
            wire.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(envelope, dict):
            raise MonitoringContractError("invalid")
        if envelope.get("schema_version") != "1.0":
            raise MonitoringContractError("unsupported_version")
        delivery_id = require_string(envelope.get("delivery_id"))
        namespace = require_string(envelope.get("source_namespace"))
        record = envelope.get("record")
        if not isinstance(record, dict):
            raise MonitoringContractError("invalid")
        kind = (
            "sensor_observation"
            if record.get("message_type") == "sensor_observation"
            else "event"
        )
        id_field = "observation_id" if kind == "sensor_observation" else "event_id"
        other = "event_id" if id_field == "observation_id" else "observation_id"
        record_id = require_string(envelope.get(id_field))
        if (
            other in envelope
            or record.get(id_field) != record_id
            or record.get("source_namespace", namespace) != namespace
        ):
            raise MonitoringContractError("invalid")
        # Also rejects overflowed JSON numbers and unpaired escaped surrogates.
        json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode("utf-8")
        record_json = json.dumps(
            record, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
    except MonitoringContractError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise MonitoringContractError("invalid") from error
    return MonitoringRecord(
        delivery_id,
        namespace,
        kind,
        record_id,
        hashlib.sha256(wire).hexdigest(),
        record_json,
    )


def validate_ack(
    ack: dict, binding: MonitoringBinding, record: MonitoringRecord
) -> dict:
    """Validate content; the caller must authenticate its channel and persist ACK."""
    expected = {
        "ack_schema_version": "1.0",
        "ack_kind": "consumer_committed",
        **binding.as_dict(),
        "delivery_id": record.delivery_id,
        "source_namespace": record.source_namespace,
        "record_type": record.record_type,
        "record_id": record.record_id,
        "payload_sha256": record.payload_sha256,
    }
    if not isinstance(ack, dict) or any(ack.get(k) != v for k, v in expected.items()):
        raise MonitoringContractError("ack_mismatch")
    if ack.get("status") not in ("stored", "duplicate"):
        raise MonitoringContractError("ack_mismatch")
    for field in ("receiver_record_id", "stored_at", "ack_emitted_at"):
        require_string(ack.get(field), "ack_mismatch")
    try:
        for field in ("stored_at", "ack_emitted_at"):
            if (
                datetime.fromisoformat(ack[field].replace("Z", "+00:00")).utcoffset()
                is None
            ):
                raise ValueError("timezone required")
    except ValueError as error:
        raise MonitoringContractError("ack_mismatch") from error
    return {
        **expected,
        **{
            key: ack[key]
            for key in ("status", "receiver_record_id", "stored_at", "ack_emitted_at")
        },
    }
