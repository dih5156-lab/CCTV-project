"""P1.4 executable specification, not a deployed receiver or source ACK worker.

All databases use pytest tmp_path. No network, GPU or physical outputs.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from src.event_ingest_store import EventIngestStore
from src.record_delivery import RecordDeliveryStore

DESTINATION = "record-mqtt:cctv/records/v1/contract-lab"
BINDING = {
    "origin_id": "origin-lab-1",
    "consumer_id": "consumer-lab-v1",
    "destination": DESTINATION,
}


def encode(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("invalid")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("invalid")


def identity(wire):
    envelope = json.loads(
        wire.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(envelope, dict) or envelope.get("schema_version") != "1.0":
        raise ValueError("unsupported_version")
    for field in ("delivery_id", "source_namespace"):
        if not isinstance(envelope.get(field), str) or not envelope[field].strip():
            raise ValueError("invalid")
    record = envelope.get("record")
    if not isinstance(record, dict):
        raise ValueError("invalid")
    kind = (
        "sensor_observation"
        if record.get("message_type") == "sensor_observation"
        else "event"
    )
    id_field = "observation_id" if kind == "sensor_observation" else "event_id"
    other = "event_id" if id_field == "observation_id" else "observation_id"
    record_id = envelope.get(id_field)
    if (
        not isinstance(record_id, str)
        or not record_id.strip()
        or other in envelope
        or record.get(id_field) != record_id
        or record.get("source_namespace", envelope["source_namespace"])
        != envelope["source_namespace"]
    ):
        raise ValueError("invalid")
    return envelope, kind, record_id


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class ReceiverModel:
    """Reference transaction and idempotency model; no transport/auth service."""

    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=EXTRA")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                business_key TEXT PRIMARY KEY, receiver_id TEXT UNIQUE NOT NULL,
                body BLOB NOT NULL, stored_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS receipts (
                delivery_key TEXT PRIMARY KEY,
                business_key TEXT NOT NULL REFERENCES records(business_key),
                digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS rejected (
                scope TEXT NOT NULL, wire BLOB NOT NULL, code TEXT NOT NULL);
            """
        )

    def close(self):
        self.connection.close()

    @staticmethod
    def keys(binding, envelope, kind, record_id):
        business = encode(
            [
                binding["origin_id"],
                binding["consumer_id"],
                envelope["source_namespace"],
                kind,
                record_id,
            ]
        ).decode()
        delivery = encode([binding, envelope["delivery_id"]]).decode()
        return business, delivery

    def lookup(self, wire, binding=BINDING, status="duplicate"):
        envelope, kind, record_id = identity(wire)
        business, delivery = self.keys(binding, envelope, kind, record_id)
        row = self.connection.execute(
            "SELECT r.receiver_id, r.stored_at, d.digest FROM receipts d "
            "JOIN records r USING(business_key) "
            "WHERE d.delivery_key=? AND d.business_key=?",
            (delivery, business),
        ).fetchone()
        if row is None:
            return None
        return {
            "ack_schema_version": "1.0",
            "ack_kind": "consumer_committed",
            **binding,
            "delivery_id": envelope["delivery_id"],
            "source_namespace": envelope["source_namespace"],
            "record_type": kind,
            "record_id": record_id,
            "payload_sha256": row[2],
            "status": status,
            "receiver_record_id": row[0],
            "stored_at": row[1],
            "ack_emitted_at": utc_now(),
        }

    def accept(self, wire, binding=BINDING, lose_ack=False):
        try:
            envelope, kind, record_id = identity(wire)
            business, delivery = self.keys(binding, envelope, kind, record_id)
            digest, body = hashlib.sha256(wire).hexdigest(), encode(envelope["record"])
            status = "stored"
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                existing = self.connection.execute(
                    "SELECT digest FROM receipts WHERE delivery_key=?", (delivery,)
                ).fetchone()
                record = self.connection.execute(
                    "SELECT body FROM records WHERE business_key=?", (business,)
                ).fetchone()
                if existing and existing[0] != digest:
                    raise ValueError("delivery_conflict")
                if record and record[0] != body:
                    raise ValueError("record_conflict")
                if record:
                    status = "duplicate"
                else:
                    self.connection.execute(
                        "INSERT INTO records VALUES (?, ?, ?, ?)",
                        (
                            business,
                            "rec_" + hashlib.sha256(business.encode()).hexdigest(),
                            body,
                            utc_now(),
                        ),
                    )
                if not existing:
                    self.connection.execute(
                        "INSERT INTO receipts VALUES (?, ?, ?)",
                        (delivery, business, digest),
                    )
        except (ValueError, UnicodeError) as error:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO rejected VALUES (?, ?, ?)",
                    (encode(binding).decode(), wire, str(error)),
                )
            raise
        if lose_ack:
            raise ConnectionError("commit succeeded; ACK lost")
        return self.lookup(wire, binding, status)


def ack_matches(ack, wire, binding=BINDING):
    """Checks content only; authenticated channel/DB state are P2.3 obligations."""
    envelope, kind, record_id = identity(wire)
    expected = {
        "ack_schema_version": "1.0",
        "ack_kind": "consumer_committed",
        **binding,
        "delivery_id": envelope["delivery_id"],
        "source_namespace": envelope["source_namespace"],
        "record_type": kind,
        "record_id": record_id,
        "payload_sha256": hashlib.sha256(wire).hexdigest(),
    }
    if not isinstance(ack, dict) or any(ack.get(k) != v for k, v in expected.items()):
        return False
    if ack.get("status") not in ("stored", "duplicate"):
        return False
    if (
        not isinstance(ack.get("receiver_record_id"), str)
        or not ack["receiver_record_id"].strip()
    ):
        return False
    try:
        for field in ("stored_at", "ack_emitted_at"):
            if (
                datetime.fromisoformat(ack[field].replace("Z", "+00:00")).utcoffset()
                is None
            ):
                return False
    except (KeyError, ValueError, TypeError, AttributeError):
        return False
    return True


@pytest.fixture
def receiver(tmp_path):
    model = ReceiverModel(tmp_path / "receiver.db")
    yield model
    model.close()


@pytest.fixture(params=["event", "sensor_observation"])
def wire(request, tmp_path):
    """Use actual source envelope generation, not a separately invented wire."""
    kind = request.param
    namespace = "tlv_mqtt" if kind == "sensor_observation" else "video_detection"
    record = (
        {
            "message_type": kind,
            "observation_id": "obs_example",
            "source_record_id": "ing_example",
            "observed_at": None,
            "measurements": {"temperature": {"value": 0, "unit": "degC"}},
        }
        if kind == "sensor_observation"
        else {"event_id": "evt_example", "severity": "critical", "label": "낙상"}
    )
    store = EventIngestStore(tmp_path / "source.db")
    ingest_id = store.accept_event_json(namespace, record, destination=None)
    if kind == "sensor_observation":
        store.record_observation(
            ingest_id, record, converter_version="contract-example"
        )
    else:
        store.record_event(ingest_id, record, converter_version="contract-example")
    delivery = RecordDeliveryStore(store.path)
    delivery.enqueue_pending(namespace, DESTINATION, "local-contract-only")
    job = delivery.claim(DESTINATION, "local-contract-only", lease_seconds=30, now=100)
    return job["wire_payload"].encode("utf-8")


def counts(receiver):
    return tuple(
        receiver.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("records", "receipts", "rejected")
    )


def test_current_wire_acceptance_and_exact_bytes_digest(receiver, wire):
    ack = receiver.accept(wire)
    assert ack_matches(ack, wire)
    assert ack["status"] == "stored"
    assert counts(receiver) == (1, 1, 0)
    assert not ack_matches(ack, wire + b" ")


def test_ack_loss_then_reopen_lookup_and_retry(receiver, wire, tmp_path):
    with pytest.raises(ConnectionError, match="ACK lost"):
        receiver.accept(wire, lose_ack=True)
    receiver.close()
    receiver.connection = sqlite3.connect(tmp_path / "receiver.db")
    recovered = receiver.lookup(wire)
    assert ack_matches(recovered, wire)
    retried = receiver.accept(wire)
    assert retried["status"] == "duplicate"
    for field in ("receiver_record_id", "stored_at", "payload_sha256"):
        assert retried[field] == recovered[field]
    assert counts(receiver) == (1, 1, 0)


def test_ledger_failure_rolls_back_record_and_has_no_ack(receiver, wire):
    receiver.connection.execute(
        "CREATE TRIGGER fail_receipt BEFORE INSERT ON receipts "
        "BEGIN SELECT RAISE(ABORT, 'injected storage failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="storage failure"):
        receiver.accept(wire)
    assert counts(receiver) == (0, 0, 0)
    assert receiver.lookup(wire) is None
    receiver.connection.execute("DROP TRIGGER fail_receipt")
    assert ack_matches(receiver.accept(wire), wire)


@pytest.mark.parametrize("new_delivery", [False, True])
def test_conflicts_preserve_original_and_rejected_bytes(receiver, wire, new_delivery):
    original = receiver.accept(wire)
    changed = json.loads(wire)
    changed["record"]["unexpected_change"] = True
    if new_delivery:
        changed["delivery_id"] = "job_different"
    changed_wire = encode(changed)
    code = "record_conflict" if new_delivery else "delivery_conflict"
    with pytest.raises(ValueError, match=code):
        receiver.accept(changed_wire)
    assert counts(receiver) == (1, 1, 1)
    assert receiver.lookup(wire)["payload_sha256"] == original["payload_sha256"]
    assert (
        receiver.connection.execute("SELECT wire FROM rejected").fetchone()[0]
        == changed_wire
    )


def test_new_delivery_same_business_record_and_separate_origin(receiver, wire):
    first = receiver.accept(wire)
    changed = json.loads(wire)
    changed["delivery_id"] = "job_second"
    second = receiver.accept(encode(changed))
    assert second["status"] == "duplicate"
    assert first["receiver_record_id"] == second["receiver_record_id"]
    assert counts(receiver) == (1, 2, 0)
    other = {**BINDING, "origin_id": "origin-other"}
    other_ack = receiver.accept(wire, binding=other)
    assert counts(receiver) == (2, 3, 0)
    assert not ack_matches(other_ack, wire)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ack_schema_version", "2.0"),
        ("ack_kind", "broker_puback"),
        ("origin_id", "other"),
        ("consumer_id", "consumer-reset-v2"),
        ("destination", "mqtt:cctv/ai/events"),
        ("delivery_id", "job_other"),
        ("source_namespace", "other"),
        ("record_type", "command"),
        ("record_id", "wrong"),
        ("payload_sha256", "0" * 64),
        ("status", "accepted"),
        ("receiver_record_id", ""),
        ("stored_at", "2026-09-21T00:00:00"),
        ("ack_emitted_at", None),
    ],
)
def test_unrelated_or_incomplete_ack_cannot_complete_job(receiver, wire, field, value):
    ack = receiver.accept(wire)
    assert not ack_matches({**ack, field: value}, wire)
    assert not ack_matches(
        {key: item for key, item in ack.items() if key != field}, wire
    )


@pytest.mark.parametrize(
    "raw",
    [b"\xff", b'{"x":1,"x":2}', b'{"value":NaN}', b'{"value":Infinity}', b"[]"],
)
def test_invalid_json_is_retained_without_success(receiver, raw):
    with pytest.raises((ValueError, UnicodeError)):
        receiver.accept(raw)
    assert counts(receiver) == (0, 0, 1)


@pytest.mark.parametrize("mutation", ["both_ids", "inner_id", "namespace", "version"])
def test_envelope_mismatch_is_rejected(receiver, wire, mutation):
    envelope, kind, _ = identity(wire)
    id_field = "observation_id" if kind == "sensor_observation" else "event_id"
    if mutation == "both_ids":
        envelope["event_id" if id_field == "observation_id" else "observation_id"] = (
            "bad"
        )
    elif mutation == "inner_id":
        envelope["record"][id_field] = "bad"
    elif mutation == "namespace":
        envelope["record"]["source_namespace"] = "bad"
    else:
        envelope["schema_version"] = "9.0"
    with pytest.raises(ValueError):
        receiver.accept(encode(envelope))
    assert counts(receiver) == (0, 0, 1)


def test_snapshot_excludes_concurrent_late_arrival_but_next_snapshot_includes_it(
    tmp_path,
):
    with sqlite3.connect(tmp_path / "manifest.db") as connection:
        connection.executescript(
            "CREATE TABLE candidates (id TEXT PRIMARY KEY, received_at TEXT, ingest_id TEXT);"
            "CREATE TABLE manifests (snapshot_id TEXT, ordinal INTEGER, id TEXT, "
            "PRIMARY KEY(snapshot_id, ordinal));"
        )
        connection.executemany(
            "INSERT INTO candidates VALUES (?, ?, ?)",
            [
                ("a", "2026-09-21T01:00:00Z", "ing-a"),
                ("b", "2026-09-21T01:00:00Z", "ing-b"),
            ],
        )
        connection.commit()

        def snapshot(snapshot_id):
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO manifests SELECT ?, "
                    "ROW_NUMBER() OVER(ORDER BY received_at, ingest_id, id), id FROM candidates",
                    (snapshot_id,),
                )

        snapshot("first")
        first_page = connection.execute(
            "SELECT ordinal, id FROM manifests WHERE snapshot_id='first' AND ordinal>0 "
            "ORDER BY ordinal LIMIT 1"
        ).fetchone()
        connection.execute(
            "INSERT INTO candidates VALUES ('late', '2026-09-20T00:00:00Z', 'ing-late')"
        )
        connection.commit()
        second_page = connection.execute(
            "SELECT ordinal, id FROM manifests WHERE snapshot_id='first' AND ordinal>? "
            "ORDER BY ordinal LIMIT 1",
            (first_page[0],),
        ).fetchone()
        assert (first_page[1], second_page[1]) == ("a", "b")
        snapshot("second")
        assert connection.execute(
            "SELECT id FROM manifests WHERE snapshot_id='second' ORDER BY ordinal"
        ).fetchall() == [("late",), ("a",), ("b",)]


def test_equal_counts_do_not_hide_missing_ids_or_content_conflicts():
    expected = {"job_a": "digest_a", "job_b": "digest_b"}
    committed = {"job_a": "different_digest", "job_c": "digest_c"}
    assert len(expected) == len(committed)
    assert expected.keys() - committed.keys() == {"job_b"}
    assert committed.keys() - expected.keys() == {"job_c"}
    assert {
        key
        for key in expected.keys() & committed.keys()
        if expected[key] != committed[key]
    } == {"job_a"}
