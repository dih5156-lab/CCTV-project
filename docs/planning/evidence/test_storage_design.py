"""P1.2 transaction/migration prototype; every database is under pytest tmp_path."""

import base64
import hashlib
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

SCHEMA = Path(__file__).with_name("storage_design_schema.sql")


def connect(path):
    connection = sqlite3.connect(path, timeout=0.1)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=EXTRA")
    return connection


@pytest.fixture
def store(tmp_path):
    connection = connect(tmp_path / "common.db")
    connection.executescript(SCHEMA.read_text())
    yield connection
    connection.close()


def accept(connection, raw=b'{"temperature":0}', mode="live", legacy_key=None):
    ingest_id = "ing_" + uuid4().hex
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        if legacy_key is not None:
            existing = connection.execute(
                "SELECT ingest_id FROM legacy_import_map WHERE "
                "source_store=? AND table_name=? AND legacy_key=?",
                legacy_key,
            ).fetchone()
            if existing:
                return existing[0]
        connection.execute(
            "INSERT INTO ingest_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ingest_id,
                "tlv_mqtt",
                "sensor-1",
                "2026-09-18T00:00:00Z",
                "bytes_base64",
                base64.b64encode(raw).decode(),
                hashlib.sha256(raw).hexdigest(),
                mode,
            ),
        )
        if legacy_key is not None:
            connection.execute(
                "INSERT INTO legacy_import_map VALUES (?, ?, ?, ?)",
                (*legacy_key, ingest_id),
            )
    return ingest_id


def normalize(connection, ingest_id, record_id="source-1", fail_before_job=False):
    """Only a design demonstration; source adapters/worker/ACK are not implemented."""
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        namespace, fingerprint, mode = connection.execute(
            "SELECT source_namespace, raw_sha256, mode FROM ingest_receipts WHERE ingest_id=?",
            (ingest_id,),
        ).fetchone()
        existing = connection.execute(
            "SELECT source_sha256 FROM canonical_records "
            "WHERE source_namespace=? AND source_record_id=?",
            (namespace, record_id),
        ).fetchone()
        status = (
            "valid"
            if existing is None
            else "duplicate" if existing[0] == fingerprint else "conflict"
        )
        payload = json.dumps({"event_id": record_id, "temperature": 0})
        if status == "valid":
            connection.execute(
                "INSERT INTO canonical_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    namespace,
                    record_id,
                    "observation",
                    "sensor-1",
                    None,
                    None,
                    fingerprint,
                    payload,
                    ingest_id,
                ),
            )
        revision = connection.execute(
            "SELECT COALESCE(MAX(revision), 0) + 1 FROM ingest_revisions WHERE ingest_id=?",
            (ingest_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO ingest_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ingest_id,
                revision,
                "1.0",
                "prototype-1",
                status,
                payload,
                '{"code":"id_collision"}' if status == "conflict" else None,
                namespace,
                record_id,
            ),
        )
        if fail_before_job:
            raise RuntimeError("injected failure before outbox INSERT")
        if status == "valid" and mode == "live":
            connection.execute(
                "INSERT INTO delivery_jobs "
                "(job_id, source_namespace, source_record_id, destination, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (uuid4().hex, namespace, record_id, "monitoring", payload),
            )
    return status


def count(connection, table):
    return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_raw_receipt_survives_normalization_rollback_and_reopen(store, tmp_path):
    ingest_id = accept(store, b"\xffbroken-tlv")
    with pytest.raises(RuntimeError, match="injected failure"):
        normalize(store, ingest_id, fail_before_job=True)
    assert [
        count(store, table)
        for table in (
            "ingest_receipts",
            "canonical_records",
            "ingest_revisions",
            "delivery_jobs",
        )
    ] == [1, 0, 0, 0]
    reopened = connect(tmp_path / "common.db")
    try:
        raw = reopened.execute("SELECT raw_payload FROM ingest_receipts").fetchone()[0]
        assert base64.b64decode(raw) == b"\xffbroken-tlv"
        assert normalize(reopened, ingest_id) == "valid"
        assert count(reopened, "delivery_jobs") == 1
    finally:
        reopened.close()


def test_duplicate_receipts_share_record_and_delivery(store):
    assert normalize(store, accept(store)) == "valid"
    assert normalize(store, accept(store)) == "duplicate"
    assert [
        count(store, table)
        for table in (
            "ingest_receipts",
            "canonical_records",
            "ingest_revisions",
            "delivery_jobs",
        )
    ] == [2, 1, 2, 1]


def test_same_id_different_body_is_preserved_as_conflict(store):
    original = accept(store)
    normalize(store, original)
    conflicting = accept(store, b'{"temperature":90}')
    assert normalize(store, conflicting) == "conflict"
    assert count(store, "ingest_receipts") == 2
    assert count(store, "delivery_jobs") == 1
    assert store.execute(
        "SELECT first_ingest_id FROM canonical_records"
    ).fetchone() == (original,)
    assert (
        store.execute(
            "SELECT error_json FROM ingest_revisions WHERE ingest_id=?", (conflicting,)
        ).fetchone()[0]
        == '{"code":"id_collision"}'
    )


@pytest.mark.parametrize("mode", ["import", "replay"])
def test_historical_records_do_not_create_delivery_jobs(store, mode):
    ingest_id = accept(store, mode=mode)
    normalize(store, ingest_id)
    normalize(store, ingest_id)
    assert count(store, "canonical_records") == 1
    assert count(store, "ingest_revisions") == 2
    assert count(store, "delivery_jobs") == 0


def test_foreign_keys_reject_orphaned_delivery(store):
    with pytest.raises(sqlite3.IntegrityError), store:
        store.execute(
            "INSERT INTO delivery_jobs "
            "(job_id, source_namespace, source_record_id, destination, payload_json) "
            "VALUES ('job', 'missing', 'missing', 'monitoring', '{}')"
        )
    assert count(store, "delivery_jobs") == 0


def test_legacy_mapping_is_idempotent_and_backup_is_readable(store, tmp_path):
    with store:
        store.execute("CREATE TABLE legacy_sensor (id INTEGER PRIMARY KEY, value REAL)")
        store.execute("INSERT INTO legacy_sensor VALUES (7, 0)")
    for _ in range(2):
        ingest_id = accept(
            store, mode="import", legacy_key=("fixture", "legacy_sensor", "7")
        )
        normalize(store, ingest_id, record_id="legacy-7")
    assert count(store, "ingest_receipts") == count(store, "legacy_import_map") == 1
    assert count(store, "delivery_jobs") == 0
    backup = connect(tmp_path / "backup.db")
    try:
        store.backup(backup)
        assert backup.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert backup.execute("PRAGMA foreign_key_check").fetchall() == []
        assert backup.execute("SELECT * FROM legacy_sensor").fetchall() == [(7, 0.0)]
        assert count(backup, "canonical_records") == 1
    finally:
        backup.close()
