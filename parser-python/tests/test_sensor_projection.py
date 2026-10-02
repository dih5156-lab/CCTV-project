import base64
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg2
import psycopg2.pool
import pytest
from psycopg2 import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.connection import DB, _ensure_schema  # noqa: E402
from database.models import T34954, SensorData  # noqa: E402
from database.sensor_projection import (  # noqa: E402
    ProjectionConflict,
    encode_records,
    ensure_projection_schema,
    write_sensor_projection,
)
from database.sensor_projection_worker import SensorProjectionWorker  # noqa: E402
from service.sensor_service import SensorService  # noqa: E402
from tlv.parser import Parser  # noqa: E402

from src.ingest_store import IngestStore  # noqa: E402

TLV = "0065761000888a00c40041a80000c40142a00000e4668100000078e4158e68c0c24f"


def envelope():
    return dict(
        app_id="app",
        dev_eui="dev",
        payload=base64.b64encode(bytes.fromhex(TLV)).decode(),
        channel=1,
        frequency=920000000,
        received_at=0,
        uplink_metadata=None,
    )


def service(store):
    instance = SensorService.__new__(SensorService)
    instance._ingest_store = store
    instance._data_processor = Mock()
    instance._tlv_parser = Parser()
    instance._edgex_forwarder = Mock()
    instance.device_info = Mock()
    instance.device_info.get_device_id.return_value = "sensor-1"
    instance.event_service = Mock()
    return instance


@pytest.fixture
def pg_db():
    dsn = os.environ.get("CCTV_TEST_PG_DSN")
    if not dsn:
        pytest.skip("CCTV_TEST_PG_DSN must point to an isolated test PostgreSQL")
    schema = "test_sensor_" + uuid4().hex
    admin = psycopg2.connect(dsn)
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    pool = None
    try:
        pool = psycopg2.pool.ThreadedConnectionPool(
            1,
            4,
            dsn=dsn,
            options=f"-c search_path={schema}",
        )
        _ensure_schema(pool)
        db = DB(pool)
        ensure_projection_schema(db)
        yield db
    finally:
        if pool is not None:
            pool.closeall()
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )
        admin.close()


def packet(temperature=21):
    return encode_records(
        [
            T34954(temperature=temperature),
            SensorData(payload_tlv={"temperature": temperature}),
        ]
    )


def counts(db):
    return [
        db.execute_query(f"SELECT COUNT(*) AS count FROM {table}")[0]["count"]
        for table in ("t34954", "sensor_data", "sensor_projection_receipts")
    ]


def test_serialized_packet_retains_zero_and_timestamps():
    records = json.loads(packet(0))
    assert records["t34954"][0]["temperature"] == 0
    assert records["sensor_data"][0]["payload_tlv"] == {"temperature": 0}
    assert "+09:00" in records["t34954"][0]["received_at"]


def test_two_rows_and_receipt_commit_together_and_retry_is_idempotent(pg_db):
    payload = packet()
    write_sensor_projection(pg_db, "ing_test", payload)
    write_sensor_projection(pg_db, "ing_test", payload)
    assert counts(pg_db) == [1, 1, 1]


def test_second_table_failure_rolls_back_first_table_and_receipt(pg_db):
    pg_db.execute_insert("ALTER TABLE t34954 ADD CHECK (temperature >= 0)")
    with pytest.raises(psycopg2.IntegrityError):
        write_sensor_projection(pg_db, "ing_test", packet(-1))
    assert counts(pg_db) == [0, 0, 0]
    write_sensor_projection(pg_db, "ing_test", packet(0))
    assert counts(pg_db) == [1, 1, 1]


def test_repeated_id_with_different_body_never_overwrites_rows(pg_db):
    write_sensor_projection(pg_db, "ing_test", packet(21))
    with pytest.raises(ProjectionConflict):
        write_sensor_projection(pg_db, "ing_test", packet(99))
    assert counts(pg_db) == [1, 1, 1]
    assert pg_db.execute_query("SELECT temperature FROM t34954")[0]["temperature"] == 21


def test_concurrent_retry_of_same_id_inserts_once(pg_db):
    payload = packet()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(write_sensor_projection, pg_db, "ing_same", payload)
            for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=5)
    assert counts(pg_db) == [1, 1, 1]


def test_different_receipts_with_equal_values_are_not_dropped(pg_db):
    payload = packet()
    write_sensor_projection(pg_db, "ing_first", payload)
    write_sensor_projection(pg_db, "ing_second", payload)
    assert counts(pg_db) == [2, 2, 2]


def test_commit_succeeded_but_response_was_lost_can_be_retried(pg_db):
    payload = packet()

    class LostResponse:
        def execute_in_transaction(self, operation):
            pg_db.execute_in_transaction(operation)
            raise ConnectionError("injected lost commit response")

    with pytest.raises(ConnectionError, match="lost commit"):
        write_sensor_projection(LostResponse(), "ing_test", payload)
    write_sensor_projection(pg_db, "ing_test", payload)
    assert counts(pg_db) == [1, 1, 1]


@pytest.mark.parametrize(
    "payload,error", [("not base64!", "invalid_base64"), ("AA==", "invalid_tlv")]
)
def test_bad_tlv_remains_in_store_and_does_not_forward(tmp_path, payload, error):
    store = IngestStore(tmp_path / "common.db")
    instance = service(store)
    raw = envelope()
    raw["payload"] = payload
    accepted = []
    original_accept = store.accept

    def accept(*args):
        ingest_id = original_accept(*args)
        accepted.append(ingest_id)
        return ingest_id

    with patch.object(store, "accept", side_effect=accept):
        instance.process_sensor_data(**raw)
    receipt = store.get(accepted[0])
    assert receipt["stage"] == "quarantined"
    assert receipt["last_error"] == error
    assert receipt["raw"] == raw
    instance._data_processor.add_data.assert_not_called()
    instance._edgex_forwarder.publish.assert_not_called()


def test_accept_failure_prevents_parsing_and_forwarding(tmp_path):
    store = IngestStore(tmp_path / "common.db")
    instance = service(store)
    instance._tlv_parser = Mock()
    with patch.object(store, "accept", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            instance.process_sensor_data(**envelope())
    instance._tlv_parser.decode_lwm2m_tlv.assert_not_called()
    instance._edgex_forwarder.publish.assert_not_called()


def test_live_input_is_persisted_before_forward_and_only_new_writer_is_used(
    pg_db, tmp_path
):
    store = IngestStore(tmp_path / "common.db")
    instance = service(store)

    def assert_persisted(**_kwargs):
        receipt = store.get(store.pending()[0])
        assert receipt["stage"] == "ready"

    instance._edgex_forwarder.publish.side_effect = assert_persisted
    instance.process_sensor_data(**envelope())
    ingest_id = store.pending()[0]
    instance._data_processor.add_data.assert_not_called()
    worker = SensorProjectionWorker(store, pg_db, instance._recover_receipt)
    worker.drain_once()
    assert store.get(ingest_id)["stage"] == "complete"
    assert counts(pg_db) == [1, 1, 1]
    assert instance._edgex_forwarder.publish.call_count == 1


def test_db_outage_retains_job_and_recovery_saves_once(pg_db, tmp_path):
    store = IngestStore(tmp_path / "common.db")
    instance = service(store)
    instance.process_sensor_data(**envelope())
    ingest_id = store.pending()[0]
    unavailable = Mock()
    unavailable.execute_in_transaction.side_effect = psycopg2.OperationalError(
        "offline"
    )
    with patch("src.ingest_store.time.time", return_value=0):
        SensorProjectionWorker(
            store, unavailable, instance._recover_receipt
        ).drain_once()
    assert store.get(ingest_id)["stage"] == "ready"
    assert store.get(ingest_id)["attempts"] == 1
    SensorProjectionWorker(store, pg_db, instance._recover_receipt).drain_once()
    assert store.get(ingest_id)["stage"] == "complete"
    assert counts(pg_db) == [1, 1, 1]


@pytest.mark.parametrize(
    "boundary", ["receipt", "prepared", "before_commit", "after_commit"]
)
def test_process_exit_and_restart_recovers_without_outputs(pg_db, tmp_path, boundary):
    path = tmp_path / "common.db"
    schema = pg_db.execute_query("SELECT current_schema() AS schema")[0]["schema"]
    script = """
import json, os, sys
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, str(Path.cwd() / 'parser-python'))
import psycopg2.pool
from database.connection import DB
from database.sensor_projection import write_sensor_projection
from service.sensor_service import SensorService
from src.ingest_store import IngestStore
from tlv.parser import Parser
path, boundary, schema, raw = sys.argv[1:]
store = IngestStore(path)
ingest_id = store.accept('tlv_mqtt', 'sensor-1', json.loads(raw))
print(ingest_id, flush=True)
if boundary == 'receipt':
    os._exit(42)
instance = SensorService.__new__(SensorService)
instance._ingest_store = store
instance._tlv_parser = Parser()
instance.device_info = Mock()
instance.device_info.get_device_id.return_value = 'sensor-1'
instance._recover_receipt(store.get(ingest_id))
projection = store.get(ingest_id)['projection_json']
if boundary == 'prepared':
    os._exit(42)
pool = psycopg2.pool.ThreadedConnectionPool(1, 1, dsn=os.environ['CCTV_TEST_PG_DSN'], options='-c search_path=' + schema)
db = DB(pool)
if boundary == 'before_commit':
    def uncommitted(operation):
        with db._get_conn() as connection:
            operation(connection)
            os._exit(42)
    db.execute_in_transaction = uncommitted
write_sensor_projection(db, ingest_id, projection)
os._exit(42)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            boundary,
            schema,
            json.dumps(envelope()),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 42, result.stderr
    ingest_id = result.stdout.strip()
    assert ingest_id.startswith("ing_")
    recovered = IngestStore(path)
    restored_service = service(recovered)
    worker = SensorProjectionWorker(recovered, pg_db, restored_service._recover_receipt)
    worker.drain_once()
    worker.drain_once()
    assert recovered.get(ingest_id)["stage"] == "complete"
    assert counts(pg_db) == [1, 1, 1]
    assert (
        pg_db.execute_query("SELECT ingest_id FROM sensor_projection_receipts")[0][
            "ingest_id"
        ]
        == ingest_id
    )
    restored_service._edgex_forwarder.publish.assert_not_called()
    restored_service.event_service.add_notification_to_queue.assert_not_called()


def test_constructor_selects_persistent_worker_and_stops_it(tmp_path):
    store = IngestStore(tmp_path / "common.db")
    with (
        patch("service.sensor_service.DataProcessor") as memory_processor,
        patch("service.sensor_service.EventService") as events,
    ):
        instance = SensorService(Mock(), ingest_store=store)
        try:
            memory_processor.assert_not_called()
            assert instance._projection_worker._thread.is_alive()
        finally:
            instance.close()
        assert not instance._projection_worker._thread.is_alive()
        events.return_value.close.assert_called_once()


def test_invalid_projection_is_retained_without_blocking_next_packet(pg_db, tmp_path):
    pg_db.execute_insert("ALTER TABLE t34954 ADD CHECK (temperature >= 0)")
    store = IngestStore(tmp_path / "common.db")
    bad = store.accept("tlv_mqtt", "sensor-1", {"value": -1})
    good = store.accept("tlv_mqtt", "sensor-1", {"value": 1})
    store.prepare(bad, packet(-1))
    store.prepare(good, packet(1))
    SensorProjectionWorker(store, pg_db, Mock()).drain_once()
    assert store.get(bad)["stage"] == "quarantined"
    assert store.get(bad)["raw"] == {"value": -1}
    assert store.get(good)["stage"] == "complete"
    assert counts(pg_db) == [1, 1, 1]


def test_conflicting_projection_is_quarantined_without_replacing_rows(pg_db, tmp_path):
    store = IngestStore(tmp_path / "common.db")
    ingest_id = store.accept("tlv_mqtt", "sensor-1", {"value": 21})
    write_sensor_projection(pg_db, ingest_id, packet(21))
    store.prepare(ingest_id, packet(99))
    SensorProjectionWorker(store, pg_db, Mock()).drain_once()
    assert store.get(ingest_id)["stage"] == "quarantined"
    assert store.get(ingest_id)["last_error"] == "ProjectionConflict"
    assert counts(pg_db) == [1, 1, 1]
