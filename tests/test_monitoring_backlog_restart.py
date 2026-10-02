"""Mixed backlog on real SQLite; modeled transport and controlled retry clock."""

import hashlib
import json
import multiprocessing
import os
import sqlite3
import time

import pytest

from src.core.video_event_ingest import VideoEventIngest
from src.monitoring_contract import MonitoringBinding
from src.monitoring_delivery import MonitoringDeliveryStore
from src.monitoring_receiver import MonitoringReceiverStore
from src.monitoring_reconciliation import current_delivery, reconcile
from src.monitoring_snapshot import MonitoringSnapshots
from src.monitoring_worker import MonitoringDeliveryWorker
from src.sensor_record import SensorRecordStore, SensorRecordWorker
from tests.test_monitoring_delivery import DESTINATION
from tests.test_monitoring_outage_mqtt import deliveries
from tests.test_monitoring_worker import Broker, Reader


def restart_worker(source_path, receiver_path, report_path, point, now):
    """A spawned process constructs all stores/workers from on-disk state."""
    started = time.monotonic()
    source = MonitoringDeliveryStore(source_path)
    binding = MonitoringBinding(source.origin_id, "backlog-consumer", DESTINATION)
    receiver = MonitoringReceiverStore(receiver_path, consumer_id=binding.consumer_id)
    broker, reader = Broker(receiver, binding), Reader(receiver, None)
    clock = [now]
    completed = 0

    def report():
        report_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "point": point,
                    "published": len(broker.sent),
                    "completed": completed,
                    "elapsed_seconds": time.monotonic() - started,
                    "retry_clock_start": now,
                    "retry_clock_end": clock[0],
                }
            )
        )

    send, complete = broker.send, source.complete_monitoring

    def send_then_exit(*args, **kwargs):
        ack = send(*args, **kwargs)
        if point == "consumer_commit" and len(broker.sent) == 37:
            report()
            os._exit(42)
        return ack

    def complete_then_exit(*args, **kwargs):
        nonlocal completed
        stored = complete(*args, **kwargs)
        completed += int(stored)
        if point == "ack_save" and completed == 51:
            report()
            os._exit(42)
        return stored

    broker.send, source.complete_monitoring = send_then_exit, complete_then_exit
    workers = [
        MonitoringDeliveryWorker(
            source,
            broker,
            reader,
            namespace=namespace,
            binding=binding,
            clock=lambda: clock[0],
        )
        for namespace in ("video_detection", "tlv_mqtt")
    ]
    for _ in range(8):
        for worker in workers:
            worker.drain_once(limit=100)
        if all(row["state"] == "stored" for row in deliveries(source).values()):
            report()
            return
        clock[0] += 61
    raise AssertionError("backlog did not drain within 8 controlled retry cycles")


def run_restart(tmp_path, source, receiver, point, now, expected_exit):
    report_path = tmp_path / f"{point}.json"
    process = multiprocessing.get_context("spawn").Process(
        target=restart_worker,
        args=(source.path, receiver.path, report_path, point, now),
    )
    process.start()
    process.join(90)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail(f"{point} process timed out")
    assert process.exitcode == expected_exit
    return {**json.loads(report_path.read_text()), "exit_code": process.exitcode}


def database_size(path):
    with sqlite3.connect(path) as connection:
        pages = connection.execute("PRAGMA page_count").fetchone()[0]
        page_bytes = connection.execute("PRAGMA page_size").fetchone()[0]
        free = connection.execute("PRAGMA freelist_count").fetchone()[0]
    return {
        "file_bytes": path.stat().st_size,
        "page_bytes": page_bytes,
        "pages": pages,
        "free_pages": free,
        "used_page_bytes": (pages - free) * page_bytes,
    }


def add_batch(inputs, start, pairs):
    video = VideoEventIngest(inputs, backend="test", topic_prefix="cctv/ai/events")
    for number in range(start, start + pairs):
        padding = hashlib.sha256(str(number).encode()).hexdigest() * 32
        values = {
            "temperature_c": 20 + number % 10,
            "humidity_pct": 50,
            "created_at": 1790000000 + number,
        }
        reference = video.accept(
            {
                "event_id": f"backlog-{number}",
                "type": "fall",
                "camera_id": "camera-1",
                "occurred_at": 1790000000 + number,
                "metadata": {"padding": padding},
            },
            "camera-1",
        )
        video.defer(reference)
        receipt = inputs.accept(
            "tlv_mqtt", "sensor-1", {"values": values, "padding": padding}
        )
        inputs.prepare(
            receipt,
            json.dumps(
                {
                    "sensor_data": [
                        {
                            "device_id": "sensor-1",
                            "object_id": "34954",
                            "payload_tlv": values,
                        }
                    ]
                }
            ),
        )
    video.recover_pending()
    SensorRecordWorker(inputs).drain_once()


def test_mixed_backlog_growth_and_restart(tmp_path, record_property):
    source = MonitoringDeliveryStore(tmp_path / "source.db")
    inputs = SensorRecordStore(source.path)
    binding = MonitoringBinding(source.origin_id, "backlog-consumer", DESTINATION)
    receiver = MonitoringReceiverStore(
        tmp_path / "receiver.db", consumer_id=binding.consumer_id
    )
    receiver.register_binding(binding)
    clock = [100.0]
    broker, reader = Broker(receiver, binding), Reader(receiver, clock)
    reader.unavailable = True
    workers = [
        MonitoringDeliveryWorker(
            source,
            broker,
            reader,
            namespace=namespace,
            binding=binding,
            clock=lambda: clock[0],
        )
        for namespace in ("video_detection", "tlv_mqtt")
    ]
    checkpoints = [
        {
            "records": 0,
            "source": database_size(source.path),
            "receiver": database_size(receiver.path),
        }
    ]
    started = time.monotonic()
    for batch in range(3):
        add_batch(inputs, batch * 50, 50)
        clock[0] += 61
        for worker in workers:
            worker.drain_once(limit=1000)
        rows = deliveries(source)
        assert len(rows) == (batch + 1) * 100
        assert all(
            row["state"] == "uncertain" and row["ack_json"] is None
            for row in rows.values()
        )
        assert not broker.sent
        checkpoints.append(
            {
                "records": len(rows),
                "source": database_size(source.path),
                "receiver": database_size(receiver.path),
            }
        )
    assert (
        checkpoints[-1]["source"]["file_bytes"] > checkpoints[0]["source"]["file_bytes"]
    )
    assert checkpoints[-1]["receiver"] == checkpoints[0]["receiver"]
    record_property("outage_build_seconds", time.monotonic() - started)
    record_property("checkpoints", json.dumps(checkpoints))
    frozen = {
        key: (row["wire_payload"], row["payload_sha256"]) for key, row in rows.items()
    }
    with sqlite3.connect(source.path) as connection:
        raw_before = connection.execute(
            "SELECT ingest_id,raw_payload,raw_sha256 FROM ingest_receipts ORDER BY ingest_id"
        ).fetchall()
    recovery_started = time.monotonic()
    first = run_restart(tmp_path, source, receiver, "consumer_commit", 1000, 42)
    assert first["published"] == 37 and first["completed"] == 0
    assert sum(row["state"] == "stored" for row in deliveries(source).values()) == 0
    with sqlite3.connect(receiver.path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_records").fetchone()[0]
            == 37
        )
    second = run_restart(tmp_path, source, receiver, "ack_save", 2000, 42)
    assert second["completed"] == 51
    assert sum(row["state"] == "stored" for row in deliveries(source).values()) == 51
    final = run_restart(tmp_path, source, receiver, "recovery", 3000, 0)
    assert final["completed"] == 249 and final["published"] == 0
    assert sum(run["published"] for run in (first, second, final)) == 300
    assert len({run["pid"] for run in (first, second, final)} | {os.getpid()}) == 4
    record_property("processes", json.dumps([first, second, final]))
    rows = deliveries(MonitoringDeliveryStore(source.path))
    assert {
        key: (row["wire_payload"], row["payload_sha256"]) for key, row in rows.items()
    } == frozen
    assert all(row["state"] == "stored" for row in rows.values())
    recovery_wall = time.monotonic() - recovery_started
    with sqlite3.connect(receiver.path) as connection:
        received = connection.execute(
            "SELECT delivery_id,raw_payload,payload_sha256,receiver_record_id FROM monitoring_receipts"
        ).fetchall()
        assert {row[0]: (row[1], row[2]) for row in received} == frozen
        assert dict(
            connection.execute(
                "SELECT source_namespace,count(*) FROM monitoring_records GROUP BY source_namespace"
            )
        ) == {"tlv_mqtt": 150, "video_detection": 150}
        assert (
            connection.execute("SELECT count(*) FROM monitoring_rejected").fetchone()[0]
            == 0
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    for delivery_id, wire, digest, receiver_record_id in received:
        ack = json.loads(rows[delivery_id]["ack_json"])
        assert ack["ack_kind"] == "consumer_committed"
        assert ack["delivery_id"] == delivery_id
        assert ack["payload_sha256"] == digest == hashlib.sha256(wire).hexdigest()
        assert ack["receiver_record_id"] == receiver_record_id
        assert ack["origin_id"] == binding.origin_id
        assert ack["consumer_id"] == binding.consumer_id
    with sqlite3.connect(source.path) as connection:
        assert (
            connection.execute(
                "SELECT ingest_id,raw_payload,raw_sha256 FROM ingest_receipts ORDER BY ingest_id"
            ).fetchall()
            == raw_before
        )
        assert len(raw_before) == 300
        assert dict(
            connection.execute(
                "SELECT source_namespace,count(*) FROM canonical_records GROUP BY source_namespace"
            )
        ) == {"tlv_mqtt": 150, "video_detection": 150}
        assert connection.execute(
            "SELECT count(*),min(state),max(state),sum(attempts) FROM delivery_jobs "
            "WHERE destination='mqtt:cctv/ai/events'"
        ).fetchone() == (150, "held", "held", 0)
        attempts = connection.execute(
            "SELECT count(*) FROM monitoring_attempts"
        ).fetchone()[0]
        assert (
            connection.execute(
                "SELECT count(*) FROM record_delivery_attempts"
            ).fetchone()[0]
            == 300
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM monitoring_attempts WHERE outcome='lease_expired'"
            ).fetchone()[0]
            == 1
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    audit = MonitoringSnapshots(tmp_path / "snapshots.db")
    result = reconcile(
        audit,
        binding,
        audit.create_source(source, binding)["snapshot_id"],
        audit.create_consumer(receiver, binding)["snapshot_id"],
        lookup_current=lambda b, d: current_delivery(source, b, d),
    )
    assert result["all_inputs_accounted"] and result["counts"]["matched"] == 300
    record_property("reconciliation", json.dumps(result))
    record_property(
        "final_sizes",
        json.dumps(
            {
                "source": database_size(source.path),
                "receiver": database_size(receiver.path),
                "snapshot": database_size(audit.path),
            }
        ),
    )
    record_property("recovery_wall_seconds", recovery_wall)
    record_property("monitoring_attempts", attempts)
    record_property("raw_utf8_bytes", sum(len(row[1].encode()) for row in raw_before))
    record_property("wire_bytes", sum(len(row[0]) for row in frozen.values()))
    record_property("physical_jobs_held", 150)
