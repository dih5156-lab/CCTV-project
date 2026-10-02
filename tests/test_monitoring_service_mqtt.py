"""Isolated real MQTT + local ACK lookup + threaded service/CLI lifecycle."""

import json
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from src.event_ingest_store import EventIngestStore
from src.monitoring_reconciliation import current_delivery, reconcile
from src.monitoring_service import build_local_service
from src.monitoring_snapshot import MonitoringSnapshots
from src.sensor_record import SensorRecordStore, SensorRecordWorker
from tests.test_monitoring_service import wait_until
from tests.test_mqtt_durable_integration import broker as broker  # noqa: F401


def make_service(tmp_path, broker, **kwargs):
    return build_local_service(
        tmp_path / "source.db",
        tmp_path / "receiver.db",
        "consumer-test",
        "record-mqtt:cctv/records/v1/" + uuid4().hex,
        broker,
        namespaces=["tlv_mqtt", "video_detection"],
        interval=0.02,
        timeout=0.5,
        lease_seconds=4,
        ack_wait=0.02,
        **kwargs,
    )


def test_real_service_sensor_video_commits_metrics_and_restart(broker, tmp_path):
    service, source, receiver, binding, timings, commits = make_service(
        tmp_path, broker, clock_sync="confirmed"
    )
    inputs = SensorRecordStore(source.path, commit_observer=commits.committed)
    event = {"event_id": "evt-service", "value": 0}
    receipt = inputs.accept_event_json(
        "video_detection", event, destination="mqtt:cctv/ai/events"
    )
    inputs.record_event(receipt, event, converter_version="test", recovery=True)
    sensor = inputs.accept("tlv_mqtt", "sensor-1", {"raw": 0})
    inputs.prepare(
        sensor,
        json.dumps(
            {
                "sensor_data": [
                    {
                        "device_id": "sensor-1",
                        "object_id": "34954",
                        "payload_tlv": {"temperature_c": 0, "created_at": 0},
                    }
                ]
            }
        ),
    )
    SensorRecordWorker(inputs).drain_once()
    service.start()
    try:
        wait_until(lambda: service.status()["completed"] == 2, timeout=15)
    finally:
        service.close()
    assert commits.report()["latency"]["valid"] == 2
    assert len(commits.report()["samples"]) == 6
    assert timings.report()["mqtt_publish"]["valid"] == 2
    audit = MonitoringSnapshots(tmp_path / "snapshots.db")
    source_set, consumer_set = audit.create_source(
        source, binding
    ), audit.create_consumer(receiver, binding)
    result = reconcile(
        audit,
        binding,
        source_set["snapshot_id"],
        consumer_set["snapshot_id"],
        lookup_current=lambda b, d: current_delivery(source, b, d),
    )
    assert result["all_inputs_accounted"] and result["counts"]["matched"] == 2
    service.start()
    service.close()
    with sqlite3.connect(source.path) as connection:
        assert (
            connection.execute(
                "SELECT state FROM delivery_jobs WHERE destination='mqtt:cctv/ai/events'"
            ).fetchone()[0]
            == "held"
        )


def test_real_consumer_storage_failure_recovers_without_new_delivery(broker, tmp_path):
    service, source, receiver, binding, timings, commits = make_service(
        tmp_path, broker
    )
    inputs = EventIngestStore(source.path)
    event = {"event_id": "evt-retry"}
    receipt = inputs.accept_event_json("video_detection", event, destination=None)
    inputs.record_event(receipt, event, converter_version="test")
    with sqlite3.connect(receiver.path) as connection:
        connection.execute(
            "CREATE TRIGGER write_failure BEFORE INSERT ON monitoring_receipts BEGIN SELECT RAISE(ABORT,'full'); END"
        )
    service.start()
    try:
        wait_until(
            lambda: service.receiver.status()["storage_or_transport_failures"] >= 1,
            timeout=15,
        )
        with sqlite3.connect(source.path) as connection:
            original = connection.execute(
                "SELECT job_id,wire_payload FROM monitoring_delivery"
            ).fetchone()
            assert (
                connection.execute(
                    "SELECT acked_at FROM monitoring_delivery"
                ).fetchone()[0]
                is None
            )
        with sqlite3.connect(receiver.path) as connection:
            connection.execute("DROP TRIGGER write_failure")
        wait_until(lambda: service.status()["completed"] == 1, timeout=15)
    finally:
        service.close()
    with sqlite3.connect(source.path) as connection:
        assert (
            connection.execute(
                "SELECT job_id,wire_payload FROM monitoring_delivery"
            ).fetchone()
            == original
        )
    assert commits.report()["latency"]["unknown"] == 1


def test_cli_sigterm_stops_and_second_process_resumes(broker, tmp_path):
    destination = "record-mqtt:cctv/records/v1/" + uuid4().hex
    command = [
        sys.executable,
        "runners/run_monitoring_local.py",
        "--source-db",
        str(tmp_path / "source.db"),
        "--receiver-db",
        str(tmp_path / "receiver.db"),
        "--consumer-id",
        "consumer-cli",
        "--destination",
        destination,
        "--mqtt-socket",
        broker,
        "--namespace",
        "video_detection",
        "--interval",
        "0.02",
        "--timeout",
        "0.5",
        "--lease-seconds",
        "4",
        "--ack-wait",
        "0.02",
    ]
    root = Path(__file__).resolve().parents[1]
    with (tmp_path / "stdout.log").open("w+") as stdout:
        process = subprocess.Popen(
            command, cwd=root, stdout=stdout, stderr=subprocess.PIPE
        )
        try:
            wait_until(
                lambda: '"event": "started"' in (tmp_path / "stdout.log").read_text(),
                timeout=10,
            )
            process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, stderr.decode()
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    inputs = EventIngestStore(tmp_path / "source.db")
    event = {"event_id": "after-stop"}
    receipt = inputs.accept_event_json("video_detection", event, destination=None)
    inputs.record_event(receipt, event, converter_version="test")
    result = subprocess.run(
        command + ["--run-seconds", "1"], cwd=root, capture_output=True, timeout=15
    )
    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(result.stdout.decode().splitlines()[-1])
    assert report["service"]["completed"] == 1
    assert report["service"]["state"] == "stopped"
    assert report["commits"]["latency"]["unknown"] == 1
