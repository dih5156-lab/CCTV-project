"""Opt-in test against an isolated unpacked TLS bundle; never uses operational DBs."""

import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

import pytest
import requests

from runners.run_monitoring_remote_poc import build_workers, delivery_summary, seed_demo
from src.monitoring_contract import (
    MonitoringBinding,
    MonitoringContractError,
    parse_record,
    validate_ack,
)
from src.protocols.monitoring_http import HttpMonitoringReader
from src.protocols.record_mqtt import MqttRecordTransport
from src.record_delivery import record_topic


@pytest.fixture
def remote():
    directory = os.environ.get("CCTV_TEST_REMOTE_DIR")
    if not directory:
        pytest.skip("CCTV_TEST_REMOTE_DIR: opt-in isolated TLS bundle required")
    root = Path(directory).resolve()
    if root.parent != Path("/tmp") or not root.name.startswith("cctv-remote-"):
        pytest.fail("Only /tmp/cctv-remote-* test directories are allowed")
    client = root / "cctv-monitoring-poc/client"
    config = json.loads((client / "config.json").read_text())
    assert config["mqtt_host"] == "127.0.0.1"
    assert config["https_url"].startswith("https://127.0.0.1:")
    source, binding, workers = build_workers(root / "source.db", client)
    return root, client, source, binding, workers


def test_real_tls_mixed_records_duplicate_and_auth_failures(remote):
    root, client, source, binding, workers = remote
    reader, sender = workers[0].reader, workers[0].sender
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            reader.base_url + "/v1/health",
            headers={"Authorization": "Bearer " + reader.token},
            verify=reader.ca_file,
            timeout=5,
        )
        assert response.status_code == 200
    seed_demo(source)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for worker in workers:
            worker.drain_once(10)
        if delivery_summary(source, binding).get("stored") == 2:
            break
        time.sleep(0.1)
    assert delivery_summary(source, binding) == {
        "states": {"stored": 2},
        "total": 2,
        "stored": 2,
    }
    with sqlite3.connect(source.path) as connection:
        connection.row_factory = sqlite3.Row
        jobs = [
            dict(row)
            for row in connection.execute(
                "SELECT j.*, m.wire_payload, m.payload_sha256, m.ack_json FROM delivery_jobs j JOIN monitoring_delivery m USING(job_id)"
            )
        ]
    assert {parse_record(job["wire_payload"]).record_type for job in jobs} == {
        "event",
        "sensor_observation",
    }
    for job in jobs:
        job["topic"] = record_topic(job["destination"])
        validate_ack(
            json.loads(job["ack_json"]), binding, parse_record(job["wire_payload"])
        )
        sender.send(job, timeout=5)
        ack = reader.get_delivery_status(binding, job["job_id"], timeout=5)
        validate_ack(ack, binding, parse_record(job["wire_payload"]))
    target = root / "cctv-monitoring-poc/state/data/receiver.db"
    time.sleep(0.2)
    with sqlite3.connect(target) as connection:
        assert (
            connection.execute("SELECT count(*) FROM monitoring_records").fetchone()[0]
            == 2
        )
        assert (
            connection.execute("SELECT count(*) FROM monitoring_receipts").fetchone()[0]
            == 2
        )
    wrong_token = HttpMonitoringReader(
        reader.base_url, secrets.token_urlsafe(32), reader.ca_file
    )
    with pytest.raises(MonitoringContractError, match="unauthorized"):
        wrong_token.get_delivery_status(binding, jobs[0]["job_id"], timeout=5)
    wrong_binding = MonitoringBinding(
        "wrong-origin", binding.consumer_id, binding.destination
    )
    with pytest.raises(MonitoringContractError, match="binding_mismatch"):
        reader.get_delivery_status(wrong_binding, jobs[0]["job_id"], timeout=5)
    untrusted = HttpMonitoringReader(
        reader.base_url, reader.token, requests.certs.where()
    )
    with pytest.raises(requests.exceptions.SSLError):
        untrusted.get_delivery_status(binding, jobs[0]["job_id"], timeout=5)
    bad_sender = MqttRecordTransport(
        sender.broker,
        port=sender.port,
        username=sender.username,
        password=secrets.token_urlsafe(32),
        tls_context=sender.tls_context,
    )
    with pytest.raises((TimeoutError, ConnectionError)):
        bad_sender.send(jobs[0], timeout=2)


def test_existing_committed_status_lookup(remote):
    _, _, source, binding, workers = remote
    with sqlite3.connect(source.path) as connection:
        jobs = connection.execute(
            "SELECT job_id,wire_payload FROM monitoring_delivery WHERE state='stored'"
        ).fetchall()
    assert (
        len(jobs) == 2
    ), "Run the mixed-record test before restarting the isolated receiver"
    for delivery_id, wire in jobs:
        ack = workers[0].reader.get_delivery_status(binding, delivery_id, timeout=5)
        validate_ack(ack, binding, parse_record(wire))
