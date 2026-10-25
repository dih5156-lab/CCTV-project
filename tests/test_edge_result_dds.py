"""DDS 경로의 신원·저장 경계와 MQTT/DDS 동시 송신 금지를 검사한다."""

import copy
import json
import os
import sqlite3
import threading
from unittest.mock import MagicMock

import pytest

from runners.run_edge_orchestration import build_parser
from src.services.edge_orchestrator import RegistryError
from src.services.edge_result_dds import EdgeDdsCollector, _Endpoint, result_topic
from src.services.edge_result_delivery import acquire_delivery_lock, deliver_pending
from tests.test_edge_ownership import lab as lab
from tests.test_edge_result_transport import LocalController, sample, setup_node
from tests.test_edge_service import private_json


def test_dds_storage_ack_preserves_original_and_rejects_topic_identity(lab, tmp_path):
    _, context, service = setup_node(lab, tmp_path)
    collector = EdgeDdsCollector.__new__(EdgeDdsCollector)
    collector.store, collector.node_ids = lab.store, set(lab.nodes)
    sent = []
    controller = LocalController(lab.store, "jetson-a")
    try:
        payload = sample(context, service["session_id"], kind="observation")
        event_id = payload["event"]["event_id"]

        def publish(value):
            sent.append(copy.deepcopy(value))
            return "waiting_for_central_receipt"

        deliver_pending(context, "jetson-a", controller, publish, threading.Event())
        assert len(context.pending()) == 1 and sent == [payload]
        collector.accept("jetson-b", json.dumps(payload))
        with pytest.raises(RegistryError, match="receipt_not_found"):
            lab.store.receipt("jetson-a", event_id)
        lab.store.db.execute("PRAGMA query_only=ON")
        with pytest.raises(sqlite3.OperationalError):
            collector.accept("jetson-a", json.dumps(payload))
        lab.store.db.execute("PRAGMA query_only=OFF")
        collector.accept("jetson-a", json.dumps(payload))
        receipt = lab.store.receipt("jetson-a", event_id)
        assert receipt["decision"] == "observed"
        original_request = controller.request
        controller.request = lambda *args: (200, dict(receipt, sha256="0" * 64))
        deliver_pending(context, "jetson-a", controller, publish, threading.Event())
        assert context.pending()
        controller.request = original_request
        deliver_pending(context, "jetson-a", controller, publish, threading.Event())
        assert context.pending() == []
        assert len(sent) == 1
        collector.accept("jetson-a", json.dumps(payload))
        assert lab.store.receipt("jetson-a", event_id) == receipt
    finally:
        context.close()


def test_delivery_drains_outage_backlog_in_bounded_batches(lab, tmp_path):
    _, context, service = setup_node(lab, tmp_path)
    controller = LocalController(lab.store, "jetson-a")
    sent = []
    stop = threading.Event()
    try:
        originals = [
            sample(context, service["session_id"], kind="observation")
            for _ in range(105)
        ]

        def publish(payload):
            sent.append(payload["event"]["event_id"])
            lab.store.receive("jetson-a", payload)
            return "waiting_for_central_receipt"

        # 정상 유입과 함께 복구할 여유를 확보하되 한 회차와 종료 응답은 제한한다.
        deliver_pending(context, "jetson-a", controller, publish, stop)
        assert len(sent) == 100
        stop.set()
        deliver_pending(context, "jetson-a", controller, publish, stop)
        assert len(sent) == 100
        stop.clear()
        for _ in range(3):
            deliver_pending(context, "jetson-a", controller, publish, stop)
        assert context.pending() == []
        assert set(sent) == {item["event"]["event_id"] for item in originals}
        assert len(sent) == len(originals)
    finally:
        context.close()


@pytest.mark.parametrize("raw", ["null", "[]", "{", "x" * 49153, '{"kind":NaN}'])
def test_dds_malformed_original_is_quarantined_without_receipt(lab, raw):
    collector = EdgeDdsCollector.__new__(EdgeDdsCollector)
    collector.store, collector.node_ids = lab.store, set(lab.nodes)
    collector.accept("jetson-a", raw)
    row = lab.store.db.execute("SELECT topic,total_bytes FROM rejected_mqtt").fetchone()
    assert row["topic"] == "dds:" + result_topic("jetson-a")
    assert row["total_bytes"] == len(raw.encode())


def test_delivery_lock_is_shared_by_transports_and_release_allows_restart(tmp_path):
    path = tmp_path / "results.sqlite3"
    fd = acquire_delivery_lock(path)
    try:
        with pytest.raises(BlockingIOError):
            acquire_delivery_lock(path)
    finally:
        os.close(fd)
    os.close(acquire_delivery_lock(path))
    lock = tmp_path / "results.sqlite3.delivery.lock"
    lock.unlink()
    lock.symlink_to(tmp_path / "other")
    with pytest.raises(OSError):
        acquire_delivery_lock(path)
    assert not (tmp_path / "other").exists()


def test_ros_node_encoding_preserves_distinct_existing_ids():
    assert len({result_topic(value) for value in ["a-b", "a_b", "a.b", "A-b"]}) == 4
    assert all(
        part.replace("_", "a").isalnum()
        for part in result_topic("jetson-a").split("/")[1:]
    )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("ROS_SECURITY_ENABLE", "false"),
        ("ROS_SECURITY_STRATEGY", "Permissive"),
        ("ROS_SECURITY_KEYSTORE", ""),
        ("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
    ],
)
def test_dds_cannot_disable_security_or_change_rmw(monkeypatch, setting, value):
    for key, expected in {
        "ROS_SECURITY_ENABLE": "true",
        "ROS_SECURITY_STRATEGY": "Enforce",
        "ROS_SECURITY_KEYSTORE": "/not-used",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }.items():
        monkeypatch.setenv(key, expected)
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError, match="SROS2"):
        _Endpoint(73, "/soma/controller")


@pytest.mark.parametrize("domain", [-1, 233, True, "73"])
def test_dds_domain_must_be_explicit_integer(domain):
    with pytest.raises(ValueError, match="domain_id"):
        _Endpoint(domain, "/soma/controller")


def test_cli_refuses_dual_controller_transport():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            [
                "controller",
                "--bind",
                "127.0.0.1",
                "--state-dir",
                "unused",
                "--credentials",
                "unused",
                "--cert",
                "unused",
                "--key",
                "unused",
                "--result-mqtt",
                "mqtt.json",
                "--result-dds",
                "dds.json",
            ]
        )
    assert error.value.code == 2


def test_sidecar_observation_runs_without_ros_and_recovers_storage_error(
    tmp_path, monkeypatch
):
    from runners import run_edge_orchestration as runner
    from src.services import edge_service

    owned, server = MagicMock(), MagicMock()
    owned.boot_id = "fixture-boot"
    owned.capture_observation.side_effect = [
        sqlite3.OperationalError("busy"),
        None,
        None,
    ]
    server.__enter__.return_value = server
    server.server_port = 12345
    stop = threading.Event()
    elapsed = [0]

    def handle_request():
        elapsed[0] += 1
        if elapsed[0] == 3:
            stop.set()

    server.handle_request.side_effect = handle_request
    monkeypatch.setattr(runner.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(edge_service, "ServiceManifest", MagicMock())
    monkeypatch.setattr(
        edge_service, "EdgeServiceSupervisor", MagicMock(return_value=owned)
    )
    monkeypatch.setattr(
        edge_service, "create_service_server", MagicMock(return_value=server)
    )
    credentials = private_json(
        tmp_path / "credentials.json",
        {"reader_token": "r" * 32, "control_token": "c" * 32},
    )
    args = build_parser().parse_args(
        [
            "service",
            "--node-id",
            "jetson-a",
            "--manifest",
            "unused.json",
            "--state-dir",
            str(tmp_path / "state"),
            "--credentials",
            str(credentials),
            "--enable-results",
            "--result-sidecar",
        ]
    )
    runner.run_service(args, stop)
    assert owned.capture_observation.call_count == 3
    assert owned.tick.call_count == 3
    owned.close.assert_called_once_with()


@pytest.mark.parametrize(
    "variable", ["ROS_SECURITY_ENCLAVE_OVERRIDE", "ROS_SECURITY_LOOKUP_DIRECTORY"]
)
def test_dds_rejects_global_security_path_override(monkeypatch, variable):
    for key, value in {
        "ROS_SECURITY_ENABLE": "true",
        "ROS_SECURITY_STRATEGY": "Enforce",
        "ROS_SECURITY_KEYSTORE": "/not-used",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        variable: "/other",
    }.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="overrides"):
        _Endpoint(73, "/soma/controller")
