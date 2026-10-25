"""실제 보안 DDS 프로세스와 HTTPS 저장 ACK의 인계·단절·재시도를 검사한다."""

import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from src.services.edge_controller_api import create_controller_server
from src.services.edge_result_context import EdgeResultContext
from src.services.edge_result_dds import EdgeDdsCollector, node_component, result_topic
from tests.test_edge_orchestration_remote import certificates as certificates
from tests.test_edge_ownership import lab as lab
from tests.test_edge_ownership import observed, prepare, result

pytestmark = pytest.mark.skipif(
    not os.environ.get("EDGE_DDS_TEST_KEYSTORES"),
    reason="격리 ROS 2/SROS2 시험 키 경로를 명시해야 한다",
)


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("DDS integration deadline exceeded")


def write_private(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


def test_secure_dds_handoff_receiver_restart_backlog_and_storage_ack(
    lab, certificates, tmp_path_factory, monkeypatch
):
    tmp_path = tmp_path_factory.mktemp("dds")
    keys = Path(os.environ["EDGE_DDS_TEST_KEYSTORES"])
    domain_id = int(os.environ.get("EDGE_DDS_TEST_DOMAIN", "73"))
    monkeypatch.setenv("ROS_SECURITY_KEYSTORE", str(keys / "controller"))
    tokens = {node_id: uuid.uuid4().hex for node_id in lab.nodes}
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert, key = certificates[0]
    tls.load_cert_chain(cert, key)
    lab.registry.allow_remote_nodes = False
    server = create_controller_server(
        lab.registry,
        lab.store,
        tokens,
        "r" * 32,
        "c" * 32,
        "x" * 32,
        host="127.0.0.1",
        port=0,
        tls_context=tls,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    collector = EdgeDdsCollector(lab.store, tokens, domain_id)
    collector.start()
    contexts, processes, files = {}, [], []
    receipts = []

    def start_relay(node_id):
        environment = dict(
            os.environ, ROS_SECURITY_KEYSTORE=str(keys / node_component(node_id))
        )
        config = write_private(
            tmp_path / (node_id + ".json"),
            {
                "domain_id": domain_id,
                "controller_url": f"https://127.0.0.1:{server.server_port}",
                "controller_ca_file": str(cert),
                "node_token": tokens[node_id],
            },
        )
        log = (tmp_path / f"{node_id}-{len(processes)}.log").open("w")
        files.append(log)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "runners.run_edge_orchestration",
                "dds-relay",
                "--config",
                str(config),
                "--context-db",
                str(contexts[node_id].path),
                "--node-id",
                node_id,
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append(process)
        return process

    def setup_node(node_id):
        assignment, _ = prepare(lab, node_id)
        service = observed(lab, assignment)
        root = tmp_path / node_id
        root.mkdir(mode=0o700)
        context = EdgeResultContext(
            root / "results.sqlite3", create=True, clock=lambda: lab.clock.now
        )
        identity = {
            name: service[name]
            for name in (
                "camera_id",
                "assignment_id",
                "epoch",
                "supervisor_boot_id",
                "session_id",
            )
        }
        context.begin(dict(identity, node_id=node_id))
        contexts[node_id] = context
        return assignment, service

    def enqueue(node_id, kind="event"):
        context = contexts[node_id]
        service = lab.nodes[node_id]["report"]["service"]
        stamp = context.capture(service["session_id"])
        event = {"event_id": stamp["event_id"], "camera_id": "camera_1"}
        event.update(
            {"frames_inferred": 30} if kind == "observation" else {"type": "person"}
        )
        payload = dict(stamp, kind=kind, event=event)
        payload.pop("event_id")
        receipts.append(context.enqueue(payload))
        return payload

    def committed(node_id, payload):
        assert all(
            process.poll() is None for process in processes if process not in stopped
        )
        collector.check()
        row = (
            contexts[node_id]
            .db.execute(
                "SELECT state FROM node_outbox WHERE event_id=?",
                (payload["event"]["event_id"],),
            )
            .fetchone()
        )
        return row[0] == "committed"

    def activate(node_id, assignment, service):
        sample = enqueue(node_id, "observation")
        wait_for(lambda: committed(node_id, sample))
        activation = contexts[node_id].activate(
            service["session_id"],
            {
                "command_id": "activate-" + node_id,
                "expected_activation_id": None,
                "activation_id": assignment["activation_id"],
            },
        )
        service["activation"] = activation
        lab.store.activate(
            assignment["assignment_id"],
            {
                "request_id": "activate-" + node_id,
                "controller_boot_id": lab.store.boot_id,
                "session_id": service["session_id"],
                **activation,
            },
        )

    stopped = []
    try:
        a_assignment, a_service = setup_node("jetson-a")
        start_relay("jetson-a")
        activate("jetson-a", a_assignment, a_service)
        a_event = enqueue("jetson-a")
        wait_for(lambda: committed("jetson-a", a_event))
        a_receipt = lab.store.receipt("jetson-a", a_event["event"]["event_id"])
        assert lab.store.consume(a_receipt["event_id"], a_receipt["sha256"])["consumed"]

        b_assignment, b_service = setup_node("jetson-b")
        b_process = start_relay("jetson-b")
        activate("jetson-b", b_assignment, b_service)
        b_event, old_event = enqueue("jetson-b"), enqueue("jetson-a")
        wait_for(
            lambda: committed("jetson-b", b_event) and committed("jetson-a", old_event)
        )
        assert (
            lab.store.receipt("jetson-a", old_event["event"]["event_id"])["decision"]
            == "rejected"
        )
        assert (
            lab.store.receipt("jetson-b", b_event["event"]["event_id"])["decision"]
            == "accepted"
        )

        collector.close()
        backlog = [enqueue("jetson-b") for _ in range(45)]
        time.sleep(2)
        assert len(contexts["jetson-b"].pending(100)) == 45
        b_process.kill()
        b_process.wait(timeout=10)
        stopped.append(b_process)
        start_relay("jetson-b")
        collector = EdgeDdsCollector(lab.store, tokens, domain_id)
        collector.start()
        wait_for(
            lambda: all(committed("jetson-b", item) for item in backlog), timeout=30
        )

        # 저장 실패 때 DDS 수신이 성공해도 원본 큐의 완료 확인은 진행되지 않는다.
        lab.store.db.execute("PRAGMA query_only=ON")
        storage_event = enqueue("jetson-b")
        time.sleep(2)
        assert not committed("jetson-b", storage_event)
        lab.store.db.execute("PRAGMA query_only=OFF")
        wait_for(lambda: committed("jetson-b", storage_event))
        for node_id, context in contexts.items():
            for row in context.db.execute(
                "SELECT event_id,sha256,state FROM node_outbox"
            ):
                assert row["state"] == "committed"
                assert (
                    lab.store.receipt(node_id, row["event_id"])["sha256"]
                    == row["sha256"]
                )
        assert len(receipts) == 51
    finally:
        lab.store.db.execute("PRAGMA query_only=OFF")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        collector.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)
        for context in contexts.values():
            context.close()
        for file in files:
            file.close()


def test_dds_credentials_and_exact_topic_permissions_are_enforced(tmp_path_factory):
    keys = Path(os.environ["EDGE_DDS_TEST_KEYSTORES"])
    directory = tmp_path_factory.mktemp("dds-security")
    domain_id = int(os.environ.get("EDGE_DDS_TEST_DOMAIN", "73"))
    own_enclave = "/soma/" + node_component("jetson-a")
    own_keys = keys / node_component("jetson-a")
    script = """
import sys
from src.services.edge_result_dds import _Endpoint
endpoint = _Endpoint(int(sys.argv[1]), sys.argv[2])
try:
    endpoint.node.create_publisher(endpoint.message_type, sys.argv[3], endpoint.qos)
    print('PUBLISHER_READY', flush=True)
finally:
    endpoint.close()
"""

    def attempt(label, keystore, enclave, topic, domain=domain_id):
        result = subprocess.run(
            [sys.executable, "-c", script, str(domain), enclave, topic],
            env=dict(os.environ, ROS_SECURITY_KEYSTORE=str(keystore)),
            capture_output=True,
            text=True,
            timeout=15,
        )
        (directory / (label + ".log")).write_text(result.stdout + result.stderr)
        return result

    good = attempt("own-topic", own_keys, own_enclave, result_topic("jetson-a"))
    assert good.returncode == 0 and "PUBLISHER_READY" in good.stdout
    wrong_topic = attempt(
        "other-topic", own_keys, own_enclave, result_topic("jetson-b")
    )
    assert wrong_topic.returncode != 0 and "PUBLISHER_READY" not in wrong_topic.stdout
    assert "SECURITY" in wrong_topic.stdout
    for label, keystore, enclave, domain in [
        ("missing-key", directory / "missing", own_enclave, domain_id),
        ("other-enclave", own_keys, "/soma/" + node_component("jetson-b"), domain_id),
        ("other-domain", own_keys, own_enclave, domain_id + 1),
    ]:
        result = attempt(label, keystore, enclave, result_topic("jetson-a"), domain)
        assert result.returncode != 0 and "PUBLISHER_READY" not in result.stdout
    tampered = directory / "tampered"
    shutil.copytree(own_keys, tampered)
    permissions = tampered / "enclaves" / own_enclave.lstrip("/") / "permissions.p7s"
    original = permissions.read_bytes()
    altered = original.replace(
        f"<id>{domain_id}</id>".encode(), f"<id>{domain_id + 1}</id>".encode(), 1
    )
    assert altered != original
    permissions.write_bytes(altered)
    result = attempt(
        "tampered-signature", tampered, own_enclave, result_topic("jetson-a")
    )
    assert result.returncode != 0 and "PUBLISHER_READY" not in result.stdout
    for role in keys.iterdir():
        private_keys = [
            path for path in role.rglob("*") if path.name.endswith("key.pem")
        ]
        assert len(private_keys) == 1 and private_keys[0].name == "key.pem"


def test_dds_qos_security_and_payload_identity_on_actual_wire(
    lab, tmp_path_factory, monkeypatch
):
    keys = Path(os.environ["EDGE_DDS_TEST_KEYSTORES"])
    domain_id = int(os.environ.get("EDGE_DDS_TEST_DOMAIN", "73"))
    directory = tmp_path_factory.mktemp("dds-wire")
    monkeypatch.setenv("ROS_SECURITY_KEYSTORE", str(keys / "controller"))
    assignment, _ = prepare(lab, "jetson-a")
    observed(lab, assignment)
    payload = result(lab, assignment, 1, kind="observation", activation=False)
    event_id = payload["event"]["event_id"]
    body = directory / "body.json"
    body.write_text(json.dumps(payload))
    collector = EdgeDdsCollector(lab.store, lab.nodes, domain_id)
    script = """
import os,sys,time
from pathlib import Path
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from src.services.edge_result_dds import _Endpoint, node_component, result_topic
from std_msgs.msg import String
mode, domain, filename = sys.argv[1:]
endpoint = None
if mode == 'unsecured':
    os.environ['ROS_SECURITY_ENABLE'] = 'false'
    rclpy.init(args=[], domain_id=int(domain))
    node = rclpy.create_node('unsecured_probe', enable_rosout=False, start_parameter_services=False)
    qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
else:
    endpoint = _Endpoint(int(domain), '/soma/' + node_component('jetson-a'))
    node, qos = endpoint.node, endpoint.qos
    if mode == 'best_effort':
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
publisher = node.create_publisher(String, result_topic('jetson-a'), qos)
try:
    message = String(data=Path(filename).read_text())
    until = time.monotonic() + 4
    while time.monotonic() < until:
        publisher.publish(message)
        time.sleep(.1)
    print('MATCHED', publisher.get_subscription_count(), flush=True)
finally:
    if endpoint:
        endpoint.close()
    else:
        node.destroy_node()
        rclpy.shutdown()
"""

    def send(label):
        process = subprocess.run(
            [sys.executable, "-c", script, label, str(domain_id), str(body)],
            env=dict(
                os.environ, ROS_SECURITY_KEYSTORE=str(keys / node_component("jetson-a"))
            ),
            capture_output=True,
            text=True,
            timeout=15,
        )
        (directory / (label + ".log")).write_text(process.stdout + process.stderr)
        assert process.returncode == 0
        collector.check()
        return process.stdout

    collector.start()
    try:
        for mode in ("unsecured", "best_effort"):
            assert "MATCHED 0" in send(mode)
            assert (
                lab.store.db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
            )
        assert "MATCHED 1" in send("reliable")
        wait_for(
            lambda: (
                lab.store.db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1
            )
        )
        assert lab.store.receipt("jetson-a", event_id)["decision"] == "observed"
        payload["edge"]["node_id"] = "jetson-b"
        payload["event"]["event_id"] = str(uuid.uuid4())
        body.write_text(json.dumps(payload))
        send("wrong-identity")

        def rejected_count():
            if not lab.store.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rejected_mqtt'"
            ).fetchone():
                return 0
            return lab.store.db.execute(
                "SELECT COUNT(*) FROM rejected_mqtt"
            ).fetchone()[0]

        wait_for(lambda: rejected_count() == 1)
        assert (
            lab.store.db.execute("SELECT reason FROM rejected_mqtt").fetchone()[0]
            == "node_identity_mismatch"
        )
        body.write_text("x" * 49153)
        send("oversize")
        wait_for(lambda: rejected_count() == 2)
        assert lab.store.db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1
    finally:
        collector.close()
