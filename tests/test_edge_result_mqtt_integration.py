"""운영과 분리한 실제 Mosquitto·TLS 제어기로 결과 전달과 ACL을 검사한다."""

import json
import os
import secrets
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.protocols._mqtt_factory import create_mqtt_client
from src.services.edge_controller_api import create_controller_server
from src.services.edge_node_agent import RegistryClient
from src.services.edge_orchestrator import RegistryError
from src.services.edge_result_transport import (
    EdgeResultCollector,
    EdgeResultRelay,
    mqtt_settings,
    raw_topic,
)
from tests.test_edge_orchestration_remote import certificates as certificates
from tests.test_edge_ownership import lab as lab
from tests.test_edge_result_transport import activate_node, sample, setup_node
from tests.test_mqtt_durable_integration import wait_until


@pytest.fixture(scope="module")
def isolated_mqtt(certificates):
    if os.environ.get("CCTV_TEST_EDGE_MQTT_DOCKER") != "1":
        pytest.skip("explicit disposable Docker MQTT test opt-in required")
    name = "cctv-edge-results-test-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="cctv-edge-results-test-") as directory:
        root = Path(directory)
        users = {
            name: secrets.token_urlsafe(24)
            for name in ("jetson-a", "jetson-b", "collector", "observer")
        }
        password_file = root / "passwords"
        password_file.write_text(
            "".join(f"{name}:{password}\n" for name, password in users.items())
        )
        password_file.chmod(0o600)
        cert, key = certificates[0]
        for source, filename in ((cert, "tls.crt"), (key, "tls.key")):
            destination = root / filename
            destination.write_bytes(source.read_bytes())
            destination.chmod(0o600)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        (root / "acl").write_text(
            "pattern write poc/edge/raw/%u/+/+/+\nuser collector\ntopic read poc/edge/raw/+/+/+/+\nuser observer\ntopic read #\n"
        )
        # 브로커 상태가 소실돼도 노드 원본과 중앙 ACK로 복구되는지 검증한다.
        (root / "mosquitto.conf").write_text(
            f"listener {port} 127.0.0.1\nallow_anonymous false\npassword_file /lab/passwords\n"
            "certfile /lab/tls.crt\nkeyfile /lab/tls.key\ntls_version tlsv1.2\n"
            "acl_file /lab/acl\npersistence false\nlog_type error\n"
        )
        common = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=8m",
            "--mount",
            f"type=bind,source={root},target=/lab",
        ]
        prepared = subprocess.run(
            common
            + [
                "--entrypoint",
                "mosquitto_passwd",
                "eclipse-mosquitto:2.0",
                "-U",
                "/lab/passwords",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if prepared.returncode:
            diagnostic = prepared.stderr
            for secret in users.values():
                diagnostic = diagnostic.replace(secret, "[redacted]")
            pytest.fail(diagnostic)
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "cctv.purpose=edge-results-test",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--memory",
            "64m",
            "--cpus",
            "0.5",
            "--pids-limit",
            "64",
            "--network",
            "host",
            "--mount",
            f"type=bind,source={root},target=/lab",
            "--entrypoint",
            "mosquitto",
            "eclipse-mosquitto:2.0",
            "-c",
            "/lab/mosquitto.conf",
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=30)
            info = json.loads(
                subprocess.run(
                    ["docker", "inspect", name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
            )[0]
            assert info["HostConfig"]["NetworkMode"] == "host"

            def settings(user):
                return {
                    "host": "127.0.0.1",
                    "port": port,
                    "ca_file": str(root / "tls.crt"),
                    "username": user,
                    "password": users[user],
                }

            yield SimpleNamespace(settings=settings, name=name)
        finally:
            try:
                state = subprocess.run(
                    ["docker", "inspect", "--format", "{{json .State}}", name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if state.returncode == 0:
                    details = json.loads(state.stdout)
                    print(
                        "isolated broker final state",
                        {
                            key: details[key]
                            for key in ("Status", "OOMKilled", "ExitCode", "Error")
                        },
                    )
                    process = Path("/proc") / str(details["Pid"])
                    for leaf in ("wchan", "stat"):
                        try:
                            print(
                                "broker process",
                                leaf,
                                (process / leaf).read_text().strip(),
                            )
                        except OSError:
                            pass
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "30", name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                diagnostic = logs.stdout + logs.stderr
                for secret in users.values():
                    diagnostic = diagnostic.replace(secret, "[redacted]")
                print("isolated broker diagnostics", diagnostic)
            finally:
                # 진단 조회가 지연돼도 자신이 만든 브로커 정리는 반드시 시도한다.
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    capture_output=True,
                    timeout=30,
                    check=True,
                )


@pytest.fixture
def environment(lab, isolated_mqtt, certificates):
    cert, key = certificates[0]
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    lab.registry.allow_remote_nodes = True
    tokens = {node: secrets.token_urlsafe(32) for node in lab.nodes}
    reader, control, consumer = [secrets.token_urlsafe(32) for _ in range(3)]
    server = create_controller_server(
        lab.registry,
        lab.store,
        tokens,
        reader,
        control,
        consumer,
        host="127.0.0.1",
        tls_context=tls,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    collector = EdgeResultCollector(
        lab.store,
        lab.nodes,
        {
            "mqtt": isolated_mqtt.settings("collector"),
            "client_id": "collector-" + uuid.uuid4().hex,
        },
    )
    collector.start()
    relays, contexts = [], []

    def node(node_id, tmp_path):
        assignment, context, service = setup_node(lab, tmp_path, node_id)
        contexts.append(context)
        client = RegistryClient(
            f"https://127.0.0.1:{server.server_port}", tokens[node_id], ca_file=cert
        )
        relay = EdgeResultRelay(
            context, node_id, client, isolated_mqtt.settings(node_id)
        )
        relays.append(relay)
        return assignment, context, service, relay

    try:
        wait_until(lambda: collector.subscriber.is_connected)
        yield SimpleNamespace(
            lab=lab, collector=collector, node=node, broker=isolated_mqtt
        )
    finally:

        def tcp_state(client):
            sock = client.socket() if client is not None else None
            if sock is None:
                return {"socket": None}
            data = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
            return {
                "mqtt_state": str(client._state),
                "tcp_state": data[0],
                "unacked": struct.unpack_from("I", data, 24)[0],
                "retransmits": struct.unpack_from("I", data, 100)[0],
            }

        print(
            "collector state",
            collector.subscriber.is_connected,
            collector.subscriber._restart_requested,
            collector.subscriber._pending_subscriptions,
            tcp_state(collector.subscriber._client),
        )
        for relay in relays:
            with relay.context.lock:
                print(
                    "relay state",
                    relay.node_id,
                    relay.client.is_connected(),
                    tcp_state(relay.client),
                    [
                        tuple(row)
                        for row in relay.context.db.execute(
                            "SELECT state,last_error,COUNT(*) FROM node_outbox GROUP BY state,last_error"
                        )
                    ],
                )
        with lab.store.lock:
            print(
                "central results",
                [
                    tuple(row)
                    for row in lab.store.db.execute(
                        "SELECT node_id,decision,reason,COUNT(*) FROM results GROUP BY node_id,decision,reason"
                    )
                ],
            )
        for relay in relays:
            relay.close()
        collector.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        for context in contexts:
            context.close()
        assert not thread.is_alive()


def receipt(store, node, payload):
    try:
        return store.receipt(node, payload["event"]["event_id"])
    except RegistryError as error:
        if error.status == 404:
            return None
        raise


def test_real_mqtt_tls_handover_and_late_previous_owner(environment, tmp_path):
    env = environment
    a, a_context, a_service, a_relay = env.node("jetson-a", tmp_path)
    activate_node(env.lab, a, a_context, a_service, env.collector)
    a_relay.start()
    payload_a = sample(a_context, a_service["session_id"])
    wait_until(lambda: not a_context.pending())
    ack_a = receipt(env.lab.store, "jetson-a", payload_a)
    assert ack_a["decision"] == "accepted"
    b, b_context, b_service, b_relay = env.node("jetson-b", tmp_path)
    prepared_b = sample(b_context, b_service["session_id"])
    b_relay.start()
    wait_until(lambda: receipt(env.lab.store, "jetson-b", prepared_b))
    assert receipt(env.lab.store, "jetson-b", prepared_b)["decision"] == "rejected"
    activate_node(env.lab, b, b_context, b_service, env.collector)
    late_a = sample(a_context, a_service["session_id"])
    active_b = sample(b_context, b_service["session_id"])
    wait_until(lambda: not a_context.pending() and not b_context.pending())
    assert receipt(env.lab.store, "jetson-a", late_a)["reason"] == "not_current_owner"
    ack_b = receipt(env.lab.store, "jetson-b", active_b)
    assert ack_b["decision"] == "accepted"
    assert not env.lab.store.consume(ack_a["event_id"], ack_a["sha256"])["consumed"]
    assert env.lab.store.consume(ack_b["event_id"], ack_b["sha256"])["consumed"]
    assert env.lab.store.consume(ack_b["event_id"], ack_b["sha256"])["duplicate"]
    assert env.lab.store.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_real_broker_acl_blocks_impersonation_and_operating_topics(
    environment, tmp_path
):
    env = environment
    assignment, context, service, relay = env.node("jetson-a", tmp_path)
    activate_node(env.lab, assignment, context, service, env.collector)
    payload = sample(context, service["session_id"])
    seen = []
    settings = env.broker.settings("observer")
    observer = create_mqtt_client(
        "edge-test-observer",
        username=settings["username"],
        password=settings["password"],
    )
    observer.tls_set_context(mqtt_settings(settings))
    subscribed = threading.Event()
    observer.on_connect = lambda client, *args: client.subscribe("#", qos=1)
    observer.on_subscribe = lambda *args: subscribed.set()
    observer.on_message = lambda client, userdata, message: seen.append(message.topic)
    observer.connect("127.0.0.1", settings["port"])
    observer.loop_start()
    relay.start()
    try:
        assert subscribed.wait(5)
        wait_until(lambda: relay.client.is_connected())
        for topic in (
            "cctv/events/camera_1/fall",
            raw_topic(payload).replace("/jetson-a/", "/jetson-b/"),
        ):
            published = relay.client.publish(topic, json.dumps(payload), qos=1)
            published.wait_for_publish(timeout=5)
            assert published.is_published()
        barrier = sample(context, service["session_id"])
        wait_until(lambda: receipt(env.lab.store, "jetson-a", barrier))
        wait_until(lambda: raw_topic(barrier) in seen)
        assert seen and all(
            topic.startswith("poc/edge/raw/jetson-a/") for topic in seen
        )
    finally:
        observer.disconnect()
        observer.loop_stop()


def test_real_mqtt_storage_failure_then_retry_with_same_body(environment, tmp_path):
    env = environment
    assignment, context, service, relay = env.node("jetson-a", tmp_path)
    activate_node(env.lab, assignment, context, service, env.collector)
    env.lab.store.db.execute(
        "CREATE TRIGGER fail_test_result BEFORE INSERT ON results BEGIN SELECT RAISE(ABORT, 'isolated test disk failure'); END"
    )
    payload = sample(context, service["session_id"])
    original = context.pending()[0]
    relay.start()
    try:
        wait_until(lambda: env.collector.subscriber._restart_requested)
        assert receipt(env.lab.store, "jetson-a", payload) is None
        assert context.pending()[0] == original
    finally:
        env.lab.store.db.execute("DROP TRIGGER fail_test_result")
    wait_until(lambda: not context.pending())
    ack = receipt(env.lab.store, "jetson-a", payload)
    assert ack["sha256"] == original["sha256"]
    assert (
        env.lab.store.db.execute(
            "SELECT COUNT(*) FROM results WHERE event_id=?", (ack["event_id"],)
        ).fetchone()[0]
        == 1
    )


def test_broker_loss_keeps_frozen_queue_until_tls_storage_ack(environment, tmp_path):
    env = environment
    assignment, context, service, relay = env.node("jetson-a", tmp_path)
    activate_node(env.lab, assignment, context, service, env.collector)
    relay.start()
    wait_until(lambda: relay.client.is_connected())
    subprocess.run(
        ["docker", "stop", "--time", "3", env.broker.name],
        check=True,
        capture_output=True,
        timeout=10,
    )
    try:
        payload = sample(context, service["session_id"])
        original = context.pending()[0]
        relay.step()
        assert context.pending()[0] == original
        assert receipt(env.lab.store, "jetson-a", payload) is None
    finally:
        subprocess.run(
            ["docker", "start", env.broker.name],
            check=True,
            capture_output=True,
            timeout=10,
        )
    wait_until(lambda: not context.pending(), timeout=20)
    ack = receipt(env.lab.store, "jetson-a", payload)
    assert ack["sha256"] == original["sha256"]
    assert ack["decision"] == "accepted"


def test_silent_tls_peer_does_not_hold_collector_shutdown(certificates):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(3)
    config = {
        "mqtt": {
            "host": "127.0.0.1",
            "port": listener.getsockname()[1],
            "ca_file": str(certificates[0][0]),
            "username": "collector",
            "password": "fixture-only",
        },
        "client_id": "silent-peer-test",
    }
    collector = EdgeResultCollector(SimpleNamespace(), ["jetson-a"], config)
    peer = None
    try:
        collector.start()
        peer, _ = listener.accept()
        started = time.monotonic()
        collector.close()
        assert time.monotonic() - started < 8
        assert not collector.thread.is_alive()
    finally:
        if peer is not None:
            peer.close()
        listener.close()
        collector.close()
