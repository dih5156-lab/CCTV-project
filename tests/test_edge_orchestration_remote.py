"""격리 HTTPS에서 원격 주소·두 노드·재시작·인증 경계를 검증한다."""

import ast
import copy
import inspect
import json
import secrets
import select
import socket
import sqlite3
import ssl
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from runners.run_edge_orchestration import private_file, read_token
from src.services.edge_node_agent import JetsonNodeAgent, RegistryClient
from src.services.edge_orchestrator import (
    NodeRegistry,
    RegistryError,
    create_registry_server,
)
from src.services.jetson_resources import JetsonResourceCollector


def test_agent_bundle_syntax_and_annotations_support_python38():
    # 문법 검사와 타입 표기 지연 평가를 함께 확인하며 실장비 실행 검증은 별도로 수행한다.
    modules = (
        "src/services/edge_node_agent.py",
        "src/services/jetson_resources.py",
        "runners/run_edge_orchestration.py",
    )
    root = Path(__file__).resolve().parents[1]
    for relative in modules:
        ast.parse((root / relative).read_text(), feature_version=(3, 8))
    for function in (
        RegistryClient.__init__,
        RegistryClient.request,
        JetsonNodeAgent.run,
        JetsonResourceCollector._cpu_percent,
        JetsonResourceCollector._memory_percent,
        JetsonResourceCollector._gpu_percent,
        private_file,
        read_token,
    ):
        annotations = inspect.get_annotations(function)
        assert annotations and all(
            isinstance(value, str) for value in annotations.values()
        )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import runners.run_edge_orchestration; "
            "assert 'src.services.edge_orchestrator' not in sys.modules; "
            "assert 'src._http_server' not in sys.modules",
        ],
        cwd=root,
        check=True,
        timeout=10,
    )


@pytest.fixture(scope="module")
def certificates(tmp_path_factory):
    directory = tmp_path_factory.mktemp("edge-tls")
    result = []
    for label in ("trusted", "other"):
        cert, key = directory / f"{label}.crt", directory / f"{label}.key"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "2",
                "-subj",
                f"/CN={label}",
                "-addext",
                "subjectAltName=IP:127.0.0.1",
                "-keyout",
                str(key),
                "-out",
                str(cert),
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
        key.chmod(0o600)
        result.append((cert, key))
    return result


@pytest.fixture
def remote_registry(tmp_path, certificates):
    cert, key = certificates[0]
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    clock = SimpleNamespace(now=time.time())
    registry = NodeRegistry(
        tmp_path / "nodes.sqlite3",
        clock=lambda: clock.now,
        allow_real_resources=True,
        allow_remote_nodes=True,
    )
    tokens = {node: secrets.token_urlsafe(32) for node in ("jetson-a", "jetson-b")}
    reader_token = secrets.token_urlsafe(32)
    servers = []

    def start(port=0):
        server = create_registry_server(
            registry, tokens, reader_token, port=port, tls_context=context
        )
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        servers.append((server, thread))
        return server, thread

    server, _ = start()
    url = f"https://127.0.0.1:{server.server_port}"
    yield SimpleNamespace(
        store=registry,
        tokens=tokens,
        clock=clock,
        url=url,
        cert=cert,
        reader=RegistryClient(url, reader_token, ca_file=cert),
        start=start,
        servers=servers,
    )
    for server, thread in servers:
        if thread.is_alive():
            server.shutdown()
            thread.join(5)
        server.server_close()
        assert not thread.is_alive()


def make_agent(remote, node_id="jetson-a", cpu=25, ip="192.0.2.11"):
    collector = Mock()
    collector.sample.return_value = {
        "cpu_percent": cpu,
        "gpu_percent": cpu + 1,
        "memory_percent": cpu + 2,
        "network_rx_bytes_per_second": None,
        "network_tx_bytes_per_second": None,
    }
    return JetsonNodeAgent(
        RegistryClient(remote.url, remote.tokens[node_id], ca_file=remote.cert),
        node_id,
        interval=0.02,
        collector=collector,
        node_ip=ip,
    )


def test_two_nodes_report_independent_values_and_liveness_over_tls(remote_registry):
    remote = remote_registry
    agents = [make_agent(remote), make_agent(remote, "jetson-b", 60, "192.0.2.12")]
    stop = threading.Event()
    workers = [
        threading.Thread(target=a.run, args=(stop,), kwargs={"report_limit": 3})
        for a in agents
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(8)
        assert not any(worker.is_alive() for worker in workers)
        code, snapshot = remote.reader.request("GET", "/v1/nodes")
        nodes = snapshot["nodes"]
        assert code == 200 and snapshot["simulation"] is False
        assert [n["ip"] for n in nodes] == ["192.0.2.11", "192.0.2.12"]
        assert [n["report"]["resources"]["cpu_percent"] for n in nodes] == [25, 60]
        assert [n["report_count"] for n in nodes] == [3, 3]
        assert all(n["report"]["service"] is None for n in nodes)
        remote.clock.now += 11
        assert all(n["freshness"] == "stale" for n in remote.store.list_nodes())
        original = agents[1]._sample

        def current_sample():
            sample = original()
            sample["sampled_at"] = datetime.fromtimestamp(
                remote.clock.now, timezone.utc
            ).isoformat()
            return sample

        with patch.object(agents[1], "_sample", side_effect=current_sample):
            assert agents[1].report_once()
        assert [
            n["freshness"]
            for n in remote.reader.request("GET", "/v1/nodes")[1]["nodes"]
        ] == ["stale", "online"]
    finally:
        stop.set()
        for worker in workers:
            worker.join(5)


def test_tls_checks_ca_and_server_identity(remote_registry, certificates):
    remote = remote_registry
    for url, ca in [
        (remote.url, certificates[1][0]),
        (remote.url.replace("127.0.0.1", "localhost"), remote.cert),
    ]:
        client = RegistryClient(url, remote.tokens["jetson-a"], ca_file=ca)
        with pytest.raises(ssl.SSLCertVerificationError):
            client.request("GET", "/v1/nodes")
    assert remote.store.list_nodes() == []


def test_tls_node_credentials_cannot_impersonate_or_read(remote_registry):
    remote = remote_registry
    agent = make_agent(remote)
    agent.node_id = "jetson-b"
    assert not agent.report_once() and agent.last_error == "unauthorized"
    assert agent.client.request("GET", "/v1/nodes")[0] == 401
    assert remote.store.list_nodes() == []


def test_idle_tls_connection_does_not_block_another_node(remote_registry):
    port = remote_registry.servers[0][0].server_port
    with socket.create_connection(("127.0.0.1", port), timeout=3):
        started = time.monotonic()
        agent = make_agent(remote_registry)
        assert agent.report_once(), agent.last_error
        assert time.monotonic() - started < 2


def test_status_recovers_after_temporary_write_lock(remote_registry, caplog):
    remote = remote_registry
    assert make_agent(remote).report_once()
    writer = sqlite3.connect(remote.store.path, check_same_thread=False)
    unlocked = threading.Event()

    def release():
        writer.rollback()
        unlocked.set()

    timer = threading.Timer(2.2, release)
    try:
        writer.execute("BEGIN EXCLUSIVE")
        writer.execute("UPDATE nodes SET last_sequence=99")
        remote.clock.now += 11
        timer.start()
        code, body = remote.reader.request("GET", "/v1/nodes")
        assert code == 200, (code, body)
        assert unlocked.is_set()
        node = body["nodes"][0]
        assert node["last_sequence"] == node["report"]["sequence"] == 1
        assert node["freshness"] == "stale"
        assert "database_locked" in caplog.text
        assert "retrying storage read once" in caplog.text
    finally:
        timer.join(5)
        writer.rollback()
        writer.close()
    with remote.store._connection() as connection:
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert remote.store.list_nodes()[0]["last_sequence"] == 1


@pytest.mark.parametrize("method,code,error,attempts", [
    ("GET", 503, "storage_unavailable", 2),
    ("POST", 503, "storage_unavailable", 1),
    ("PUT", 503, "storage_unavailable", 1),
    ("GET", 503, "other_failure", 1),
    ("GET", 401, "unauthorized", 1),
])
def test_storage_retry_is_bounded_and_only_for_reads(method, code, error, attempts):
    client = RegistryClient("http://127.0.0.1:12345", "unused")
    with patch("src.services.edge_node_agent.http.client.HTTPConnection") as connect:
        response = connect.return_value.getresponse.return_value
        response.status = code
        response.read.return_value = json.dumps({"error": error}).encode()
        assert client.request(method, "/v1/nodes") == (code, {"error": error})
        assert connect.call_count == attempts
        assert connect.return_value.close.call_count == attempts


def test_tls_disconnect_restart_and_boot_fencing(remote_registry):
    remote = remote_registry
    agent = make_agent(remote)
    assert agent.report_once()
    old_server, old_thread = remote.servers[0]
    port = old_server.server_port
    old_server.shutdown()
    old_thread.join(5)
    old_server.server_close()
    assert not agent.report_once()
    pending = copy.deepcopy(agent.pending)
    # 같은 DB/포트로 제어기만 복구해도 노드는 같은 부팅과 표본으로 재시도한다.
    remote.start(port)
    assert agent.report_once() and agent.sequence == 2
    assert remote.store.list_nodes()[0]["report"] == pending
    new_agent = make_agent(remote)
    assert new_agent.report_once()
    assert not agent.report_once() and agent.last_error == "retired_boot"
    assert (
        NodeRegistry(remote.store.path).list_nodes()[0]["boot_id"] == new_agent.boot_id
    )


def test_tls_lost_ack_keeps_sample_and_sequence(remote_registry):
    agent = make_agent(remote_registry)
    original = agent.client.request
    dropped = []

    def drop_ack(method, path, payload=None):
        result = original(method, path, payload)
        if path.endswith("/reports") and not dropped:
            dropped.append(copy.deepcopy(payload))
            raise OSError("lost ACK")
        return result

    with patch.object(agent.client, "request", side_effect=drop_ack):
        assert not agent.report_once()
        assert agent.pending == dropped[0]
        assert agent.report_once()
    assert agent.collector.sample.call_count == 1
    assert remote_registry.store.list_nodes()[0]["report_count"] == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://192.0.2.10:18443",
        "https://127.0.0.1:18443",
        "http://127.0.0.1:80\n",
        "https://user:secret@host:443",
        "http://127.0.0.1:80?token=x",
    ],
)
def test_remote_insecure_or_ambiguous_urls_are_rejected(url):
    with pytest.raises(ValueError):
        RegistryClient(url, "unused")


@pytest.mark.parametrize(
    "ip", ["0.0.0.0", "224.0.0.1", "255.255.255.255", "::1", "not-an-ip", 2130706433]
)
def test_remote_registration_rejects_unusable_addresses(remote_registry, ip):
    agent = make_agent(remote_registry)
    assert agent.report_once()
    registration = {
        key: remote_registry.store.list_nodes()[0][key]
        for key in (
            "node_id",
            "node_type",
            "boot_id",
            "ip",
            "capabilities",
            "simulation",
        )
    }
    registration["ip"] = ip
    with pytest.raises(RegistryError, match="invalid_node_ip"):
        remote_registry.store.register(registration)


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "224.0.0.1", "255.255.255.255", "192.0.2.10"]
)
def test_nonlocal_listener_cannot_open_without_tls(tmp_path, host):
    store = NodeRegistry(tmp_path / "nodes.sqlite3", allow_remote_nodes=True)
    with pytest.raises(ValueError):
        create_registry_server(
            store,
            {"jetson-a": secrets.token_urlsafe(32)},
            secrets.token_urlsafe(32),
            host=host,
        )


def test_private_tokens_are_not_hardcoded_or_world_readable(tmp_path):
    token_file = tmp_path / "agent.token"
    token = secrets.token_urlsafe(32)
    token_file.write_text(token + "\n")
    token_file.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        read_token(token_file)
    token_file.chmod(0o600)
    assert read_token(token_file) == token
    token_file.write_text("x" * 32 + "\r\nInjected: header")
    with pytest.raises(ValueError, match="URL-safe"):
        read_token(token_file)
    with pytest.raises(ValueError):
        private_file(tmp_path)


def test_controller_start_does_not_depend_on_reverse_dns(tmp_path):
    store = NodeRegistry(tmp_path / "nodes.sqlite3")
    with patch("socket.getfqdn", side_effect=AssertionError("DNS must not run")):
        with create_registry_server(
            store, {"jetson-a": secrets.token_urlsafe(32)}, secrets.token_urlsafe(32)
        ) as server:
            assert server.server_name == "127.0.0.1" and server.server_port > 0


@pytest.mark.parametrize("ownership", [False, True])
def test_three_cli_processes_report_and_stop_without_operating_services(
    tmp_path, certificates, ownership
):
    cert, key = certificates[0]
    tokens = {node: secrets.token_urlsafe(32) for node in ("jetson-a", "jetson-b")}
    reader = secrets.token_urlsafe(32)
    credentials = tmp_path / "controller.json"
    data = {"node_tokens": tokens, "reader_token": reader}
    if ownership:
        data.update(control_token=secrets.token_urlsafe(32), consumer_token=secrets.token_urlsafe(32))
    credentials.write_text(json.dumps(data))
    credentials.chmod(0o600)
    files = {}
    for name, token in {**tokens, "reader": reader}.items():
        path = tmp_path / f"{name}.token"
        path.write_text(token)
        path.chmod(0o600)
        files[name] = path
    prefix = [sys.executable, "-m", "runners.run_edge_orchestration"]
    process = subprocess.Popen(
        prefix
        + [
            "controller",
            "--bind",
            "127.0.0.1",
            "--state-dir",
            str(tmp_path / "state"),
            "--credentials",
            str(credentials),
            "--cert",
            str(cert),
            "--key",
            str(key),
        ] + (["--enable-ownership"] if ownership else []),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    agents = []
    try:
        assert select.select([process.stdout], [], [], 10)[0], "controller not ready"
        endpoint = json.loads(process.stdout.readline())
        url = endpoint["controller"]
        if ownership:
            client = RegistryClient(url, reader, ca_file=cert)
            code, camera = client.request("GET", "/v1/cameras/camera_1")
            assert code == 200 and camera["controller_boot_id"] == endpoint["controller_boot_id"]
            assert camera["active"] is None and not camera["accepting_results"]
            assert client.request("GET", "/v1/results/pending")[0] == 401
        for node_id in tokens:
            agents.append(
                subprocess.Popen(
                    prefix
                    + [
                        "agent",
                        "--controller",
                        url,
                        "--ca-file",
                        str(cert),
                        "--node-id",
                        node_id,
                        "--node-ip",
                        "127.0.0.1",
                        "--token-file",
                        str(files[node_id]),
                        "--interval",
                        "0.05",
                        "--reports",
                        "3",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for agent in agents:
            stdout, stderr = agent.communicate(timeout=15)
            assert agent.returncode == 0, stderr
            assert json.loads(stdout)["last_sequence"] == 3
        query = subprocess.run(
            prefix
            + [
                "status",
                "--controller",
                url,
                "--ca-file",
                str(cert),
                "--token-file",
                str(files["reader"]),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        snapshot = json.loads(query.stdout)
        assert [n["report_count"] for n in snapshot["nodes"]] == [3, 3]
        assert all(
            n["freshness"] == "online" and n["simulation"] is False
            for n in snapshot["nodes"]
        )
        assert all(
            n["report"]["resources"]["cpu_percent"] is not None
            for n in snapshot["nodes"]
        )
        assert reader not in query.stdout and not any(
            token in query.stdout for token in tokens.values()
        )
    finally:
        for agent in agents:
            if agent.poll() is None:
                agent.terminate()
                agent.communicate(timeout=5)
        process.terminate()
        _, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    assert [
        n["report_count"]
        for n in NodeRegistry(tmp_path / "state/nodes.sqlite3").list_nodes()
    ] == [3, 3]
