"""소유 프로세스·명령 영속성·실측 지표 경계의 격리 시험."""

import ast
import hashlib
import json
import os
import select
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services.edge_node_agent import JetsonNodeAgent, RegistryClient
from src.services.edge_orchestrator import (
    NodeRegistry,
    RegistryError,
    create_registry_server,
    validate_service_status,
)
from src.services.edge_service import (
    EdgeServiceSupervisor,
    ServiceManifest,
    create_service_server,
)


def private_json(path, data):
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    return path


def process_identity(pid):
    # 고아 자식 정리 때 PID가 재사용됐더라도 다른 프로세스에는 신호를 보내지 않는다.
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


@pytest.fixture
def setup(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    model = tmp_path / "model.engine"
    model.write_bytes(b"isolated model fixture, not a real TensorRT engine")
    model.chmod(0o600)
    environment = private_json(
        tmp_path / "env.json", {"FIXTURE_MODE_PATH": str(tmp_path / "mode")}
    )
    data = {
        "manifest_id": "lab-v1",
        "service_id": "cctv-ai",
        "camera_id": "camera_1",
        "argv": [
            sys.executable,
            str(Path(__file__).parent / "fixtures" / "edge_ai_process.py"),
        ],
        "cwd": str(tmp_path),
        "environment_file": str(environment),
        "metrics_url": f"http://127.0.0.1:{port}/metrics",
        "ready_timeout_seconds": 3,
        "stop_timeout_seconds": 0.2,
        "stale_after_seconds": 0.6,
        "models": [
            {
                "model_id": name,
                "metric_name": name,
                "path": str(model),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "allow_empty": name == "helmet",
            }
            for name in ("primary", "helmet")
        ],
    }
    manifest = private_json(tmp_path / "manifest.json", data)
    return tmp_path, manifest, data


@pytest.fixture
def supervisor(setup):
    root, path, _ = setup
    instance = EdgeServiceSupervisor("jetson-b", ServiceManifest(path), root / "state")
    try:
        yield instance
    finally:
        instance.close()


def command(supervisor, *, epoch=1, state="running", command_id=None):
    return {
        "command_id": command_id or f"command-{epoch}-{state}",
        "assignment_id": f"owner-{epoch}",
        "epoch": epoch,
        "desired_state": state,
        "manifest_id": "lab-v1",
        "expected_boot_id": supervisor.boot_id,
    }


def ready(supervisor):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        supervisor.tick()
        if supervisor.snapshot()["service"]["ready"]:
            return
        if supervisor.snapshot()["service"]["status"] == "error":
            pytest.fail(str(supervisor.snapshot()))
        time.sleep(0.05)
    pytest.fail(str(supervisor.snapshot()))


def test_lifecycle_receipt_is_not_readiness_and_stop_restart(supervisor):
    first = command(supervisor)
    ack = supervisor.submit("camera_1", first)
    assert ack["accepted"] is True
    assert supervisor.process is None and not supervisor.snapshot()["service"]["ready"]
    ready(supervisor)
    child = supervisor.process
    service = supervisor.snapshot()["service"]
    assert service["backend"] == "deepstream" and service["inference_fps"] > 0
    assert service["models_verified"] and len(service["models"]) == 2
    assert supervisor.submit("camera_1", first) == ack
    supervisor.tick()
    assert supervisor.process is child
    supervisor.submit("camera_1", command(supervisor, state="stopped"))
    assert not supervisor.snapshot()["service"]["ready"]
    supervisor.tick()
    assert (
        child.poll() is not None
        and supervisor.snapshot()["service"]["status"] == "stopped"
    )
    supervisor.submit("camera_1", command(supervisor, epoch=2))
    ready(supervisor)
    assert supervisor.process.pid != child.pid
    assert supervisor.session_id != service["session_id"]


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"epoch": 0}, "invalid_epoch"),
        ({"epoch": True}, "invalid_epoch"),
        ({"expected_boot_id": "old-boot"}, "stale_supervisor_boot"),
        ({"manifest_id": "other"}, "unknown_manifest"),
        ({"argv": ["kill"]}, "invalid_fields"),
    ],
)
def test_command_validation(supervisor, changes, code):
    value = command(supervisor)
    value.update(changes)
    with pytest.raises(RegistryError, match=code):
        supervisor.submit("camera_1", value)
    assert supervisor.process is None


def test_command_conflicts_fence_delayed_start(supervisor):
    first = command(supervisor)
    supervisor.submit("camera_1", first)
    with pytest.raises(RegistryError, match="command_id_conflict"):
        supervisor.submit("camera_1", dict(first, desired_state="stopped"))
    with pytest.raises(RegistryError, match="assignment_conflict"):
        supervisor.submit("camera_1", dict(first, command_id="different-id"))
    supervisor.submit("camera_1", command(supervisor, epoch=2, state="stopped"))
    with pytest.raises(RegistryError, match="stale_epoch"):
        supervisor.submit("camera_1", dict(first, command_id="late"))
    with pytest.raises(RegistryError, match="assignment_conflict"):
        supervisor.submit("camera_1", command(supervisor, epoch=2))


def test_restart_preserves_epoch_and_never_replays_start(setup):
    root, path, _ = setup
    first = EdgeServiceSupervisor("jetson-b", ServiceManifest(path), root / "state")
    first.submit("camera_1", command(first, epoch=4))
    ready(first)
    child = first.process
    first.close()
    second = EdgeServiceSupervisor("jetson-b", ServiceManifest(path), root / "state")
    try:
        second.tick()
        assert child.poll() is not None and second.process is None
        assert second.snapshot()["service"]["epoch"] == 4
        with pytest.raises(RegistryError, match="stale_epoch"):
            second.submit(
                "camera_1", command(second, epoch=4, command_id="new-after-reboot")
            )
        second.submit("camera_1", command(second, epoch=5))
        ready(second)
    finally:
        second.close()


@pytest.mark.parametrize("mode", ["cpu", "wrong-session", "stalled", "duplicate"])
def test_wrong_backend_identity_stall_and_malformed_metrics_fail_closed(
    supervisor, setup, mode
):
    root, _, _ = setup
    supervisor.submit("camera_1", command(supervisor))
    ready(supervisor)
    (root / "mode").write_text(mode)
    supervisor.tick()
    assert not supervisor.snapshot()["service"]["ready"]
    (root / "mode").write_text("running")
    ready(supervisor)


def test_model_tampering_prevents_execution(supervisor, setup):
    root, _, _ = setup
    (root / "model.engine").write_bytes(b"changed")
    supervisor.submit("camera_1", command(supervisor))
    supervisor.tick()
    assert supervisor.process is None
    assert supervisor.snapshot()["service"]["last_error"] == "model_hash_mismatch"


def test_publicly_writable_model_is_not_executed(supervisor, setup):
    root, _, _ = setup
    (root / "model.engine").chmod(0o664)
    supervisor.submit("camera_1", command(supervisor))
    supervisor.tick()
    assert supervisor.process is None
    assert supervisor.snapshot()["service"]["last_error"] == "model_file_unsafe"


def test_readiness_timeout_stops_only_owned_child(supervisor, setup):
    root, _, _ = setup
    (root / "mode").write_text("wrong-session")
    supervisor.submit("camera_1", command(supervisor))
    supervisor.tick()
    child = supervisor.process
    supervisor.clock = lambda: (
        supervisor.started + 4 if supervisor.started is not None else 4
    )
    supervisor.tick()
    assert child.poll() is not None
    assert supervisor.snapshot()["service"]["last_error"] == "readiness_timeout"


@pytest.mark.parametrize("recover", [True, False])
def test_running_service_gets_a_new_recovery_window(supervisor, monkeypatch, recover):
    supervisor.submit("camera_1", command(supervisor))
    ready(supervisor)
    child, observe = supervisor.process, supervisor._observe
    now = [supervisor.started + 20]
    supervisor.clock = lambda: now[0]

    def unavailable():
        raise TimeoutError("injected local metrics timeout")

    monkeypatch.setattr(supervisor, "_observe", unavailable)
    supervisor.tick()
    assert child.poll() is None
    assert not supervisor.snapshot()["service"]["ready"]
    now[0] += 2.9
    supervisor.tick()
    assert child.poll() is None
    if recover:
        monkeypatch.setattr(supervisor, "_observe", observe)
        ready(supervisor)
        assert supervisor.process is child
        now[0] += 20
        monkeypatch.setattr(supervisor, "_observe", unavailable)
        supervisor.tick()
        assert child.poll() is None
        now[0] += 2.9
        supervisor.tick()
        assert child.poll() is None
    now[0] += 0.2
    supervisor.tick()
    assert child.poll() is not None and supervisor.process is None
    assert supervisor.snapshot()["service"]["last_error"] == "readiness_timeout"


def test_unexpected_exit_not_automatically_restarted(supervisor):
    supervisor.submit("camera_1", command(supervisor))
    ready(supervisor)
    supervisor.process.kill()
    supervisor.process.wait(timeout=3)
    supervisor.tick()
    supervisor.tick()
    assert supervisor.process is None
    assert supervisor.snapshot()["service"]["last_error"] == "process_exited"


def test_stop_kills_only_owned_process_even_when_term_ignored(setup):
    root, path, data = setup
    private_json(
        Path(data["environment_file"]),
        {"FIXTURE_MODE_PATH": str(root / "mode"), "FIXTURE_IGNORE_TERM": "1"},
    )
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    supervisor = EdgeServiceSupervisor(
        "jetson-b", ServiceManifest(path), root / "state"
    )
    try:
        supervisor.submit("camera_1", command(supervisor))
        ready(supervisor)
        child = supervisor.process
        supervisor.close()
        assert child.returncode == -signal.SIGKILL
        assert sentinel.poll() is None
    finally:
        supervisor.close()
        sentinel.terminate()
        sentinel.wait(timeout=3)


def test_single_owner_lock_and_persistent_thread(supervisor, setup):
    root, path, _ = setup
    with pytest.raises(BlockingIOError):
        EdgeServiceSupervisor("jetson-b", ServiceManifest(path), root / "state")
    failures = []

    def other_thread():
        try:
            supervisor.tick()
        except RuntimeError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join(timeout=3)
    assert len(failures) == 1


def test_http_read_control_tokens_separate_and_no_remote_shell(supervisor):
    control, reader = "c" * 40, "r" * 40
    with create_service_server(supervisor, control, reader) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        read_client = RegistryClient(base, reader)
        write_client = RegistryClient(base, control)
        try:
            assert read_client.request("GET", "/v1/node/status")[0] == 200
            assert (
                read_client.request(
                    "PUT", "/v1/assignments/camera_1", command(supervisor)
                )[0]
                == 401
            )
            assert (
                write_client.request(
                    "PUT", "/v1/assignments/camera_1", command(supervisor)
                )[0]
                == 200
            )
            assert write_client.request("GET", "/v1/node/status")[0] == 401
            assert (
                write_client.request(
                    "PUT",
                    "/v1/assignments/camera_1",
                    {"argv": ["touch", "/tmp/unwanted"]},
                )[0]
                == 400
            )
        finally:
            server.shutdown()
            thread.join(timeout=3)


def test_python38_syntax_and_no_ai_dependencies_required():
    for relative in (
        "src/services/edge_service.py",
        "src/services/_edge_process.py",
        "src/services/edge_orchestrator.py",
    ):
        ast.parse(Path(relative).read_text(), feature_version=(3, 8))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.services.edge_service; assert 'torch' not in sys.modules; assert 'gi' not in sys.modules",
        ],
        check=True,
        timeout=5,
    )


@pytest.mark.parametrize("ignore_term", [False, True])
def test_parent_crash_signals_owned_ai_and_inherited_lock_blocks_duplicate(
    setup, ignore_term
):
    root, path, data = setup
    environment = json.loads(Path(data["environment_file"]).read_text())
    environment["FIXTURE_IGNORE_TERM"] = "1" if ignore_term else "0"
    private_json(Path(data["environment_file"]), environment)
    runner = """
import json, sys, time
from src.services.edge_service import EdgeServiceSupervisor, ServiceManifest
s = EdgeServiceSupervisor("jetson-b", ServiceManifest(sys.argv[1]), sys.argv[2])
s.submit("camera_1", dict(command_id="crash-start", assignment_id="crash-owner", epoch=1,
    desired_state="running", manifest_id="lab-v1", expected_boot_id=s.boot_id))
deadline = time.monotonic() + 5
while not s.snapshot()["service"]["ready"]:
    s.tick()
    if time.monotonic() > deadline: raise RuntimeError(s.snapshot())
    time.sleep(.05)
print(json.dumps(dict(pid=s.process.pid)), flush=True)
while True: s.tick(); time.sleep(.1)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", runner, str(path), str(root / "state")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid = None
    child_identity = None
    try:
        assert select.select([parent.stdout], [], [], 8)[0]
        child_pid = json.loads(parent.stdout.readline())["pid"]
        child_identity = process_identity(child_pid)
        assert child_identity is not None
        parent.kill()
        parent.wait(timeout=3)
        if ignore_term:
            with pytest.raises(BlockingIOError):
                EdgeServiceSupervisor("jetson-b", ServiceManifest(path), root / "state")
            os.kill(child_pid, signal.SIGKILL)
        deadline = time.monotonic() + 4
        while True:
            try:
                second = EdgeServiceSupervisor(
                    "jetson-b", ServiceManifest(path), root / "state"
                )
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        try:
            second.tick()
            assert second.process is None
            assert second.snapshot()["service"]["status"] == "stopped"
        finally:
            second.close()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        if (
            child_pid is not None
            and child_identity is not None
            and process_identity(child_pid) == child_identity
        ):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.stdout.close()
        parent.stderr.close()


def test_agent_reports_service_and_restart_does_not_restart_ai(supervisor, tmp_path):
    controller = NodeRegistry(
        tmp_path / "nodes.sqlite3", allow_real_resources=True, allow_real_services=True
    )
    supervisor.submit("camera_1", command(supervisor))
    ready(supervisor)
    child = supervisor.process
    read_token, control_token, node_token = "r" * 40, "c" * 40, "n" * 40
    resources = {
        key: None
        for key in (
            "cpu_percent",
            "gpu_percent",
            "memory_percent",
            "network_rx_bytes_per_second",
            "network_tx_bytes_per_second",
        )
    }
    with (
        create_service_server(supervisor, control_token, read_token) as service_server,
        create_registry_server(
            controller, {"jetson-b": node_token}, read_token
        ) as registry_server,
    ):
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (service_server, registry_server)
        ]
        for thread in threads:
            thread.start()
        client = RegistryClient(
            f"http://127.0.0.1:{registry_server.server_port}", node_token
        )
        service_client = RegistryClient(
            f"http://127.0.0.1:{service_server.server_port}", read_token
        )

        def make_agent():
            return JetsonNodeAgent(
                client,
                "jetson-b",
                collector=SimpleNamespace(sample=lambda: resources),
                service_client=service_client,
            )

        try:
            agent = make_agent()
            assert agent.report_once()
            node = controller.list_nodes()[0]
            assert node["service_freshness"] == "online"
            assert node["report"]["service"]["session_id"] == supervisor.session_id
            assert node["report"]["service"]["ready"]
            reboot = make_agent()
            assert reboot.report_once() and reboot.boot_id != agent.boot_id
            assert supervisor.process is child
            service_server.shutdown()
            service_server.server_close()
            assert reboot.report_once()
            assert controller.list_nodes()[0]["report"]["service"] is None
        finally:
            service_server.shutdown()
            registry_server.shutdown()
            for thread in threads:
                thread.join(timeout=3)


@pytest.mark.parametrize(
    "change,code",
    [
        ({"ready": True}, "inconsistent_readiness"),
        ({"ready": 1}, "invalid_readiness"),
        ({"backend": "video_processor"}, "unsupported_backend"),
        ({"inference_fps": float("nan")}, "invalid_number"),
        ({"epoch": True}, "invalid_epoch"),
    ],
)
def test_registry_rejects_inconsistent_service_observations(supervisor, change, code):
    service = supervisor.snapshot()["service"]
    service.update(change)
    with pytest.raises(RegistryError, match=code):
        validate_service_status(service, time.time())


@pytest.mark.parametrize("response", [[], {}, {"node_id": "other"}])
def test_bad_observation_never_blocks_resource_report(response):
    observer = SimpleNamespace(request=lambda *_: (200, response))
    agent = JetsonNodeAgent(
        None,
        "jetson-b",
        service_client=observer,
        collector=SimpleNamespace(sample=lambda: {"cpu_percent": 5}),
    )
    sample = agent._sample()
    assert sample["service"] is None and sample["resources"]["cpu_percent"] == 5


def test_state_directory_rejects_other_manifest(setup):
    root, path, data = setup
    first = EdgeServiceSupervisor("jetson-b", ServiceManifest(path), root / "state")
    first.close()
    data["manifest_id"] = "different"
    private_json(path, data)
    with pytest.raises(ValueError, match="another node or manifest"):
        EdgeServiceSupervisor("jetson-b", ServiceManifest(path), root / "state")


def test_result_activation_binds_running_session_and_is_reportable(setup):
    from src.services.edge_result_context import EdgeResultContext

    root, path, _ = setup
    supervisor = EdgeServiceSupervisor(
        "jetson-b", ServiceManifest(path), root / "state", enable_results=True
    )
    producer = None
    try:
        start = command(supervisor)
        start["assignment_id"] = str(uuid.uuid4())
        supervisor.submit("camera_1", start)
        ready(supervisor)
        producer = EdgeResultContext(supervisor.result_context.path)
        queued = producer.capture(supervisor.session_id)
        activation = {
            "command_id": "activate-result",
            "expected_boot_id": supervisor.boot_id,
            "assignment_id": start["assignment_id"],
            "epoch": 1,
            "session_id": supervisor.session_id,
            "activation_id": str(uuid.uuid4()),
            "expected_activation_id": None,
        }
        ack = supervisor.activate_results("camera_1", activation)
        status = supervisor.snapshot()["service"]
        validate_service_status(status, time.time())
        assert status["activation"] == ack and ack["sequence_floor"] == 1
        with pytest.raises(RegistryError, match="invalid_epoch"):
            supervisor.activate_results("camera_1", dict(activation, epoch=True))
        assert queued["edge"]["activation_id"] is None
        assert (
            producer.capture(supervisor.session_id)["edge"]["activation_id"]
            == ack["activation_id"]
        )
        with pytest.raises(RegistryError, match="result_session_not_ready"):
            supervisor.activate_results(
                "camera_1", dict(activation, session_id="old-session")
            )
        session = supervisor.session_id
        supervisor.close()
        with pytest.raises(RegistryError, match="retired_result_session"):
            producer.capture(session)
    finally:
        if producer is not None:
            producer.close()
        supervisor.close()


def test_cli_tls_service_assignment_agent_and_shutdown(setup):
    root, path, _ = setup
    cert, key = root / "tls.crt", root / "tls.key"
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
            "/CN=edge-lab",
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
    reader, control, node_token = "r" * 40, "c" * 40, "n" * 40
    credentials = private_json(
        root / "credentials.json", {"reader_token": reader, "control_token": control}
    )
    for name, value in (("reader", reader), ("control", control), ("node", node_token)):
        token = root / f"{name}.token"
        token.write_text(value)
        token.chmod(0o600)
    cli = [sys.executable, "-m", "runners.run_edge_orchestration"]
    service = subprocess.Popen(
        [
            *cli,
            "service",
            "--node-id",
            "jetson-b",
            "--manifest",
            str(path),
            "--state-dir",
            str(root / "state"),
            "--credentials",
            str(credentials),
            "--cert",
            str(cert),
            "--key",
            str(key),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    registry = NodeRegistry(
        root / "nodes.sqlite3", allow_real_resources=True, allow_real_services=True
    )
    with create_registry_server(
        registry, {"jetson-b": node_token}, reader, tls_context=context
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert select.select([service.stdout], [], [], 8)[0]
            endpoint = json.loads(service.stdout.readline())
            read_client = RegistryClient(endpoint["service"], reader, ca_file=cert)
            code, initial = read_client.request("GET", "/v1/node/status")
            assert code == 200 and initial["service"]["status"] == "stopped"
            payload = {
                "command_id": "cli-start",
                "assignment_id": "cli-owner",
                "epoch": 1,
                "desired_state": "running",
                "manifest_id": "lab-v1",
                "expected_boot_id": endpoint["boot_id"],
            }
            command_file = private_json(root / "command.json", payload)
            result = subprocess.run(
                [
                    *cli,
                    "assign",
                    "--node-url",
                    endpoint["service"],
                    "--ca-file",
                    str(cert),
                    "--token-file",
                    str(root / "control.token"),
                    "--camera-id",
                    "camera_1",
                    "--command-file",
                    str(command_file),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=8,
            )
            assert json.loads(result.stdout)["accepted"]
            deadline = time.monotonic() + 5
            while True:
                code, observed = read_client.request("GET", "/v1/node/status")
                if observed["service"]["ready"]:
                    break
                assert time.monotonic() < deadline, observed
                time.sleep(0.1)
            subprocess.run(
                [
                    *cli,
                    "agent",
                    "--node-id",
                    "jetson-b",
                    "--node-ip",
                    "127.0.0.1",
                    "--controller",
                    f"https://127.0.0.1:{server.server_port}",
                    "--ca-file",
                    str(cert),
                    "--token-file",
                    str(root / "node.token"),
                    "--service-url",
                    endpoint["service"],
                    "--service-ca-file",
                    str(cert),
                    "--service-token-file",
                    str(root / "reader.token"),
                    "--reports",
                    "2",
                    "--interval",
                    ".1",
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
            node = registry.list_nodes()[0]
            assert node["report_count"] == 2 and node["service_freshness"] == "online"
            assert (
                node["report"]["service"]["session_id"]
                == observed["service"]["session_id"]
            )
            assert "ai-service-status" in node["capabilities"]
            service.terminate()
            assert service.wait(timeout=5) == 0
            restarted = EdgeServiceSupervisor(
                "jetson-b", ServiceManifest(path), root / "state"
            )
            restarted.close()
        finally:
            if service.poll() is None:
                service.terminate()
                try:
                    service.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    service.kill()
                    service.wait(timeout=3)
            service.stdout.close()
            service.stderr.close()
            server.shutdown()
            thread.join(timeout=3)
