"""다른 카메라 실행을 유지하는 선택 인계와 입력/보고 격리를 검사한다."""

import copy
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.base_processor import validate_managed_camera
from src.services.edge_node_agent import JetsonNodeAgent
from src.services.edge_orchestrator import (
    NodeRegistry,
    RegistryError,
    observed_camera_service,
)
from tests.test_edge_auto_handoff import automatic as automatic
from tests.test_edge_auto_handoff import warmup
from tests.test_edge_handoff import Steps
from tests.test_edge_handoff import handoff_lab as handoff_lab
from tests.test_edge_orchestrator import registration
from tests.test_edge_ownership import lab as lab
from tests.test_edge_service import setup as setup
from tests.test_edge_service import supervisor as supervisor


def multiple_reports(env):
    """기존 단일 카메라 모형을 유지하며 노드 보고만 여러 카메라로 확장한다."""
    # 기존 인계 모형의 명령 응답은 재사용하고 배경 카메라에는 명령을 허용하지 않는다.
    backgrounds = {
        node: {
            f"{node}-background-{index}": {
                **copy.deepcopy(entry["report"]["service"]),
                "camera_id": f"{node}-background-{index}",
                "session_id": str(uuid.uuid4()),
                "status": "running",
                "epoch": 7,
            }
            for index in range(3 if node == "jetson-a" else 4)
        }
        for node, entry in env.lab.nodes.items()
    }

    def nodes():
        answer = copy.deepcopy(list(env.lab.nodes.values()))
        for entry in answer:
            entry["report"]["services"] = {
                "camera_1": entry["report"].pop("service"),
                **copy.deepcopy(backgrounds[entry["node_id"]]),
            }
            entry["report"]["service"] = None
            entry["services_freshness"] = {
                camera: "online" for camera in entry["report"]["services"]
            }
        return answer

    env.lab.registry.list_nodes = nodes
    return backgrounds


@pytest.mark.parametrize("failed_start", [False, True])
def test_selected_camera_moves_or_rolls_back_without_background_commands(handoff_lab, failed_start):
    env = handoff_lab
    backgrounds = multiple_reports(env)
    before = copy.deepcopy(backgrounds)
    env.fail_start = failed_start
    task = env.make()
    for node, peer in tuple(task.nodes.items()):
        task.nodes[node] = {"cameras": {"camera_1": peer}}
        for camera in backgrounds[node]:
            unused = Mock()
            unused.request.side_effect = AssertionError("background camera received a command")
            task.nodes[node]["cameras"][camera] = {"reader": unused, "control": unused}
    try:
        result = task.run("camera_1", "jetson-b", "lab-v1", timeout=5)
        assert result["status"] == ("aborted" if failed_start else "complete")
        assert env.lab.store.camera("camera_1")["active"]["node_id"] == (
            "jetson-a" if failed_start else "jetson-b"
        )
        assert backgrounds == before
        assert all(path == "/v1/assignments/camera_1" or path.endswith("/activate")
                   for (node, method, path), _ in env.calls if node and method == "PUT")
        assert all(node == ("jetson-b" if failed_start else "jetson-a")
                   for node, _, _ in env.stops)
    finally:
        task.close()


@pytest.mark.parametrize("failback", [False, True])
def test_automatic_move_uses_selected_camera_in_multi_service_report(automatic, failback):
    env = automatic
    if failback:
        env.policy["failback"] = {"preferred_node": "jetson-a", "sustain_seconds": 30}
    backgrounds = multiple_reports(env)
    before = copy.deepcopy(backgrounds)
    task = env.make_auto()
    for node, peer in tuple(task.nodes.items()):
        task.nodes[node] = {"cameras": {"camera_1": peer}}
    warmup(env, task)
    result = task.step(Steps())
    assert result["status"] == "complete"
    assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-b"
    assert backgrounds == before
    assert task.step(Steps())["reason"] == "cooldown"
    env.resources["jetson-a"]["cpu_percent"] = 10
    env.refresh(61)
    if failback:
        # 다중 보고에서도 기존 지속 회복·cooldown 뒤 선택 카메라만 돌아와야 한다.
        assert task.step(Steps())["reason"] == "collecting_recovery_samples"
        for _ in range(5):
            env.refresh(5)
            assert task.step(Steps())["reason"] == "collecting_recovery_samples"
        env.refresh(5)
        assert task.step(Steps())["status"] == "complete"
        assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-a"
        assert task.decision["reasons"] == ["preferred_node_recovered"]
        assert task.step(Steps())["reason"] == "cooldown"
        assert [node for node, _, _ in env.stops] == ["jetson-a", "jetson-b"]
    else:
        assert task.step(Steps())["reason"] == "below_threshold"
        assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-b"
    assert backgrounds == before


def test_stale_selected_camera_cannot_borrow_background_readiness(handoff_lab):
    env = handoff_lab
    multiple_reports(env)
    node = env.lab.registry.list_nodes()[1]
    node["services_freshness"]["camera_1"] = "stale"
    with pytest.raises(RegistryError, match="node_not_observed"):
        observed_camera_service(node, "camera_1")
    node["services_freshness"]["camera_1"] = "online"
    node["report"]["services"]["camera_1"] = None
    with pytest.raises(RegistryError):
        observed_camera_service(node, "camera_1")


def test_agent_and_registry_preserve_per_camera_failure(supervisor, tmp_path):
    snapshot = supervisor.snapshot()
    good = Mock()
    good.request.return_value = (200, snapshot)
    failed = Mock()
    failed.request.side_effect = TimeoutError()
    wrong = Mock()
    wrong.request.return_value = (200, snapshot)
    resources = dict(cpu_percent=10, gpu_percent=10, memory_percent=20,
                     network_rx_bytes_per_second=0, network_tx_bytes_per_second=0)
    agent = JetsonNodeAgent(
        Mock(), "jetson-b", collector=SimpleNamespace(sample=lambda: resources),
        service_clients={"camera_1": good, "camera_2": failed, "camera_3": wrong},
    )
    report = agent._sample()
    assert report["service"] is None
    assert report["services"]["camera_1"]["camera_id"] == "camera_1"
    assert report["services"]["camera_2"] is report["services"]["camera_3"] is None
    registry = NodeRegistry(tmp_path / "registry.sqlite3", allow_real_resources=True,
                            allow_real_services=True)
    entry = registration("jetson-b", boot_id=agent.boot_id)
    entry.update(simulation=False, capabilities=agent.capabilities)
    registry.register(entry)
    registry.report("jetson-b", report)
    node = registry.list_nodes()[0]
    assert node["services_freshness"] == {
        "camera_1": "online", "camera_2": "stale", "camera_3": "stale"
    }
    assert observed_camera_service(node, "camera_1")["supervisor_boot_id"] == supervisor.boot_id
    malformed = copy.deepcopy(report)
    malformed["sequence"] += 1
    malformed["services"]["camera_2"] = snapshot["service"]
    with pytest.raises(RegistryError, match="service_camera_mismatch"):
        registry.report("jetson-b", malformed)


@pytest.mark.parametrize("camera", ["camera_2", None, ""])
def test_managed_camera_rejects_other_inputs(monkeypatch, camera):
    monkeypatch.setenv("EDGE_SERVICE_CAMERA_ID", "camera_1")
    validate_managed_camera("camera_1")
    with pytest.raises(ValueError, match="assigned camera"):
        validate_managed_camera(camera)
    monkeypatch.delenv("EDGE_SERVICE_CAMERA_ID")
    validate_managed_camera("camera_2")


@pytest.mark.parametrize("cameras", [[], [{"id": "camera_2"}],
                                    [{"id": "camera_1"}, {"id": "camera_2"}]])
def test_managed_runtime_rejects_before_model_loading(monkeypatch, cameras):
    from src.bootstrap import runtime

    monkeypatch.setenv("EDGE_SERVICE_CAMERA_ID", "camera_1")
    factory = Mock(side_effect=AssertionError("model loading must not start"))
    monkeypatch.setattr(runtime, "create_processor", factory)
    with pytest.raises(ValueError, match="managed service"):
        runtime.start_processor_runtime(cameras, None)
    factory.assert_not_called()


@pytest.mark.parametrize("missing_node", ["jetson-a", "jetson-b"])
def test_missing_camera_endpoint_fails_before_assignment_write(handoff_lab, missing_node):
    env = handoff_lab
    multiple_reports(env)
    task = env.make()
    for node, peer in tuple(task.nodes.items()):
        task.nodes[node] = {"cameras": {"camera_1": peer}}
    task.nodes[missing_node]["cameras"] = {}
    try:
        with pytest.raises(ValueError, match="camera endpoint"):
            task.run("camera_1", "jetson-b", "lab-v1", timeout=5)
        assert env.lab.store.camera("camera_1")["pending"] is None
        assert env.lab.store.camera("camera_1")["active"] == env.first
        assert all(method == "GET" for (_, method, _), _ in env.calls)
    finally:
        task.close()


def test_target_rejects_second_camera_while_first_is_preparing(handoff_lab):
    from tests.test_edge_ownership import prepare

    env = handoff_lab
    backgrounds = multiple_reports(env)
    prepare(env.lab, "jetson-b")
    camera = next(iter(backgrounds["jetson-b"]))
    payload = {
        "request_id": str(uuid.uuid4()),
        "controller_boot_id": env.lab.store.boot_id,
        "node_id": "jetson-b",
        "manifest_id": "lab-v1",
        "supervisor_boot_id": backgrounds["jetson-b"][camera]["supervisor_boot_id"],
        "expected_assignment_id": None,
        "expected_epoch": 0,
    }
    with pytest.raises(RegistryError, match="target_preparation_in_progress"):
        env.lab.store.prepare(camera, payload)
    assert env.lab.store.camera(camera)["pending"] is None


def test_agent_queries_cameras_together_then_samples_resources(supervisor):
    import threading

    barrier = threading.Barrier(2, timeout=2)
    finished = set()
    snapshot = supervisor.snapshot()

    def request(camera):
        barrier.wait()
        answer = copy.deepcopy(snapshot)
        answer["service"]["camera_id"] = camera
        finished.add(camera)
        return 200, answer

    def resources():
        assert finished == {"camera_1", "camera_2"}
        return {"cpu_percent": 10}

    clients = {
        camera: Mock(request=lambda *_, camera=camera: request(camera))
        for camera in ("camera_1", "camera_2")
    }
    agent = JetsonNodeAgent(Mock(), "jetson-b", service_clients=clients,
                           collector=SimpleNamespace(sample=resources))
    report = agent._sample()
    assert all(report["services"].values())


@pytest.mark.parametrize("backend", ["deepstream", "opencv"])
def test_managed_processor_rejects_dynamic_other_camera_before_side_effects(monkeypatch, backend):
    from src.core.deepstream_processor import DeepStreamProcessor
    from src.core.processor import VideoProcessor

    monkeypatch.setenv("EDGE_SERVICE_CAMERA_ID", "camera_1")
    processor_type = DeepStreamProcessor if backend == "deepstream" else VideoProcessor
    processor = processor_type.__new__(processor_type)
    with pytest.raises(ValueError, match="assigned camera"):
        processor.add_camera("camera_2", "unused")


def test_stopping_selected_real_child_keeps_background_processes(setup):
    import json
    import os
    import socket
    from pathlib import Path

    from src.services.edge_service import EdgeServiceSupervisor, ServiceManifest
    from tests.test_edge_service import command, private_json, ready

    root, _, template = setup
    supervisors = []
    try:
        for node, camera in (("jetson-a", "camera_1"), ("jetson-a", "camera_2"),
                             ("jetson-b", "camera_3"), ("jetson-b", "camera_1")):
            directory = root / (node + "-" + camera)
            directory.mkdir(mode=0o700)
            data = copy.deepcopy(template)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            environment = json.loads(Path(data["environment_file"]).read_text())
            environment["FIXTURE_MODE_PATH"] = str(directory / "mode")
            # 잘못된 환경값보다 감독자 manifest의 카메라 식별자가 우선해야 한다.
            environment["EDGE_SERVICE_CAMERA_ID"] = "wrong-camera"
            data.update(camera_id=camera, cwd=str(directory),
                        environment_file=str(private_json(directory / "env.json", environment)),
                        metrics_url=f"http://127.0.0.1:{port}/metrics")
            manifest = private_json(directory / "manifest.json", data)
            instance = EdgeServiceSupervisor(node, ServiceManifest(manifest), directory / "state")
            supervisors.append(instance)
            instance.submit(camera, command(instance))
            ready(instance)
            # 새 세션이어도 시험 부모의 CPU 한정·우선순위를 그대로 상속해야 한다.
            assert os.sched_getaffinity(instance.process.pid) == os.sched_getaffinity(0)
            assert os.getpriority(os.PRIO_PROCESS, instance.process.pid) == os.getpriority(os.PRIO_PROCESS, 0)
        source, background_a, background_b, target = supervisors
        identities = [(item.process, item.session_id) for item in (background_a, background_b)]
        source_child = source.process
        assert target.snapshot()["service"]["ready"]
        source.submit("camera_1", command(source, state="stopped"))
        source.tick()
        assert source_child.poll() is not None
        for item, (child, session) in zip((background_a, background_b), identities):
            item.tick()
            assert child.poll() is None and item.process is child
            assert item.session_id == session and item.snapshot()["service"]["ready"]
        assert target.process.poll() is None
    finally:
        for instance in reversed(supervisors):
            instance.close()


def test_multi_camera_reports_cross_http_size_boundaries(supervisor, tmp_path):
    import json
    import threading
    from datetime import datetime, timezone

    from src.services.edge_node_agent import RegistryClient
    from src.services.edge_orchestrator import create_registry_server

    registry = NodeRegistry(tmp_path / "http.sqlite3", allow_real_resources=True,
                            allow_real_services=True)
    # 64KiB 응답 경계를 충분히 넘도록 32카메라 보고를 4개 모의 노드에 쌓는다.
    credentials = {f"jetson-{index}": str(index) * 40 for index in range(4)}
    service = supervisor.snapshot()["service"]
    services = {f"camera_{index}": dict(service, camera_id=f"camera_{index}")
                for index in range(32)}
    payload = {
        "boot_id": str(uuid.uuid4()), "sequence": 1, "simulation": False,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "resources": dict(cpu_percent=10, gpu_percent=10, memory_percent=10,
                          network_rx_bytes_per_second=0, network_tx_bytes_per_second=0),
        "service": None, "services": services,
    }
    assert len(json.dumps(payload).encode()) > 16384
    with create_registry_server(registry, credentials, "r" * 40) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            for node, token in credentials.items():
                client = RegistryClient(url, token)
                entry = registration(node, payload["boot_id"])
                entry.update(simulation=False, capabilities=["jetson-resources", "ai-service-status"])
                assert client.request("POST", "/v1/nodes/register", entry)[0] == 200
                assert client.request("POST", f"/v1/nodes/{node}/reports", payload)[0] == 200
            code, response = RegistryClient(url, "r" * 40).request("GET", "/v1/nodes")
            assert code == 200 and len(response["nodes"]) == 4
            assert len(json.dumps(response).encode()) > 65536
            assert len(response["nodes"][0]["report"]["services"]) == 32
            assert client.request("POST", f"/v1/nodes/{node}/reports",
                                  {"padding": "x" * 262144}) == (400, {"error": "invalid_content_length"})
        finally:
            server.shutdown()
            thread.join(timeout=3)


def test_camera_endpoint_configuration_keeps_legacy_and_selects_exact_camera(tmp_path):
    from src.services.edge_handoff import EdgeHandoff
    from src.services.edge_preview import EdgePreviewRelay
    from tests.test_edge_service import private_json

    credentials = private_json(tmp_path / "secrets.json", {
        "reader_token": "r" * 40, "control_token": "c" * 40, "stream_token": "s" * 40,
    })
    stream_credentials = private_json(tmp_path / "stream.json", {"stream_token": "s" * 40})
    peer = {"url": "http://127.0.0.1:18001", "ca_file": None,
            "credentials_file": str(credentials)}
    nodes = {"jetson-a": {"cameras": {"camera_1": peer}}, "jetson-b": peer}
    config = private_json(tmp_path / "handoff.json", {"controller": peer, "nodes": nodes})
    with_context = EdgeHandoff.from_file(config, tmp_path / "handoff")
    try:
        assert with_context._peer("jetson-a", "camera_1")["reader"].port == 18001
        assert with_context._peer("jetson-b", "camera_1")["control"].port == 18001
        with pytest.raises(ValueError, match="camera endpoint"):
            with_context._peer("jetson-a", "camera_2")
    finally:
        with_context.close()
    peer["credentials_file"] = str(stream_credentials)
    preview = private_json(tmp_path / "preview.json", {"nodes": nodes})
    relay = EdgePreviewRelay.from_file(preview, Mock())
    assert relay.clients["jetson-a"]["cameras"]["camera_1"].port == 18001
    assert relay.clients["jetson-b"].port == 18001


def test_agent_cli_loads_camera_service_map(tmp_path, monkeypatch):
    from runners import run_edge_orchestration as runner
    from tests.test_edge_service import private_json

    token = tmp_path / "reader.token"
    token.write_text("r" * 40)
    token.chmod(0o600)
    config = private_json(tmp_path / "services.json", {
        "camera_1": {"url": "http://127.0.0.1:18001", "ca_file": None, "token_file": str(token)},
        "camera_2": {"url": "http://127.0.0.1:18002", "ca_file": None, "token_file": str(token)},
    })
    constructor = Mock(return_value=SimpleNamespace(run=lambda *_, **__: None,
                                                   sequence=1, last_error=None))
    clients = Mock()
    monkeypatch.setattr(runner, "RegistryClient", clients)
    monkeypatch.setattr(runner, "JetsonNodeAgent", constructor)
    runner.main(["agent", "--node-id", "jetson-a", "--node-ip", "127.0.0.1",
                 "--controller", "https://controller.example.invalid:18000",
                 "--ca-file", "/unused/ca.crt", "--token-file", str(token),
                 "--services-config", str(config), "--reports", "1"])
    options = constructor.call_args.kwargs
    assert options["service_client"] is None
    assert set(options["service_clients"]) == {"camera_1", "camera_2"}
    assert all(call.kwargs["timeout"] == 1 for call in clients.call_args_list[1:])



@pytest.mark.parametrize("payload", [None, 1, [], "invalid"])
def test_invalid_report_shape_fails_validation(tmp_path, payload):
    registry = NodeRegistry(tmp_path / "invalid.sqlite3")
    with pytest.raises(RegistryError, match="invalid_fields"):
        registry.report("jetson-a", payload)
