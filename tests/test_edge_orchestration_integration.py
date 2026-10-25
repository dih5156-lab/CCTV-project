"""실제 loopback HTTP로 모형 2노드의 주기 보고와 실패 복구를 검증한다."""

import copy
import secrets
import threading
from unittest.mock import Mock, patch

import pytest

from runners.run_edge_orchestration_demo import run_demo
from src.services.edge_node_agent import JetsonNodeAgent, MockNodeAgent, RegistryClient
from src.services.edge_orchestrator import NodeRegistry, create_registry_server


@pytest.fixture
def live_registry(tmp_path):
    registry = NodeRegistry(tmp_path / "nodes.sqlite3", allow_real_resources=True)
    tokens = {key: secrets.token_urlsafe(32) for key in ("mock-a", "mock-b")}
    reader = secrets.token_urlsafe(32)
    server = create_registry_server(registry, tokens, reader)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield registry, tokens, RegistryClient(url, reader), url
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


def test_two_nodes_report_periodically_over_http(live_registry):
    store, tokens, reader, url = live_registry
    agents = [
        MockNodeAgent(RegistryClient(url, token), key, interval=0.03, active=i == 0)
        for i, (key, token) in enumerate(tokens.items())
    ]
    stop = threading.Event()
    workers = [
        threading.Thread(target=a.run, args=(stop,), kwargs={"report_limit": 3})
        for a in agents
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
        assert not any(worker.is_alive() for worker in workers)
        code, result = reader.request("GET", "/v1/nodes")
        assert code == 200 and result["simulation"] is True
        assert [n["report_count"] for n in result["nodes"]] == [3, 3]
        assert [n["report"]["service"]["status"] for n in result["nodes"]] == [
            "running",
            "standby",
        ]
        assert all(n["freshness"] == "online" for n in result["nodes"])
        assert all(
            n["report"]["service"]["uptime_seconds"] >= 0.06 for n in result["nodes"]
        )
        persisted = NodeRegistry(store.path).list_nodes()
        assert [n["report"] for n in persisted] == [
            n["report"] for n in result["nodes"]
        ]
        assert all(n["report_count"] == 3 for n in persisted)
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=5)


def test_node_token_cannot_register_or_report_as_another_node(live_registry):
    _, tokens, reader, url = live_registry
    client = RegistryClient(url, tokens["mock-a"])
    impostor = MockNodeAgent(client, "mock-b")
    assert not impostor.report_once()
    assert impostor.last_error == "unauthorized"
    assert reader.request("GET", "/v1/nodes")[1]["nodes"] == []
    assert client.request("GET", "/v1/nodes")[0] == 401
    assert client.request("POST", "/v1/nodes/mock-b/reports", {})[0] == 401
    assert (
        RegistryClient(url, secrets.token_urlsafe(32)).request("GET", "/v1/nodes")[0]
        == 401
    )


def test_lost_report_ack_retries_identical_payload_once(live_registry):
    store, tokens, _, url = live_registry
    client = RegistryClient(url, tokens["mock-a"])
    real_request = client.request
    dropped = []

    def drop_after_commit(method, path, payload=None):
        result = real_request(method, path, payload)
        if path.endswith("/reports") and not dropped:
            dropped.append(copy.deepcopy(payload))
            raise OSError("simulated lost ACK")
        return result

    agent = MockNodeAgent(client, "mock-a")
    with patch.object(client, "request", side_effect=drop_after_commit):
        assert not agent.report_once()
        assert agent.pending == dropped[0]
        assert agent.report_once()
    assert store.list_nodes()[0]["report_count"] == 1
    assert agent.sequence == 1 and agent.pending is None


def test_transport_failure_recovers_without_changing_boot(live_registry):
    store, tokens, _, url = live_registry
    client = RegistryClient(url, tokens["mock-a"])
    agent = MockNodeAgent(client, "mock-a")
    boot = agent.boot_id
    with patch.object(client, "request", side_effect=ConnectionRefusedError):
        assert not agent.report_once()
    assert agent.report_once()
    assert store.list_nodes()[0]["boot_id"] == boot
    assert agent.last_error is None


def test_database_failure_returns_503_and_agent_retries(live_registry):
    import sqlite3

    store, tokens, _, url = live_registry
    agent = MockNodeAgent(RegistryClient(url, tokens["mock-a"]), "mock-a")
    with patch.object(
        store, "report", side_effect=sqlite3.OperationalError("private path")
    ):
        assert not agent.report_once()
        assert agent.last_error == "storage_unavailable"
    assert store.list_nodes()[0]["report_count"] == 0
    assert agent.report_once()
    assert store.list_nodes()[0]["report_count"] == 1


def test_agent_restart_uses_new_boot_and_old_agent_is_fenced(live_registry):
    store, tokens, _, url = live_registry
    client = RegistryClient(url, tokens["mock-a"])
    old = MockNodeAgent(client, "mock-a")
    assert old.report_once()
    new = MockNodeAgent(client, "mock-a")
    assert new.report_once()
    assert not old.report_once()
    assert old.last_error == "retired_boot"
    assert store.list_nodes()[0]["boot_id"] == new.boot_id


def test_invalid_http_body_and_unknown_endpoint(live_registry):
    _, tokens, reader, url = live_registry
    client = RegistryClient(url, tokens["mock-a"])
    assert client.request("POST", "/v1/nodes/register", {"x": "a" * 17000})[0] == 400
    assert reader.request("GET", "/unknown")[0] == 404


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.2.141:80",
        "http://localhost:80",
        "https://127.0.0.1:80",
        "http://127.0.0.1:80/path",
    ],
)
def test_nonlocal_or_ambiguous_endpoint_is_rejected(url):
    with pytest.raises(ValueError):
        RegistryClient(url, "unused")


def test_stop_interrupts_long_report_wait(live_registry):
    _, tokens, _, url = live_registry
    agent = MockNodeAgent(RegistryClient(url, tokens["mock-a"]), "mock-a", interval=60)
    stop = threading.Event()
    sent = threading.Event()
    original = agent.report_once

    def observe():
        result = original()
        sent.set()
        return result

    with patch.object(agent, "report_once", side_effect=observe):
        worker = threading.Thread(target=agent.run, args=(stop,))
        worker.start()
        try:
            assert sent.wait(5)
        finally:
            stop.set()
            worker.join(timeout=5)
        assert not worker.is_alive()


def test_demo_uses_isolated_runtime_and_finishes():
    snapshot = run_demo(interval=0.02, reports=2)
    assert len(snapshot["nodes"]) == 2
    assert all(n["report_count"] == 2 and n["simulation"] for n in snapshot["nodes"])


def resource_sample(cpu=40):
    return {
        "cpu_percent": cpu,
        "memory_percent": 25,
        "gpu_percent": 35.7,
        "network_rx_bytes_per_second": None,
        "network_tx_bytes_per_second": None,
    }


def test_real_resource_agent_reports_periodically_and_persists(live_registry):
    store, tokens, reader, url = live_registry
    samples = [resource_sample(None), resource_sample(40), resource_sample(50)]
    collector = Mock()
    collector.sample.side_effect = samples
    agent = JetsonNodeAgent(
        RegistryClient(url, tokens["mock-a"]),
        "mock-a",
        interval=0.02,
        collector=collector,
    )
    agent.run(threading.Event(), report_limit=3)
    code, result = reader.request("GET", "/v1/nodes")
    node = result["nodes"][0]
    assert code == 200 and result["simulation"] is False
    assert node["simulation"] is False and node["report"]["simulation"] is False
    assert node["freshness"] == "online" and node["report_count"] == 3
    assert node["report"]["resources"] == samples[-1]
    assert node["report"]["service"] is None
    assert "mock-inference" not in node["capabilities"]
    assert NodeRegistry(store.path).list_nodes()[0]["report"] == node["report"]


def test_real_agent_ack_retry_does_not_resample_or_claim_ai(live_registry):
    store, tokens, _, url = live_registry
    client = RegistryClient(url, tokens["mock-a"])
    collector = Mock()
    collector.sample.return_value = resource_sample()
    agent = JetsonNodeAgent(client, "mock-a", collector=collector)
    original = client.request
    dropped = []

    def lose_ack(method, path, payload=None):
        result = original(method, path, payload)
        if path.endswith("/reports") and not dropped:
            dropped.append(copy.deepcopy(payload))
            raise OSError("lost ACK")
        return result

    with patch.object(client, "request", side_effect=lose_ack):
        assert not agent.report_once()
        assert agent.pending == dropped[0]
        assert agent.report_once()
    assert collector.sample.call_count == 1
    assert store.list_nodes()[0]["report_count"] == 1
    collector.sample.return_value = {key: None for key in resource_sample()}
    assert agent.report_once()
    node = store.list_nodes()[0]
    assert all(value is None for value in node["report"]["resources"].values())
    assert node["report"]["service"] is None and node["report_count"] == 2


def test_real_demo_uses_one_local_host_and_keeps_default_mock_mode():
    with patch(
        "src.services.edge_node_agent.JetsonResourceCollector"
    ) as collector_type:
        collector_type.return_value.sample.return_value = resource_sample()
        snapshot = run_demo(interval=0.02, reports=2, real_resources=True)
        collector_type.assert_called_once_with()
        assert collector_type.return_value.sample.call_count == 2
        assert snapshot["simulation"] is False
        assert [node["node_id"] for node in snapshot["nodes"]] == ["local-jetson"]
        assert snapshot["nodes"][0]["report"]["resources"]["gpu_percent"] == 35.7
        assert snapshot["nodes"][0]["report"]["service"] is None
        collector_type.reset_mock()
        mock = run_demo(interval=0.02, reports=2)
        collector_type.assert_not_called()
        assert len(mock["nodes"]) == 2 and mock["simulation"] is True
