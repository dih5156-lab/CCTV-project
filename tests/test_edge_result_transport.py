"""MQTT 전송 접수와 중앙 저장 ACK를 구분하고 원본 변경·저장 실패를 검사한다."""

import copy
import json
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.services.edge_orchestrator import RegistryError
from src.services.edge_result_context import EdgeResultContext
from src.services.edge_result_transport import (
    EdgeResultCollector,
    EdgeResultRelay,
    mqtt_settings,
    raw_topic,
)
from tests.test_edge_ownership import lab as lab
from tests.test_edge_ownership import observed, prepare
from tests.test_edge_service import private_json


def settings():
    return {
        "host": "127.0.0.1",
        "port": 1883,
        "ca_file": None,
        "username": "isolated-test",
        "password": "unit-fixture-only",
    }


def setup_node(lab, tmp_path, node_id="jetson-a"):
    assignment, _ = prepare(lab, node_id)
    service = observed(lab, assignment)
    root = tmp_path / node_id
    root.mkdir(mode=0o700)
    context = EdgeResultContext(
        root / "results.sqlite3", create=True, clock=lambda: lab.clock.now
    )
    identity = {
        key: service[key]
        for key in (
            "camera_id",
            "assignment_id",
            "epoch",
            "supervisor_boot_id",
            "session_id",
        )
    }
    context.begin(dict(identity, node_id=node_id))
    return assignment, context, service


def sample(context, session, *, kind="event"):
    stamp = context.capture(session)
    event = {"event_id": stamp["event_id"], "camera_id": stamp["edge"]["camera_id"]}
    event.update(
        {"frames_inferred": 30} if kind == "observation" else {"type": "person"}
    )
    payload = {"kind": kind, "event": event, "edge": stamp["edge"]}
    context.enqueue(payload)
    return payload


def activate_node(lab, assignment, context, service, collector):
    observation = sample(context, service["session_id"], kind="observation")
    collector.accept(raw_topic(observation), json.dumps(observation).encode())
    context.confirm(
        lab.store.receipt(assignment["node_id"], observation["event"]["event_id"])
    )
    activation = context.activate(
        service["session_id"],
        {
            "command_id": "activate-" + assignment["node_id"],
            "expected_activation_id": None,
            "activation_id": assignment["activation_id"],
        },
    )
    service["activation"] = activation
    lab.store.activate(
        assignment["assignment_id"],
        {
            "request_id": "activate-" + assignment["node_id"],
            "controller_boot_id": lab.store.boot_id,
            "session_id": service["session_id"],
            **activation,
        },
    )


class LocalController:
    def __init__(self, store, node_id):
        self.store, self.node_id = store, node_id

    def request(self, method, path):
        assert method == "GET"
        try:
            return 200, self.store.receipt(self.node_id, path.rsplit("/", 1)[1])
        except RegistryError as error:
            return error.status, {"error": error.code}


def test_publish_rc_zero_is_not_storage_ack_and_conflicting_ack_is_retained(
    lab, tmp_path
):
    assignment, context, service = setup_node(lab, tmp_path)
    collector = EdgeResultCollector(
        lab.store, lab.nodes, {"mqtt": settings(), "client_id": "collector"}
    )
    controller = LocalController(lab.store, "jetson-a")
    relay = EdgeResultRelay(context, "jetson-a", controller, settings())
    sent = []
    relay.client.publish = lambda *args, **kwargs: (
        sent.append((args, kwargs)) or SimpleNamespace(rc=0)
    )
    try:
        activate_node(lab, assignment, context, service, collector)
        payload = sample(context, service["session_id"])
        relay.step()
        assert len(context.pending()) == 1 and len(sent) == 1
        assert sent[0][1] == {"qos": 1, "retain": False}
        collector.accept(*[sent[0][0][0], sent[0][0][1].encode()])
        ack = lab.store.receipt("jetson-a", payload["event"]["event_id"])
        original_request = controller.request
        controller.request = lambda *args: (200, dict(ack, sha256="0" * 64))
        relay.step()
        assert len(context.pending()) == 1
        controller.request = original_request
        relay.step()
        assert context.pending() == []
        assert lab.store.consume(ack["event_id"], ack["sha256"])["consumed"]
        assert lab.store.consume(ack["event_id"], ack["sha256"])["duplicate"]
    finally:
        relay.close()
        collector.close()
        context.close()


def test_new_arrivals_do_not_starve_pending_retries_after_restart(lab, tmp_path):
    _, context, service = setup_node(lab, tmp_path)
    relay = EdgeResultRelay(
        context, "jetson-a", LocalController(lab.store, "jetson-a"), settings()
    )
    sent = []
    relay.client.publish = lambda topic, body, **kwargs: (
        sent.append(json.loads(body)["event"]["event_id"]) or SimpleNamespace(rc=4)
    )
    try:
        original = sample(context, service["session_id"])
        original_id = original["event"]["event_id"]
        relay.step()
        assert sent == [original_id]
        path = context.path
        context.close()
        context = EdgeResultContext(path, clock=lambda: lab.clock.now)
        relay.context = context
        # 장애 중 새 결과가 한 배치씩 들어와도 과거 재시도를 계속 밀어내지 않는다.
        for _ in range(3):
            lab.clock.now += 1
            for _ in range(20):
                sample(context, service["session_id"])
            context.enqueue(original)
            sent.clear()
            relay.step()
            assert original_id in sent
        assert (
            next(
                item["payload"]
                for item in context.pending(100)
                if item["event_id"] == original_id
            )
            == original
        )
    finally:
        relay.close()
        context.close()


def test_storage_failure_never_returns_normally_and_poison_is_durably_quarantined(
    lab, tmp_path
):
    assignment, context, service = setup_node(lab, tmp_path)
    collector = EdgeResultCollector(
        lab.store, lab.nodes, {"mqtt": settings(), "client_id": "collector"}
    )
    try:
        activate_node(lab, assignment, context, service, collector)
        payload = sample(context, service["session_id"])
        lab.store.db.execute("PRAGMA query_only=ON")
        with pytest.raises(sqlite3.OperationalError):
            collector.accept(raw_topic(payload), json.dumps(payload).encode())
        with pytest.raises(sqlite3.OperationalError):
            collector.accept(raw_topic(payload), b"not-json")
        lab.store.db.execute("PRAGMA query_only=OFF")
        collector.accept(raw_topic(payload), b"not-json")
        wrong = copy.deepcopy(payload)
        wrong["edge"]["node_id"] = "jetson-b"
        collector.accept(raw_topic(payload), json.dumps(wrong).encode())
        assert (
            lab.store.db.execute("SELECT COUNT(*) FROM rejected_mqtt").fetchone()[0]
            == 2
        )
        with pytest.raises(RegistryError, match="receipt_not_found"):
            lab.store.receipt("jetson-a", payload["event"]["event_id"])
        collector.accept(raw_topic(payload), json.dumps(payload).encode())
        assert (
            lab.store.receipt("jetson-a", payload["event"]["event_id"])["decision"]
            == "accepted"
        )
    finally:
        collector.close()
        context.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"host": "192.0.2.1"},
        {"host": "localhost"},
        {"username": ""},
        {"password": ""},
        {"port": True},
    ],
)
def test_remote_transport_cannot_disable_tls_or_inherit_operating_credentials(changes):
    with pytest.raises((ValueError, RegistryError)):
        mqtt_settings(dict(settings(), **changes))


@pytest.mark.parametrize("role", ["service", "controller"])
def test_transport_shutdown_error_still_closes_owned_resources(
    tmp_path, monkeypatch, role
):
    from runners import run_edge_orchestration as runner
    from src.services import (
        edge_controller_api,
        edge_orchestrator,
        edge_ownership,
        edge_result_transport,
        edge_service,
    )

    owned, transport, server = MagicMock(), MagicMock(), MagicMock()
    transport.close.side_effect = RuntimeError("injected transport shutdown error")
    server.__enter__.return_value.server_port = 12345
    secrets = {"reader_token": "r" * 32, "control_token": "c" * 32}
    if role == "controller":
        secrets.update(node_tokens={"jetson-a": "a" * 32}, consumer_token="x" * 32)
    credentials = private_json(tmp_path / "secrets.json", secrets)
    arguments = [
        role,
        "--state-dir",
        str(tmp_path / "state"),
        "--credentials",
        str(credentials),
    ]
    if role == "service":
        monkeypatch.setattr(edge_service, "ServiceManifest", MagicMock())
        monkeypatch.setattr(
            edge_service, "EdgeServiceSupervisor", MagicMock(return_value=owned)
        )
        monkeypatch.setattr(
            edge_service, "create_service_server", MagicMock(return_value=server)
        )
        monkeypatch.setattr(
            edge_result_transport.EdgeResultRelay,
            "from_file",
            MagicMock(return_value=transport),
        )
        arguments += [
            "--node-id",
            "jetson-a",
            "--manifest",
            "unused.json",
            "--enable-results",
            "--result-transport",
            "unused.json",
        ]
        owned.boot_id = "fixture-boot"
    else:
        monkeypatch.setattr(runner.ssl, "SSLContext", MagicMock())
        monkeypatch.setattr(edge_orchestrator, "NodeRegistry", MagicMock())
        monkeypatch.setattr(
            edge_ownership, "EdgeOwnershipStore", MagicMock(return_value=owned)
        )
        monkeypatch.setattr(
            edge_controller_api,
            "create_controller_server",
            MagicMock(return_value=server),
        )
        monkeypatch.setattr(
            edge_result_transport.EdgeResultCollector,
            "from_file",
            MagicMock(return_value=transport),
        )
        arguments += [
            "--bind",
            "127.0.0.1",
            "--cert",
            str(credentials),
            "--key",
            str(credentials),
            "--enable-ownership",
            "--result-mqtt",
            "unused.json",
        ]
        owned.boot_id = "fixture-boot"
    stop = threading.Event()
    stop.set()
    with pytest.raises(RuntimeError, match="injected transport shutdown error"):
        getattr(runner, "run_" + role)(
            runner.build_parser().parse_args(arguments), stop
        )
    owned.close.assert_called_once_with()
    server.__exit__.assert_called_once()


@pytest.mark.parametrize("worker_class", [EdgeResultCollector, EdgeResultRelay])
def test_stalled_worker_still_disconnects_mqtt(worker_class):
    worker = worker_class.__new__(worker_class)
    worker.stop_event = threading.Event()
    worker.thread = MagicMock()
    worker.thread.is_alive.return_value = True
    worker.client = worker.subscriber = MagicMock()
    with pytest.raises(RuntimeError, match="has not stopped"):
        worker.close()
    assert worker.stop_event.is_set()
    worker.client.disconnect.assert_called_once_with()
    if worker_class is EdgeResultRelay:
        worker.client.loop_stop.assert_called_once_with()
