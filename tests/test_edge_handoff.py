"""실제 담당 DB와 모형 노드로 수동 인계의 명령 재사용·정리 경계를 확인한다."""

import copy
import json
import sqlite3
import uuid
from types import SimpleNamespace

import pytest

from src.services.edge_handoff import EdgeHandoff
from src.services.edge_orchestrator import RegistryError
from tests.test_edge_ownership import activate, prepare, result
from tests.test_edge_ownership import lab as lab


class Steps:
    """대기 시간을 늘리지 않고 정해진 상태 전이 뒤 중단을 재현한다."""

    def __init__(self, count=50):
        self.remaining = count

    def is_set(self):
        return self.remaining <= 0

    def wait(self, _):
        self.remaining -= 1


def command_stage(node, method, path):
    if method == "GET":
        return None
    if node is None:
        return path.rsplit("/", 1)[1]
    if path.endswith("/activate"):
        return "node_activate"
    return "start" if node == "jetson-b" else "stop"


class Peer:
    def __init__(self, environment, node=None):
        self.env, self.node = environment, node

    def request(self, method, path, payload=None):
        env, store = self.env, self.env.lab.store
        key = (self.node, method, path)
        env.calls.append((key, copy.deepcopy(payload)))
        try:
            if self.node is None:
                if method == "GET" and path == "/v1/nodes":
                    answer = {"nodes": store.registry.list_nodes()}
                elif method == "GET" and path.endswith("/progress"):
                    answer = store.result_progress("camera_1")
                elif path.endswith("/prepare"):
                    answer = store.prepare("camera_1", payload)
                elif path.endswith("/activate"):
                    answer = store.activate(path.split("/")[-2], payload)
                    if env.deliver_result:
                        store.receive(
                            answer["node_id"],
                            result(env.lab, answer, 2, kind="observation"),
                        )
                elif path.endswith("/abort"):
                    answer = store.abort(path.split("/")[-2], payload)
                else:
                    raise AssertionError(path)
            else:
                service = env.lab.nodes[self.node]["report"]["service"]
                if method == "GET":
                    answer = {
                        "node_id": self.node,
                        "boot_id": service["supervisor_boot_id"],
                        "service": copy.deepcopy(service),
                    }
                elif payload["expected_boot_id"] != service["supervisor_boot_id"]:
                    raise RegistryError(409, "stale_supervisor_boot")
                elif payload["command_id"] in env.commands:
                    previous, answer = env.commands[payload["command_id"]]
                    assert previous == payload
                elif path.endswith("/activate"):
                    service["activation"] = {
                        "activation_id": payload["activation_id"],
                        "sequence_floor": 1,
                    }
                    answer = copy.deepcopy(service["activation"])
                    env.commands[payload["command_id"]] = (
                        copy.deepcopy(payload),
                        answer,
                    )
                else:
                    if payload["epoch"] < service["epoch"]:
                        raise RegistryError(409, "stale_epoch")
                    running = payload["desired_state"] == "running"
                    service.update(
                        assignment_id=payload["assignment_id"],
                        epoch=payload["epoch"],
                        status="error"
                        if running and env.fail_start
                        else "running"
                        if running
                        else "stopped",
                        ready=running and not env.fail_start,
                        models_verified=running,
                        backend="deepstream" if running else None,
                        session_id=str(uuid.uuid4()) if running else None,
                        activation=None,
                    )
                    if running and not env.fail_start:
                        candidate = store.camera("camera_1")["pending"]
                        store.receive(
                            self.node,
                            result(
                                env.lab,
                                candidate,
                                1,
                                kind="observation",
                                activation=False,
                            ),
                        )
                    if not running:
                        env.stops.append(
                            (
                                self.node,
                                payload["assignment_id"],
                                store.result_progress("camera_1")["latest_result"],
                            )
                        )
                    answer = {
                        "accepted": True,
                        "command_id": payload["command_id"],
                        "boot_id": service["supervisor_boot_id"],
                    }
                    env.commands[payload["command_id"]] = (
                        copy.deepcopy(payload),
                        answer,
                    )
        except RegistryError as error:
            return error.status, {"error": error.code}
        if env.after_request:
            env.after_request(self.node, method, path, payload, answer)
        if key == env.drop_after:
            env.drop_after = None
            raise ConnectionResetError("response lost after commit")
        return 200, copy.deepcopy(answer)


@pytest.fixture
def handoff_lab(lab, tmp_path):
    candidate, _ = prepare(lab, "jetson-a")
    first, _ = activate(lab, candidate)
    lab.nodes["jetson-a"]["report"]["service"]["status"] = "running"
    lab.nodes["jetson-b"]["report"]["service"]["status"] = "stopped"
    env = SimpleNamespace(
        lab=lab,
        first=first,
        calls=[],
        stops=[],
        commands={},
        deliver_result=True,
        fail_start=False,
        drop_after=None,
        after_request=None,
    )
    controller = Peer(env)
    nodes = {
        name: {"reader": Peer(env, name), "control": Peer(env, name)}
        for name in lab.nodes
    }
    env.make = lambda directory=None: EdgeHandoff(
        directory or tmp_path / "handoff",
        {"reader": controller, "control": controller},
        nodes,
        config_id="test-config",
        clock=lambda: lab.clock.now,
    )
    return env


def test_manual_handoff_waits_for_new_result_then_stops_old_execution(handoff_lab):
    env = handoff_lab
    task = env.make()
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", timeout=5)
        assert outcome["status"] == "complete"
        assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-b"
        assert len(env.stops) == 1
        node, assignment, proof = env.stops[0]
        assert node == "jetson-a" and assignment == env.first["assignment_id"]
        assert proof and proof["kind"] == "observation"
    finally:
        task.close()


def test_prepare_failure_cleans_only_candidate_and_keeps_a(handoff_lab):
    env = handoff_lab
    env.fail_start = True
    task = env.make()
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", timeout=5)
        assert outcome["status"] == "aborted"
        assert (
            env.lab.store.camera("camera_1")["active"]["assignment_id"]
            == env.first["assignment_id"]
        )
        assert env.lab.store.camera("camera_1")["pending"] is None
        assert [node for node, _, _ in env.stops] == ["jetson-b"]
    finally:
        task.close()


@pytest.mark.parametrize(
    "stage", ["prepare", "start", "node_activate", "activate", "stop"]
)
@pytest.mark.parametrize("crash", [False, True])
def test_committed_command_survives_response_loss_and_runner_restart(
    handoff_lab, stage, crash
):
    env = handoff_lab

    def lose_response(node, method, path, payload, answer):
        if command_stage(node, method, path) == stage:
            env.after_request = None
            raise (
                SystemExit("runner crashed")
                if crash
                else ConnectionResetError("response lost")
            )

    env.after_request = lose_response
    task = env.make()
    try:
        if crash:
            with pytest.raises(SystemExit, match="runner crashed"):
                task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
            operation_id = task.state["operation_id"]
            task.close()
            task = env.make()
            assert task.state["operation_id"] == operation_id
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert outcome["status"] == "complete"
        assert len(env.stops) == 1 and env.stops[0][0] == "jetson-a"
        commands = {}
        for (_, method, _), body in env.calls:
            if method != "GET":
                key = body.get("command_id", body.get("request_id"))
                assert commands.setdefault(key, body) == body
        assert len(env.commands) == 3
    finally:
        task.close()


def test_active_b_without_fresh_result_keeps_a_then_resumes(handoff_lab):
    env = handoff_lab
    env.deliver_result = False
    task = env.make()
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps(20))
        assert outcome["status"] == "waiting_for_result"
        assert not env.stops and outcome["proof"] is None
        assert len(outcome["history"]) < 12
        task.close()
        task = env.make()
        active = env.lab.store.camera("camera_1")["active"]
        env.lab.store.receive(
            "jetson-b", result(env.lab, active, 2, kind="observation")
        )
        assert (
            task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())["status"]
            == "complete"
        )
        assert [node for node, _, _ in env.stops] == ["jetson-a"]
    finally:
        task.close()


def test_target_failure_after_activation_restores_previous_owner_with_new_epoch(
    handoff_lab,
):
    env = handoff_lab
    env.deliver_result = False

    def fail_new_owner(node, method, path, payload, answer):
        if (
            command_stage(node, method, path) == "activate"
            and answer["node_id"] == "jetson-b"
        ):
            service = env.lab.nodes["jetson-b"]["report"]["service"]
            service.update(status="error", ready=False)
            env.deliver_result = True
            env.after_request = None

    env.after_request = fail_new_owner
    task = env.make()
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert outcome["status"] == "rolled_back"
        active = env.lab.store.camera("camera_1")["active"]
        assert active["node_id"] == "jetson-a"
        assert active["epoch"] > env.first["epoch"]
        assert active["assignment_id"] != env.first["assignment_id"]
        assert env.lab.store.result_progress("camera_1")["latest_result"]
        assert outcome["intent"]["target_node"] == "jetson-b"
        assert outcome["effective_intent"]["target_node"] == "jetson-a"
        assert [node for node, _, _ in env.stops] == ["jetson-b"]
        calls = len(env.calls)
        assert task.run("camera_1", "jetson-b", "lab-v1")["status"] == "rolled_back"
        assert len(env.calls) == calls
    finally:
        task.close()


@pytest.mark.parametrize("changed", ["activation", "session_id"])
def test_changed_target_activation_never_uses_old_result_to_stop_previous(
    handoff_lab, changed
):
    env = handoff_lab

    def change_activation(node, method, path, payload, answer):
        if command_stage(node, method, path) == "activate":
            service = env.lab.nodes["jetson-b"]["report"]["service"]
            service[changed] = (
                {"activation_id": str(uuid.uuid4()), "sequence_floor": 2}
                if changed == "activation"
                else str(uuid.uuid4())
            )
            env.after_request = None

    env.after_request = change_activation
    task = env.make()
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert outcome["status"] == "reconcile_required"
        assert not env.stops
    finally:
        task.close()


@pytest.mark.parametrize(
    "stage", ["prepare", "start", "node_activate", "activate", "stop"]
)
@pytest.mark.parametrize("crash", [False, True])
def test_rollback_reuses_its_own_command_ids_after_response_loss(
    handoff_lab, stage, crash
):
    env = handoff_lab
    failed = False

    def fail_and_lose_response(node, method, path, payload, answer):
        nonlocal failed
        if not failed and command_stage(node, method, path) == "activate":
            env.lab.nodes["jetson-b"]["report"]["service"].update(
                status="error", ready=False, session_id=None, activation=None
            )
            failed = True
            return
        if method == "GET" or not failed:
            return
        actual = (
            path.rsplit("/", 1)[1]
            if node is None
            else "node_activate"
            if path.endswith("/activate")
            else "start"
            if payload["desired_state"] == "running"
            else "stop"
        )
        if actual == stage:
            env.after_request = None
            raise (
                SystemExit("rollback crashed")
                if crash
                else ConnectionResetError("response lost")
            )

    env.after_request = fail_and_lose_response
    task = env.make()
    try:
        if crash:
            with pytest.raises(SystemExit, match="rollback crashed"):
                task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
            original_id = task.state["operation_id"]
            task.close()
            task = env.make()
            assert task.state["operation_id"] == original_id
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert outcome["status"] == "rolled_back"
        assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-a"
        assert [node for node, _, _ in env.stops] == ["jetson-b"]
        identities = {}
        for (_, method, _), body in env.calls:
            if method != "GET":
                identity = body.get("command_id", body.get("request_id"))
                assert identities.setdefault(identity, body) == body
        assert len(env.commands) == 5
        assert sum(name.startswith("rollback:") for name in task.state["requests"]) == 5
    finally:
        task.close()


def test_failed_rollback_does_not_bounce_back_to_failed_node(handoff_lab):
    env = handoff_lab

    def fail_each_owner(node, method, path, payload, answer):
        if command_stage(node, method, path) == "activate":
            env.lab.nodes[answer["node_id"]]["report"]["service"].update(
                status="error", ready=False, session_id=None, activation=None
            )

    env.after_request = fail_each_owner
    task = env.make()
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert outcome["status"] == "reconcile_required"
        assert outcome["last_error"] == "active_target_failed"
        assert outcome["rollback"] and not env.stops
        assert sum(path.endswith("/prepare") for (_, _, path), _ in env.calls) == 2
    finally:
        task.close()


def test_rollback_never_overwrites_replaced_previous_execution(handoff_lab):
    env = handoff_lab

    def fail_and_replace(node, method, path, payload, answer):
        if command_stage(node, method, path) == "activate":
            env.lab.nodes["jetson-b"]["report"]["service"].update(
                status="error", ready=False
            )
            env.lab.nodes["jetson-a"]["report"]["service"]["assignment_id"] = str(
                uuid.uuid4()
            )
            env.after_request = None

    env.after_request = fail_and_replace
    task = env.make()
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert outcome["status"] == "reconcile_required"
        assert outcome["last_error"] == "previous_execution_changed"
        assert not outcome["rollback"] and not env.stops
        assert sum(path.endswith("/prepare") for (_, _, path), _ in env.calls) == 1
    finally:
        task.close()


def test_rollback_prepare_cas_does_not_replace_concurrent_owner(handoff_lab):
    env = handoff_lab

    def fail_owner(node, method, path, payload, answer):
        if command_stage(node, method, path) == "activate":
            env.lab.nodes["jetson-b"]["report"]["service"].update(
                status="error", ready=False
            )
            env.after_request = None

    env.after_request = fail_owner
    task = env.make()
    try:
        for _ in range(10):
            outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps(1))
            if outcome["status"] == "rolling_back":
                break
        assert outcome["status"] == "rolling_back"
        candidate, _ = prepare(env.lab, "jetson-a")
        replacement, _ = activate(env.lab, candidate)
        commands = len(env.commands)
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert outcome["status"] == "failed"
        assert (
            env.lab.store.camera("camera_1")["active"]["assignment_id"]
            == replacement["assignment_id"]
        )
        assert len(env.commands) == commands and not env.stops
    finally:
        task.close()


def test_controller_restart_requires_reconciliation_without_stop(handoff_lab):
    env = handoff_lab
    task = env.make()
    try:
        task.run("camera_1", "jetson-b", "lab-v1", stop=Steps(2))
        env.lab.store.close()
        env.lab.store = type(env.lab.store)(
            env.lab.root, env.lab.registry, clock=lambda: env.lab.clock.now
        )
        calls_before = len(env.calls)
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert outcome["status"] == "reconcile_required"
        assert outcome["last_error"] == "controller_boot_changed"
        assert not env.stops
        assert all(key[1] == "GET" for key, _ in env.calls[calls_before:])
    finally:
        task.close()


def test_replaced_previous_execution_is_never_stopped(handoff_lab):
    env = handoff_lab

    def replace_a(node, method, path, payload, answer):
        if command_stage(node, method, path) == "activate":
            service = env.lab.nodes["jetson-a"]["report"]["service"]
            service["assignment_id"] = str(uuid.uuid4())
            service["epoch"] += 2
            env.after_request = None

    env.after_request = replace_a
    task = env.make()
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert outcome["status"] == "reconcile_required"
        assert outcome["previous_stop"] == "replaced"
        assert not env.stops
    finally:
        task.close()


def test_concurrent_owner_change_cleans_own_candidate_without_stopping_new_owner(
    handoff_lab,
):
    env = handoff_lab
    replacement = {}

    def replace_owner(node, method, path, payload, answer):
        if command_stage(node, method, path) == "start":
            store = env.lab.store
            candidate = store.camera("camera_1")["pending"]
            store.abort(
                candidate["assignment_id"],
                {"request_id": str(uuid.uuid4()), "controller_boot_id": store.boot_id},
            )
            new, _ = prepare(env.lab, "jetson-a")
            replacement.update(activate(env.lab, new)[0])
            env.after_request = None

    env.after_request = replace_owner
    task = env.make()
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert (
            outcome["status"] == "superseded" and outcome["candidate_stop"] == "stopped"
        )
        assert (
            env.lab.store.camera("camera_1")["active"]["assignment_id"]
            == replacement["assignment_id"]
        )
        assert [node for node, _, _ in env.stops] == ["jetson-b"]
    finally:
        task.close()


def test_journal_failure_prevents_external_command(handoff_lab):
    env = handoff_lab
    task = env.make()
    try:
        task.run("camera_1", "jetson-b", "lab-v1", stop=Steps(1))
        task.db.execute("PRAGMA query_only=ON")
        calls_before = len(env.calls)
        with pytest.raises(sqlite3.OperationalError):
            task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
        assert all(key[1] == "GET" for key, _ in env.calls[calls_before:])
        with pytest.raises(RuntimeError, match="storage unavailable"):
            task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
    finally:
        task.close()


def test_journal_rejects_concurrent_runner_and_changed_intent(handoff_lab):
    env = handoff_lab
    task = env.make()
    try:
        with pytest.raises(BlockingIOError):
            env.make()
        task.run("camera_1", "jetson-b", "lab-v1", stop=Steps(1))
        with pytest.raises(ValueError, match="another request"):
            task.run("camera_1", "jetson-a", "lab-v1", stop=Steps())
    finally:
        task.close()


def test_decision_owner_guard_refuses_changed_owner_before_commands(handoff_lab):
    env = handoff_lab
    expected = {
        "controller_boot_id": env.lab.store.boot_id,
        "assignment_id": env.first["assignment_id"],
        "epoch": env.first["epoch"],
        "decided_at": env.lab.clock.now,
        "expires_at": env.lab.clock.now + 5,
    }
    task = env.make()
    try:
        task.run(
            "camera_1", "jetson-b", "lab-v1", expected_owner=expected, stop=Steps(0)
        )
    finally:
        task.close()
    candidate, _ = prepare(env.lab, "jetson-b")
    activate(env.lab, candidate)
    env.calls.clear()
    task = env.make()
    try:
        outcome = task.run(
            "camera_1", "jetson-b", "lab-v1", expected_owner=expected, stop=Steps()
        )
        assert outcome["status"] == "failed"
        assert outcome["last_error"] == "decision_owner_changed"
        assert all(key[1] == "GET" for key, _ in env.calls)
    finally:
        task.close()


def test_decision_guard_is_part_of_persisted_intent(handoff_lab):
    env = handoff_lab
    expected = {
        "controller_boot_id": env.lab.store.boot_id,
        "assignment_id": env.first["assignment_id"],
        "epoch": env.first["epoch"],
        "decided_at": env.lab.clock.now,
        "expires_at": env.lab.clock.now + 5,
    }
    task = env.make()
    try:
        outcome = task.run(
            "camera_1", "jetson-b", "lab-v1", expected_owner=expected, stop=Steps()
        )
        assert outcome["status"] == "complete"
        with pytest.raises(ValueError, match="another request"):
            task.run("camera_1", "jetson-b", "lab-v1", stop=Steps())
    finally:
        task.close()


def test_timeout_before_prepare_sends_no_new_command(handoff_lab, monkeypatch):
    env = handoff_lab
    task = env.make()
    ticks = iter([0, 2])
    monkeypatch.setattr(
        "src.services.edge_handoff.time.monotonic", lambda: next(ticks, 2)
    )
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", timeout=1, stop=Steps())
        assert (
            outcome["status"] == "failed" and outcome["last_error"] == "handoff_timeout"
        )
        assert all(key[1] == "GET" for key, _ in env.calls)
        assert env.lab.store.camera("camera_1")["pending"] is None
    finally:
        task.close()


def test_timeout_with_lost_prepare_ack_reconciles_then_aborts_without_start(
    handoff_lab, monkeypatch
):
    env = handoff_lab
    task = env.make()
    env.drop_after = (None, "POST", "/v1/cameras/camera_1/prepare")
    ticks = iter([0, 0, 2, 2, 2])
    monkeypatch.setattr(
        "src.services.edge_handoff.time.monotonic", lambda: next(ticks, 2)
    )
    try:
        outcome = task.run("camera_1", "jetson-b", "lab-v1", timeout=1, stop=Steps())
        assert (
            outcome["status"] == "aborted"
            and outcome["candidate_stop"] == "not_started"
        )
        assert env.lab.store.camera("camera_1")["pending"] is None
        assert (
            env.lab.store.camera("camera_1")["active"]["assignment_id"]
            == env.first["assignment_id"]
        )
        assert not env.commands and not env.stops
    finally:
        task.close()


def test_foreign_database_is_not_modified(handoff_lab, tmp_path):
    root = tmp_path / "foreign"
    root.mkdir(mode=0o700)
    path = root / "handoff.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE protected(value)")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not an edge handoff"):
        handoff_lab.make(root)
    assert path.read_bytes() == before


def test_public_state_directory_is_rejected(handoff_lab, tmp_path):
    root = tmp_path / "public"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with pytest.raises(ValueError, match="owner-only"):
        handoff_lab.make(root)
    assert not list(root.iterdir())


@pytest.mark.parametrize("invalid", ["remote_http", "bad_token", "public_config"])
def test_handoff_configuration_keeps_private_credentials_and_tls_boundary(
    tmp_path, invalid
):
    from tests.test_edge_service import private_json

    credentials = private_json(
        tmp_path / "credentials.json",
        {
            "reader_token": uuid.uuid4().hex,
            "control_token": uuid.uuid4().hex,
        },
    )
    endpoint = {
        "url": "http://127.0.0.1:12345",
        "ca_file": None,
        "credentials_file": str(credentials),
    }
    settings = {"controller": endpoint, "nodes": {"jetson-b": endpoint}}
    if invalid == "remote_http":
        endpoint["url"] = "http://192.0.2.1:12345"
    elif invalid == "bad_token":
        credentials.write_text(
            json.dumps({"reader_token": "bad", "control_token": uuid.uuid4().hex})
        )
    config = private_json(tmp_path / "handoff.json", settings)
    if invalid == "public_config":
        config.chmod(0o644)
    with pytest.raises(ValueError):
        EdgeHandoff.from_file(config, tmp_path / "state")
    assert not (tmp_path / "state").exists()
