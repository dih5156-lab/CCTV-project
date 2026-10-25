"""단일 담당 CAS·준비 큐 차단·commit ACK·재기동/출력 경계를 SQLite로 검증한다."""

import copy
import hashlib
import json
import sqlite3
import threading
import uuid
from types import SimpleNamespace

import pytest

from src.services.edge_orchestrator import RegistryError, _utc
from src.services.edge_ownership import EdgeOwnershipStore


@pytest.fixture
def lab(tmp_path):
    clock = SimpleNamespace(now=1800000000.0)
    nodes = {}
    for node_id in ("jetson-a", "jetson-b"):
        nodes[node_id] = {
            "node_id": node_id,
            "simulation": False,
            "freshness": "online",
            "service_freshness": "online",
            "received_at": _utc(clock.now),
            "report": {
                "service": {
                    "camera_id": "camera_1",
                    "manifest_id": "lab-v1",
                    "supervisor_boot_id": str(uuid.uuid4()),
                    "observed_at": _utc(clock.now),
                    "epoch": 0,
                    "assignment_id": None,
                    "session_id": None,
                    "ready": False,
                    "backend": None,
                    "models_verified": False,
                    "activation": None,
                }
            },
        }
    registry = SimpleNamespace(list_nodes=lambda: copy.deepcopy(list(nodes.values())))
    store = EdgeOwnershipStore(tmp_path / "control", registry, clock=lambda: clock.now)
    result = SimpleNamespace(
        store=store,
        clock=clock,
        nodes=nodes,
        root=tmp_path / "control",
        registry=registry,
    )
    try:
        yield result
    finally:
        result.store.close()


def prepare(lab, node_id, *, request_id=None, expected=None):
    owner = lab.store.camera("camera_1")["active"] if expected is None else expected
    payload = {
        "request_id": request_id or str(uuid.uuid4()),
        "controller_boot_id": lab.store.boot_id,
        "node_id": node_id,
        "manifest_id": "lab-v1",
        "supervisor_boot_id": lab.nodes[node_id]["report"]["service"][
            "supervisor_boot_id"
        ],
        "expected_assignment_id": owner["assignment_id"] if owner else None,
        "expected_epoch": owner["epoch"] if owner else 0,
    }
    return lab.store.prepare("camera_1", payload), payload


def observed(lab, assignment, *, sequence_floor=1):
    node = lab.nodes[assignment["node_id"]]
    service = node["report"]["service"]
    service.update(
        epoch=assignment["epoch"],
        assignment_id=assignment["assignment_id"],
        session_id=assignment["session_id"] or str(uuid.uuid4()),
        ready=True,
        backend="deepstream",
        models_verified=True,
        observed_at=_utc(lab.clock.now),
        activation={
            "activation_id": assignment["activation_id"],
            "sequence_floor": sequence_floor,
        },
    )
    node["received_at"] = _utc(lab.clock.now)
    return service


def result(lab, assignment, sequence, *, kind="event", activation=True, event_id=None):
    service = lab.nodes[assignment["node_id"]]["report"]["service"]
    event = {"event_id": event_id or str(uuid.uuid4()), "camera_id": "camera_1"}
    if kind == "observation":
        event["frames_inferred"] = 30
    else:
        event.update(type="fall", confidence=0.9)
    return {
        "kind": kind,
        "event": event,
        "edge": {
            "node_id": assignment["node_id"],
            "camera_id": "camera_1",
            "assignment_id": assignment["assignment_id"],
            "epoch": assignment["epoch"],
            "supervisor_boot_id": assignment["supervisor_boot_id"],
            "session_id": service["session_id"],
            "activation_id": assignment["activation_id"] if activation else None,
            "event_sequence": sequence,
            "captured_at": _utc(lab.clock.now),
            "expires_at": _utc(lab.clock.now + 60),
        },
    }


def activate(lab, assignment, *, sequence_floor=1, sample=True):
    service = observed(lab, assignment, sequence_floor=sequence_floor)
    if sample:
        ack = lab.store.receive(
            assignment["node_id"],
            result(
                lab, assignment, sequence_floor, kind="observation", activation=False
            ),
        )
        assert ack["decision"] == "observed"
    payload = {
        "request_id": str(uuid.uuid4()),
        "controller_boot_id": lab.store.boot_id,
        "session_id": service["session_id"],
        "activation_id": assignment["activation_id"],
        "sequence_floor": sequence_floor,
    }
    return lab.store.activate(assignment["assignment_id"], payload), payload


def test_preparation_has_no_owner_and_ack_never_promotes_queued_results(lab):
    candidate, _ = prepare(lab, "jetson-a")
    observed(lab, candidate)
    queued = result(lab, candidate, 2, activation=False)
    rejected = lab.store.receive("jetson-a", queued)
    assert rejected["stored"] and rejected["decision"] == "rejected"
    assert lab.store.camera("camera_1")["active"] is None
    owner, _ = activate(lab, candidate)
    assert lab.store.receive("jetson-a", queued) == rejected
    late_queued = result(lab, owner, 3, activation=False)
    assert lab.store.receive("jetson-a", late_queued)["reason"] == "inactive_execution"
    new = result(lab, owner, 4)
    ack = lab.store.receive("jetson-a", new)
    assert ack["decision"] == "accepted"
    pending = lab.store.pending()
    assert [item["event_id"] for item in pending] == [new["event"]["event_id"]]
    assert lab.store.consume(ack["event_id"], ack["sha256"])["consumed"]
    assert lab.store.consume(ack["event_id"], ack["sha256"])["duplicate"]
    assert not lab.store.pending()


def test_handover_keeps_a_until_b_sample_activation_then_rejects_late_a(lab):
    first, _ = prepare(lab, "jetson-a")
    a, _ = activate(lab, first)
    event_a = result(lab, a, 2)
    ack_a = lab.store.receive("jetson-a", event_a)
    b, _ = prepare(lab, "jetson-b")
    assert lab.store.camera("camera_1")["active"]["assignment_id"] == a["assignment_id"]
    b, _ = activate(lab, b)
    assert b["epoch"] > a["epoch"]
    # 검증 직후 전환됐더라도 최종 수신 경계에서 과거 담당의 출력을 버린다.
    assert lab.store.consume(ack_a["event_id"], ack_a["sha256"])["consumed"] is False
    assert (
        lab.store.receive("jetson-a", result(lab, a, 3))["reason"]
        == "not_current_owner"
    )
    assert lab.store.receive("jetson-b", result(lab, b, 2))["decision"] == "accepted"
    assert (
        lab.store.db.execute(
            "SELECT COUNT(*) FROM assignments WHERE state='active'"
        ).fetchone()[0]
        == 1
    )


@pytest.mark.parametrize("kind", ["event", "observation"])
def test_progress_requires_fresh_stored_result_from_active_boundary(lab, kind):
    candidate, _ = prepare(lab, "jetson-a")
    owner, _ = activate(lab, candidate)
    assert lab.store.result_progress("camera_1")["latest_result"] is None
    old = result(lab, owner, 2, kind=kind, activation=False)
    lab.store.receive("jetson-a", old)
    assert lab.store.result_progress("camera_1")["latest_result"] is None
    payload = result(lab, owner, 3, kind=kind)
    lab.clock.now += 0.5
    ack = lab.store.receive("jetson-a", payload)
    progress = lab.store.result_progress("camera_1")
    assert progress["active"]["assignment_id"] == owner["assignment_id"]
    latest = progress["latest_result"]
    assert latest["event_id"] == ack["event_id"]
    assert latest["sha256"] == ack["sha256"]
    assert latest["kind"] == kind and latest["event_sequence"] == 3
    assert latest["capture_to_receive_seconds"] == 0.5
    assert "event" not in latest and "body" not in latest
    lab.clock.now += 11
    assert lab.store.result_progress("camera_1")["latest_result"] is None


def test_progress_does_not_reuse_previous_owner_or_controller_sample(lab):
    candidate, _ = prepare(lab, "jetson-a")
    first, _ = activate(lab, candidate)
    lab.store.receive("jetson-a", result(lab, first, 2))
    candidate, _ = prepare(lab, "jetson-b")
    second, _ = activate(lab, candidate)
    assert lab.store.result_progress("camera_1")["latest_result"] is None
    lab.store.receive("jetson-b", result(lab, second, 2))
    assert lab.store.result_progress("camera_1")["latest_result"]
    lab.store.close()
    lab.store = EdgeOwnershipStore(lab.root, lab.registry, clock=lambda: lab.clock.now)
    assert not lab.store.result_progress("camera_1")["accepting_results"]
    assert lab.store.result_progress("camera_1")["latest_result"] is None


def test_prepare_retry_conflict_and_abort_preserve_epoch(lab):
    candidate, payload = prepare(lab, "jetson-a")
    assert lab.store.prepare("camera_1", payload) == candidate
    with pytest.raises(RegistryError, match="request_id_conflict"):
        lab.store.prepare("camera_1", dict(payload, node_id="jetson-b"))
    with pytest.raises(RegistryError, match="preparation_in_progress"):
        prepare(lab, "jetson-b")
    cancel = {"request_id": "cancel", "controller_boot_id": lab.store.boot_id}
    aborted = lab.store.abort(candidate["assignment_id"], cancel)
    assert aborted["state"] == "aborted"
    assert lab.store.abort(candidate["assignment_id"], cancel) == aborted
    next_candidate, _ = prepare(lab, "jetson-b")
    assert next_candidate["epoch"] > candidate["epoch"]


def test_owner_cas_and_activation_retry(lab):
    candidate, _ = prepare(lab, "jetson-a")
    owner, activate_command = activate(lab, candidate)
    assert lab.store.activate(candidate["assignment_id"], activate_command) == owner
    with pytest.raises(RegistryError, match="owner_changed"):
        prepare(lab, "jetson-b", expected={})


def test_activation_requires_observed_barrier_and_saved_sample(lab):
    candidate, _ = prepare(lab, "jetson-a")
    with pytest.raises(RegistryError, match="central_sample_required"):
        activate(lab, candidate, sample=False)
    lab.store.receive(
        "jetson-a", result(lab, candidate, 1, kind="observation", activation=False)
    )
    service = lab.nodes["jetson-a"]["report"]["service"]
    payload = {
        "request_id": "start",
        "controller_boot_id": lab.store.boot_id,
        "session_id": service["session_id"],
        "activation_id": candidate["activation_id"],
        "sequence_floor": 1,
    }
    service["activation"] = None
    with pytest.raises(RegistryError, match="activation_not_observed"):
        lab.store.activate(candidate["assignment_id"], payload)
    service["activation"] = {
        "activation_id": candidate["activation_id"],
        "sequence_floor": 1,
    }
    lab.clock.now += 11
    service["observed_at"] = _utc(lab.clock.now)
    lab.nodes["jetson-a"]["received_at"] = _utc(lab.clock.now)
    with pytest.raises(RegistryError, match="central_sample_required"):
        lab.store.activate(candidate["assignment_id"], payload)


def test_result_retry_is_durable_and_conflicting_identity_does_not_overwrite(lab):
    assignment, _ = prepare(lab, "jetson-a")
    assignment, _ = activate(lab, assignment)
    original = result(lab, assignment, 2)
    ack = lab.store.receive("jetson-a", original)
    assert lab.store.receipt("jetson-a", ack["event_id"]) == ack
    assert lab.store.receive("jetson-a", original) == ack
    changed = copy.deepcopy(original)
    changed["event"]["confidence"] = 0.1
    with pytest.raises(RegistryError, match="event_id_conflict"):
        lab.store.receive("jetson-a", changed)
    changed = copy.deepcopy(original)
    changed["event"]["event_id"] = str(uuid.uuid4())
    with pytest.raises(RegistryError, match="sequence_conflict"):
        lab.store.receive("jetson-a", changed)
    with pytest.raises(RegistryError, match="receipt_not_found"):
        lab.store.receipt("jetson-b", ack["event_id"])
    assert (
        lab.store.db.execute(
            "SELECT COUNT(*) FROM results WHERE decision='accepted'"
        ).fetchone()[0]
        == 1
    )


def test_storage_failure_returns_no_ack_and_does_not_lose_retry(lab):
    assignment, _ = prepare(lab, "jetson-a")
    assignment, _ = activate(lab, assignment)
    event = result(lab, assignment, 2)
    lab.store.db.execute("PRAGMA query_only=ON")
    with pytest.raises(sqlite3.OperationalError):
        lab.store.receive("jetson-a", event)
    lab.store.db.execute("PRAGMA query_only=OFF")
    with pytest.raises(RegistryError, match="receipt_not_found"):
        lab.store.receipt("jetson-a", event["event"]["event_id"])
    assert lab.store.receive("jetson-a", event)["stored"]


def test_restart_requires_new_activation_and_sample_but_retains_saved_ack(lab):
    candidate, _ = prepare(lab, "jetson-a")
    owner, old_command = activate(lab, candidate)
    saved = result(lab, owner, 2)
    ack = lab.store.receive("jetson-a", saved)
    lab.store.close()
    lab.clock.now += 1
    lab.store = EdgeOwnershipStore(lab.root, lab.registry, clock=lambda: lab.clock.now)
    assert not lab.store.camera("camera_1")["accepting_results"]
    assert lab.store.receive("jetson-a", saved) == ack
    assert lab.store.consume(ack["event_id"], ack["sha256"])["consumed"] is False
    with pytest.raises(RegistryError, match="stale_controller_boot"):
        lab.store.activate(owner["assignment_id"], old_command)
    suspended = lab.store.camera("camera_1")["active"]
    assert suspended["activation_id"] != owner["activation_id"]
    resumed, _ = activate(lab, suspended, sequence_floor=3)
    old_buffer = result(lab, resumed, 4)
    old_buffer["edge"]["activation_id"] = owner["activation_id"]
    assert lab.store.receive("jetson-a", old_buffer)["reason"] == "inactive_execution"
    assert (
        lab.store.receive("jetson-a", result(lab, resumed, 5))["decision"] == "accepted"
    )


def test_expired_result_and_accepted_queue_expiry_do_not_produce_output(lab):
    candidate, _ = prepare(lab, "jetson-a")
    owner, _ = activate(lab, candidate)
    event = result(lab, owner, 2)
    ack = lab.store.receive("jetson-a", event)
    lab.clock.now += 61
    assert lab.store.pending() == []
    assert lab.store.consume(ack["event_id"], ack["sha256"])["reason"] == "expired"
    event["event"]["event_id"] = str(uuid.uuid4())
    event["edge"]["event_sequence"] = 3
    assert lab.store.receive("jetson-a", event)["reason"] == "expired"


def test_activation_boundary_and_wrong_node_never_accepted(lab):
    candidate, _ = prepare(lab, "jetson-a")
    owner, _ = activate(lab, candidate, sequence_floor=10)
    assert (
        lab.store.receive("jetson-a", result(lab, owner, 9))["reason"]
        == "before_activation_boundary"
    )
    event = result(lab, owner, 11)
    with pytest.raises(RegistryError, match="node_identity_mismatch"):
        lab.store.receive("jetson-b", event)
    event["edge"]["node_id"] = "jetson-b"
    assert (
        lab.store.receive("jetson-b", event)["reason"] == "assignment_identity_mismatch"
    )


def test_concurrent_prepare_has_exactly_one_pending_assignment(lab):
    barrier = threading.Barrier(3)
    results = []

    def attempt(node):
        barrier.wait()
        try:
            results.append(prepare(lab, node)[0]["state"])
        except RegistryError as error:
            results.append(error.code)

    threads = [threading.Thread(target=attempt, args=(node,)) for node in lab.nodes]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert sorted(results) == ["preparation_in_progress", "preparing"]


def test_exclusive_controller_and_foreign_database_guard(lab, tmp_path):
    with pytest.raises(BlockingIOError):
        EdgeOwnershipStore(lab.root, lab.registry)
    root = tmp_path / "foreign"
    root.mkdir(mode=0o700)
    database = root / "ownership.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE important(value)")
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not an edge ownership database"):
        EdgeOwnershipStore(root, lab.registry)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_ack_hash_matches_frozen_original_json(lab):
    assignment, _ = prepare(lab, "jetson-a")
    observed(lab, assignment)
    event = result(lab, assignment, 1, activation=False)
    ack = lab.store.receive("jetson-a", event)
    original = json.dumps(
        event, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert ack["sha256"] == hashlib.sha256(original).hexdigest()


def test_http_roles_commit_receipt_and_consumer_revalidation(lab):
    from src.services.edge_controller_api import create_controller_server
    from src.services.edge_node_agent import RegistryClient

    lab.registry.allow_remote_nodes = False
    tokens = {
        name: name[0] * 40 for name in ("reader", "control", "sink", "alpha", "beta")
    }
    with create_controller_server(
        lab.registry,
        lab.store,
        {"jetson-a": tokens["alpha"], "jetson-b": tokens["beta"]},
        tokens["reader"],
        tokens["control"],
        tokens["sink"],
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        clients = {
            name: RegistryClient(f"http://127.0.0.1:{server.server_port}", token)
            for name, token in tokens.items()
        }
        try:
            assert (
                clients["reader"].request("GET", "/v1/cameras/camera_1")[1][
                    "controller_boot_id"
                ]
                == lab.store.boot_id
            )
            payload = {
                "request_id": "prepare-via-http",
                "controller_boot_id": lab.store.boot_id,
                "node_id": "jetson-b",
                "manifest_id": "lab-v1",
                "expected_assignment_id": None,
                "expected_epoch": 0,
                "supervisor_boot_id": lab.nodes["jetson-b"]["report"]["service"][
                    "supervisor_boot_id"
                ],
            }
            assert (
                clients["reader"].request(
                    "POST", "/v1/cameras/camera_1/prepare", payload
                )[0]
                == 401
            )
            status, assignment = clients["control"].request(
                "POST", "/v1/cameras/camera_1/prepare", payload
            )
            assert status == 200 and assignment["state"] == "preparing"
            service = observed(lab, assignment)
            sample = result(lab, assignment, 1, kind="observation", activation=False)
            assert (
                clients["beta"].request("POST", "/v1/nodes/jetson-b/results", sample)[
                    1
                ]["decision"]
                == "observed"
            )
            activation = {
                "request_id": "activate-via-http",
                "controller_boot_id": lab.store.boot_id,
                "session_id": service["session_id"],
                "activation_id": assignment["activation_id"],
                "sequence_floor": 1,
            }
            assert (
                clients["control"].request(
                    "POST",
                    f"/v1/assignments/{assignment['assignment_id']}/activate",
                    activation,
                )[0]
                == 200
            )
            event = result(lab, assignment, 2)
            assert (
                clients["alpha"].request("POST", "/v1/nodes/jetson-b/results", event)[0]
                == 401
            )
            assert (
                clients["alpha"].request("POST", "/v1/nodes/jetson-a/results", event)[0]
                == 403
            )
            status, ack = clients["beta"].request(
                "POST", "/v1/nodes/jetson-b/results", event
            )
            assert status == 200 and ack["stored"]
            status, progress = clients["reader"].request(
                "GET", "/v1/cameras/camera_1/progress"
            )
            assert (
                status == 200
                and progress["latest_result"]["event_id"] == ack["event_id"]
            )
            assert "event" not in progress["latest_result"]
            for role in ("alpha", "beta", "control", "sink"):
                assert (
                    clients[role].request("GET", "/v1/cameras/camera_1/progress")[0]
                    == 401
                )
            assert (
                clients["control"].request(
                    "POST", "/v1/cameras/camera_1/progress", payload
                )[0]
                == 404
            )
            assert (
                clients["beta"].request("POST", "/v1/nodes/jetson-b/results", event)[1]
                == ack
            )
            assert (
                clients["beta"].request(
                    "GET", f"/v1/nodes/jetson-b/receipts/{ack['event_id']}"
                )[1]
                == ack
            )
            assert clients["beta"].request("GET", "/v1/results/pending")[0] == 401
            assert (
                len(clients["sink"].request("GET", "/v1/results/pending")[1]["results"])
                == 1
            )
            lab.clock.now += 61
            endpoint = f"/v1/results/{ack['event_id']}/consume"
            assert (
                clients["reader"].request("POST", endpoint, {"sha256": ack["sha256"]})[
                    0
                ]
                == 401
            )
            assert (
                clients["sink"].request("POST", endpoint, {"sha256": ack["sha256"]})[1][
                    "consumed"
                ]
                is False
            )
        finally:
            server.shutdown()
            thread.join(timeout=3)


def test_pending_response_is_bounded_and_keeps_remaining_rows(lab):
    assignment, _ = prepare(lab, "jetson-a")
    assignment, _ = activate(lab, assignment)
    for sequence in range(2, 12):
        event = result(lab, assignment, sequence)
        event["event"]["details"] = "large-fixture " * 1000
        lab.store.receive("jetson-a", event)
    pending = lab.store.pending()
    assert 0 < len(pending) < 10
    assert len(json.dumps({"results": pending}).encode()) < 65536
    for event in pending:
        assert lab.store.consume(event["event_id"], event["sha256"])["consumed"]
    remaining = lab.store.pending()
    assert remaining and not {item["event_id"] for item in remaining} & {
        item["event_id"] for item in pending
    }
