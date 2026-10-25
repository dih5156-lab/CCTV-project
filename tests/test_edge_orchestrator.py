"""모형 제어기의 영속성·세대·보고 순서와 입력 경계를 검증한다."""

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.services.edge_orchestrator import NodeRegistry, RegistryError


def registration(node_id="mock-a", boot_id="boot-1"):
    return {
        "node_id": node_id,
        "node_type": "AEC-100",
        "boot_id": boot_id,
        "ip": "127.0.0.1",
        "capabilities": ["status-report"],
        "simulation": True,
    }


def report(sequence=1, boot_id="boot-1", sampled_at=1700000000.0):
    return {
        "boot_id": boot_id,
        "sequence": sequence,
        "simulation": True,
        "sampled_at": datetime.fromtimestamp(sampled_at, timezone.utc).isoformat(),
        "resources": {
            "cpu_percent": 42,
            "memory_percent": 50,
            "gpu_percent": None,
            "network_rx_bytes_per_second": 1000,
            "network_tx_bytes_per_second": 200,
        },
        "service": {
            "service_id": "mock-ai",
            "status": "running",
            "uptime_seconds": 2,
            "inference_fps": 15,
            "model_id": "mock-v1",
        },
    }


@pytest.fixture
def registry(tmp_path):
    clock = SimpleNamespace(now=1700000000.0)
    return NodeRegistry(
        tmp_path / "nodes.sqlite3", stale_after=10, clock=lambda: clock.now
    ), clock


def test_two_nodes_and_report_ack_are_persisted(registry):
    store, clock = registry
    store.register(registration())
    store.register(registration("mock-b"))
    assert [n["freshness"] for n in store.list_nodes()] == ["pending", "pending"]
    assert store.report("mock-a", report()) == {"sequence": 1, "duplicate": False}
    reopened = NodeRegistry(store.path, clock=lambda: clock.now)
    nodes = reopened.list_nodes()
    assert [n["freshness"] for n in nodes] == ["online", "pending"]
    assert nodes[0]["report"]["resources"]["gpu_percent"] is None
    assert nodes[0]["report_count"] == nodes[0]["last_sequence"] == 1
    assert nodes[0]["simulation"] is True


def test_duplicate_report_and_registration_do_not_refresh_liveness(registry):
    store, clock = registry
    store.register(registration())
    store.report("mock-a", report())
    clock.now += 11
    assert store.register(registration())["last_sequence"] == 1
    assert store.report("mock-a", report())["duplicate"] is True
    node = store.list_nodes()[0]
    assert node["freshness"] == "stale" and node["report_count"] == 1
    store.report("mock-a", report(2, sampled_at=clock.now))
    assert store.list_nodes()[0]["freshness"] == "online"


def test_new_boot_fences_old_reports_and_old_registration_after_restart(registry):
    store, clock = registry
    store.register(registration())
    store.report("mock-a", report(9))
    store.register(registration(boot_id="boot-2"))
    store = NodeRegistry(store.path, clock=lambda: clock.now)
    for action in (
        lambda: store.register(registration()),
        lambda: store.report("mock-a", report(10)),
    ):
        with pytest.raises(RegistryError, match="retired_boot"):
            action()
    assert store.list_nodes()[0]["freshness"] == "pending"
    store.report("mock-a", report(1, "boot-2"))
    assert store.list_nodes()[0]["last_sequence"] == 1


def test_stale_conflicting_and_regressing_reports_leave_state_intact(registry):
    store, clock = registry
    store.register(registration())
    store.report("mock-a", report(2))
    changed = report(2)
    changed["resources"]["cpu_percent"] = 1
    for candidate in (report(1), changed, report(3, sampled_at=clock.now - 1)):
        with pytest.raises(RegistryError) as exc:
            store.report("mock-a", candidate)
        assert exc.value.status == 409
    assert store.list_nodes()[0]["last_sequence"] == 2
    assert store.list_nodes()[0]["report_count"] == 1


def test_old_sample_and_clock_rollback_are_not_online(registry):
    store, clock = registry
    store.register(registration())
    store.report("mock-a", report(sampled_at=clock.now - 20))
    assert store.list_nodes()[0]["freshness"] == "stale"
    store.report("mock-a", report(2, sampled_at=clock.now))
    clock.now -= 1
    assert store.list_nodes()[0]["freshness"] == "stale"


def test_concurrent_duplicate_report_commits_once(registry):
    store, _ = registry
    store.register(registration())
    other = NodeRegistry(store.path, clock=store.clock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda s: s.report("mock-a", report()), (store, other)))
    assert sum(r["duplicate"] for r in replies) == 1
    assert store.list_nodes()[0]["report_count"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_id", "../other"),
        ("boot_id", ""),
        ("simulation", False),
        ("ip", "192.168.2.141"),
        ("capabilities", []),
        ("capabilities", ["x", "x"]),
    ],
)
def test_invalid_registration_is_not_saved(registry, field, value):
    store, _ = registry
    payload = registration()
    payload[field] = value
    with pytest.raises(RegistryError):
        store.register(payload)
    assert store.list_nodes() == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence", True),
        ("sequence", 0),
        ("sequence", 2**63),
        ("simulation", False),
        ("sampled_at", "2026-01-01T00:00:00"),
        ("sampled_at", "2030-01-01T00:00:00Z"),
        ("resources", []),
        ("service", None),
    ],
)
def test_invalid_report_is_not_saved(registry, field, value):
    store, _ = registry
    store.register(registration())
    payload = report()
    payload[field] = value
    with pytest.raises(RegistryError):
        store.report("mock-a", payload)
    assert store.list_nodes()[0]["report_count"] == 0


@pytest.mark.parametrize("number", [True, -1, 101, float("nan"), float("inf"), 10**500])
def test_bad_resource_numbers_are_rejected(registry, number):
    store, _ = registry
    store.register(registration())
    payload = report()
    payload["resources"]["cpu_percent"] = number
    with pytest.raises(RegistryError, match="invalid_number"):
        store.report("mock-a", payload)


def test_registration_conflict_and_unregistered_report(registry):
    store, _ = registry
    with pytest.raises(RegistryError, match="node_not_registered"):
        store.report("mock-a", report())
    payload = registration()
    store.register(payload)
    changed = copy.deepcopy(payload)
    changed["capabilities"].append("other")
    with pytest.raises(RegistryError, match="registration_conflict"):
        store.register(changed)
    assert store.list_nodes()[0]["capabilities"] == payload["capabilities"]


def test_foreign_database_is_never_modified(tmp_path):
    path = tmp_path / "existing.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE existing (value TEXT)")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not an edge node registry"):
        NodeRegistry(path)
    assert path.read_bytes() == before


def test_database_contains_no_auth_tokens(registry):
    store, _ = registry
    payload = registration()
    payload["token"] = "must-not-be-stored"
    with pytest.raises(RegistryError, match="invalid_fields"):
        store.register(payload)
    assert "must-not-be-stored" not in json.dumps(store.list_nodes())


def test_real_resource_mode_is_explicit_and_survives_reopen(registry):
    store, clock = registry
    payload = registration()
    payload["simulation"] = False
    with pytest.raises(RegistryError, match="simulation_required"):
        store.register(payload)
    store = NodeRegistry(store.path, clock=lambda: clock.now, allow_real_resources=True)
    store.register(payload)
    sample = report()
    sample.update(simulation=False, service=None)
    store.report("mock-a", sample)
    reopened = NodeRegistry(
        store.path, clock=lambda: clock.now, allow_real_resources=True
    )
    node = reopened.list_nodes()[0]
    assert node["simulation"] is False and node["report"]["service"] is None
    assert node["report_count"] == 1 and node["freshness"] == "online"


@pytest.mark.parametrize("simulation", [True, False])
def test_registered_mode_cannot_change_in_report(registry, simulation):
    store, clock = registry
    store = NodeRegistry(store.path, clock=lambda: clock.now, allow_real_resources=True)
    payload = registration()
    payload["simulation"] = simulation
    store.register(payload)
    sample = report()
    sample["simulation"] = not simulation
    if not sample["simulation"]:
        sample["service"] = None
    with pytest.raises(RegistryError, match="report_mode_conflict"):
        store.report("mock-a", sample)
    assert store.list_nodes()[0]["report_count"] == 0


def test_real_resources_cannot_claim_unobserved_ai_status(registry):
    store, clock = registry
    store = NodeRegistry(store.path, clock=lambda: clock.now, allow_real_resources=True)
    payload = registration()
    payload["simulation"] = False
    store.register(payload)
    sample = report()
    sample["simulation"] = False
    with pytest.raises(RegistryError, match="unobserved_service_required"):
        store.report("mock-a", sample)


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_real_mode_still_requires_boolean_flags(registry, value):
    store, clock = registry
    store = NodeRegistry(store.path, clock=lambda: clock.now, allow_real_resources=True)
    payload = registration()
    payload["simulation"] = value
    with pytest.raises(RegistryError, match="simulation_required"):
        store.register(payload)
    store.register(registration())
    sample = report()
    sample["simulation"] = value
    with pytest.raises(RegistryError, match="simulation_required"):
        store.report("mock-a", sample)
