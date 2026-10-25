"""실제 담당 저장소와 모형 보고로 자동 판단·실행·재개 경계를 검사한다."""

import copy
import sqlite3
import uuid

import pytest

from src.services.edge_auto_handoff import EdgeAutoHandoff, validate_policy
from src.services.edge_orchestrator import RegistryError, _utc
from tests.test_edge_handoff import Steps
from tests.test_edge_handoff import handoff_lab as handoff_lab
from tests.test_edge_ownership import activate, prepare
from tests.test_edge_ownership import lab as lab


@pytest.fixture
def automatic(handoff_lab, tmp_path):
    env = handoff_lab
    env.monotonic = 0.0
    env.policy = {
        "camera_id": "camera_1",
        "manifests": {"jetson-a": "lab-v1", "jetson-b": "lab-v1"},
        "overload_percent": {
            "cpu_percent": 85,
            "gpu_percent": 90,
            "memory_percent": 85,
        },
        "available_percent": {
            "cpu_percent": 60,
            "gpu_percent": 60,
            "memory_percent": 70,
        },
        "sustain_seconds": 15,
        "min_samples": 3,
        "max_report_age_seconds": 5,
        "cooldown_seconds": 60,
        "poll_seconds": 1,
        "handoff_timeout_seconds": 30,
    }
    env.resources = {
        "jetson-a": {"cpu_percent": 95, "gpu_percent": 20, "memory_percent": 30},
        "jetson-b": {"cpu_percent": 10, "gpu_percent": 10, "memory_percent": 20},
    }
    for node in env.lab.nodes.values():
        node["boot_id"] = str(uuid.uuid4())
        node["last_sequence"] = 0
    env.created = []

    def refresh(seconds=0, names=None):
        env.lab.clock.now += seconds
        env.monotonic += seconds
        for name in names or env.lab.nodes:
            node = env.lab.nodes[name]
            node["last_sequence"] += 1
            node["received_at"] = _utc(env.lab.clock.now)
            node["report"].update(
                simulation=False,
                boot_id=node["boot_id"],
                sequence=node["last_sequence"],
                sampled_at=_utc(env.lab.clock.now),
                resources=copy.deepcopy(env.resources[name]),
            )
            node["report"]["service"]["observed_at"] = _utc(env.lab.clock.now)

    def make(policy=None):
        manual = env.make(tmp_path / "peers")
        try:
            task = EdgeAutoHandoff(
                tmp_path / "automatic",
                manual.controller,
                manual.nodes,
                policy or env.policy,
                config_id="test-peers",
                clock=lambda: env.lab.clock.now,
                monotonic=lambda: env.monotonic,
            )
        finally:
            manual.close()
        env.created.append(task)
        return task

    env.refresh, env.make_auto = refresh, make
    refresh()
    yield env
    for task in env.created:
        task.close()


def warmup(env, task):
    assert task.step(Steps())["status"] == "waiting"
    env.refresh(5)
    assert task.step(Steps())["status"] == "waiting"
    env.refresh(5)
    assert task.step(Steps())["status"] == "waiting"
    env.refresh(5)


def test_sustained_overload_uses_existing_handoff_and_no_healthy_failback(automatic):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    outcome = task.step(Steps())
    assert outcome["status"] == "complete"
    assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-b"
    decision = task.decision
    assert decision["reasons"] == ["cpu_percent"]
    assert decision["sustained_seconds"] == {"cpu_percent": 15}
    assert decision["outcome"]["previous_stop"] == "stopped"
    assert task.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert task.step(Steps())["reason"] == "cooldown"
    env.resources["jetson-a"]["cpu_percent"] = 10
    env.refresh(61)
    assert task.step(Steps())["reason"] == "below_threshold"
    assert task.decision["decision_id"] == decision["decision_id"]


def test_duplicate_and_stale_reports_do_not_prove_duration(automatic):
    env = automatic
    task = env.make_auto()
    task.step(Steps())
    for _ in range(20):
        env.monotonic += 1
        assert task.step(Steps())["reason"] == "waiting_for_new_samples"
    assert task.decision is None
    env.lab.clock.now += 6
    assert task.step(Steps())["reason"] == "report_not_fresh"
    env.refresh()
    assert task.step(Steps())["reason"] == "collecting_overload_samples"
    assert task.decision is None


def test_wall_clock_jump_does_not_complete_monotonic_window(automatic):
    env = automatic
    task = env.make_auto()
    task.step(Steps())
    env.lab.clock.now += 10000
    env.refresh(1)
    task.step(Steps())
    env.refresh(1)
    assert task.step(Steps())["reason"] == "collecting_overload_samples"
    assert task.decision is None


def test_monitor_gap_restarts_sustained_window(automatic):
    env = automatic
    task = env.make_auto()
    task.step(Steps())
    env.refresh(5)
    task.step(Steps())
    env.refresh(10)
    assert task.step(Steps())["reason"] == "collecting_overload_samples"
    assert task.decision is None


def test_expired_uncommitted_decision_is_not_executed_after_restart(automatic):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    assert task.step(Steps(0))["status"] == "executing"
    task.close()
    env.refresh(10)
    task = env.make_auto()
    env.calls.clear()
    outcome = task.step(Steps())
    assert outcome["status"] == "failed"
    assert outcome["outcome"]["last_error"] == "decision_expired"
    assert env.calls == []


def test_short_spike_and_boot_change_reset_window(automatic):
    env = automatic
    task = env.make_auto()
    task.step(Steps())
    env.resources["jetson-a"]["cpu_percent"] = 40
    env.refresh(10)
    assert task.step(Steps())["reason"] == "below_threshold"
    env.resources["jetson-a"]["cpu_percent"] = 95
    env.refresh()
    task.step(Steps())
    env.refresh(5)
    task.step(Steps())
    env.lab.nodes["jetson-b"]["boot_id"] = str(uuid.uuid4())
    env.refresh(10)
    assert task.step(Steps())["reason"] == "collecting_overload_samples"
    assert task.decision is None


def test_metric_recovery_resets_even_when_other_node_repeats_sample(automatic):
    env = automatic
    task = env.make_auto()
    task.step(Steps())
    env.refresh(5)
    task.step(Steps())
    env.resources["jetson-a"].update(cpu_percent=40, gpu_percent=95)
    env.refresh(1, ["jetson-a"])
    task.step(Steps())
    env.resources["jetson-a"].update(cpu_percent=95, gpu_percent=10)
    env.refresh(4)
    task.step(Steps())
    env.refresh(5)
    assert task.step(Steps())["reason"] == "collecting_overload_samples"
    assert task.decision is None


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            lambda env: env.lab.nodes["jetson-a"].update(freshness="stale"),
            "report_not_fresh",
        ),
        (
            lambda env: env.lab.nodes["jetson-b"].update(service_freshness="stale"),
            "report_not_fresh",
        ),
        (
            lambda env: env.lab.nodes["jetson-b"]["report"]["resources"].update(
                gpu_percent=None
            ),
            "resources_missing",
        ),
        (
            lambda env: env.lab.nodes["jetson-b"]["report"]["resources"].update(
                cpu_percent=95
            ),
            "candidate_busy",
        ),
        (
            lambda env: env.lab.nodes["jetson-b"]["report"]["service"].update(
                manifest_id="other"
            ),
            "candidate_unavailable",
        ),
        (
            lambda env: env.lab.nodes["jetson-a"]["report"]["service"].update(
                session_id="other"
            ),
            "owner_execution_changed",
        ),
        (
            lambda env: env.lab.nodes["jetson-b"]["report"].update(simulation=True),
            "report_not_fresh",
        ),
    ],
)
def test_unusable_observations_never_send_commands(automatic, change, reason):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    change(env)
    assert task.step(Steps())["reason"] == reason
    assert task.decision is None
    assert all(key[1] == "GET" for key, _ in env.calls)


def test_interrupted_execution_resumes_same_durable_decision(automatic):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    assert task.step(Steps(2))["status"] == "executing"
    decision_id = task.decision["decision_id"]
    task.close()
    env.refresh(10)
    task = env.make_auto()
    assert task.step(Steps())["status"] == "complete"
    assert task.decision["decision_id"] == decision_id
    assert task.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    task.close()
    env.lab.clock.now += 100000
    task = env.make_auto()
    assert task.step(Steps())["reason"] == "cooldown"


def test_restarted_controller_blocks_new_decisions(automatic):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    task.step(Steps(2))
    env.lab.store.boot_id = str(uuid.uuid4())
    assert task.step(Steps())["status"] == "reconcile_required"
    count = len(env.calls)
    env.refresh(90)
    assert task.step(Steps())["status"] == "reconcile_required"
    assert len(env.calls) == count


def test_manual_owner_change_between_decision_and_execution_is_not_overwritten(
    automatic,
):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    original_execute = task._execute

    def changed(stop):
        candidate, _ = prepare(env.lab, "jetson-b")
        activate(env.lab, candidate)
        env.calls.clear()
        return original_execute(stop)

    task._execute = changed
    outcome = task.step(Steps())
    assert outcome["status"] == "failed"
    assert outcome["outcome"]["last_error"] == "decision_owner_changed"
    assert all(key[1] == "GET" for key, _ in env.calls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "changed"),
        ("status", "stopped"),
        ("supervisor_boot_id", "changed"),
    ],
)
def test_source_execution_change_after_decision_prevents_preparation(
    automatic, field, value
):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    execute = task._execute

    def changed(stop):
        env.lab.nodes["jetson-a"]["report"]["service"][field] = value
        return execute(stop)

    task._execute = changed
    outcome = task.step(Steps())
    assert outcome["status"] == "failed"
    assert outcome["outcome"]["last_error"] == "decision_execution_changed"
    assert all(key[1] == "GET" for key, _ in env.calls)


def test_backwards_wall_clock_cannot_extend_uncommitted_decision(automatic):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    task.step(Steps(0))
    env.lab.clock.now -= 1
    env.calls.clear()
    outcome = task.step(Steps())
    assert outcome["status"] == "failed"
    assert outcome["outcome"]["last_error"] == "decision_expired"
    assert env.calls == []


def test_failed_candidate_keeps_previous_owner_and_cools_down(automatic):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    env.fail_start = True
    assert task.step(Steps())["status"] == "aborted"
    assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-a"
    assert task.step(Steps())["reason"] == "cooldown"


def test_decision_storage_failure_prevents_external_writes(automatic):
    env = automatic
    task = env.make_auto()
    warmup(env, task)
    task.db.execute("PRAGMA query_only=ON")
    with pytest.raises(sqlite3.OperationalError):
        task.step(Steps())
    assert all(key[1] == "GET" for key, _ in env.calls)
    with pytest.raises(RuntimeError, match="storage unavailable"):
        task.step(Steps())


def test_lock_and_policy_identity_prevent_second_runner(automatic):
    env = automatic
    task = env.make_auto()
    with pytest.raises(BlockingIOError):
        env.make_auto()
    task.close()
    changed = copy.deepcopy(env.policy)
    changed["sustain_seconds"] = 20
    with pytest.raises(ValueError, match="another configuration"):
        env.make_auto(changed)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(min_samples=True),
        lambda p: p.update(sustain_seconds=float("nan")),
        lambda p: p.update(cooldown_seconds=0),
        lambda p: p.update(poll_seconds=6),
        lambda p: p["available_percent"].update(cpu_percent=85),
        lambda p: p["overload_percent"].update(gpu_percent=None),
        lambda p: p["manifests"].pop("jetson-b"),
    ],
)
def test_policy_rejects_unsafe_or_ambiguous_values(automatic, change):
    policy = copy.deepcopy(automatic.policy)
    change(policy)
    with pytest.raises((ValueError, RegistryError)):
        validate_policy(policy)



def enable_failback(env):
    env.policy["failback"] = {"preferred_node": "jetson-a", "sustain_seconds": 30}
    task = env.make_auto()
    warmup(env, task)
    assert task.step(Steps())["status"] == "complete"
    env.resources["jetson-a"]["cpu_percent"] = 10
    env.refresh(61)
    return task


def test_optional_failback_returns_to_preferred_node_after_sustained_recovery(automatic):
    env = automatic
    task = enable_failback(env)
    assert task.step(Steps())["reason"] == "collecting_recovery_samples"
    for _ in range(5):
        env.refresh(5)
        assert task.step(Steps())["reason"] == "collecting_recovery_samples"
    env.refresh(5)
    assert task.step(Steps())["status"] == "complete"
    assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-a"
    assert task.decision["reasons"] == ["preferred_node_recovered"]
    assert task.decision["sustained_seconds"] == {"preferred_node_recovered": 30}
    assert task.decision["outcome"]["previous_stop"] == "stopped"
    assert task.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 2
    assert task.step(Steps())["reason"] == "cooldown"
    env.refresh(61)
    assert task.step(Steps())["reason"] == "below_threshold"


def test_failback_resets_when_recovered_target_becomes_busy(automatic):
    env = automatic
    task = enable_failback(env)
    task.step(Steps())
    for _ in range(4):
        env.refresh(5)
        task.step(Steps())
    env.resources["jetson-a"]["cpu_percent"] = 65
    env.refresh(5)
    assert task.step(Steps())["reason"] == "candidate_busy"
    env.resources["jetson-a"]["cpu_percent"] = 10
    env.refresh(5)
    assert task.step(Steps())["reason"] == "collecting_recovery_samples"
    assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-b"


def test_failback_duplicate_reports_and_restart_do_not_inherit_recovery_time(automatic):
    env = automatic
    task = enable_failback(env)
    task.step(Steps())
    for _ in range(7):
        env.monotonic += 5
        assert task.step(Steps())["reason"] == "waiting_for_new_samples"
    assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-b"
    task.close()
    task = env.make_auto()
    assert task.step(Steps())["reason"] == "cooldown"
    env.refresh(61)
    assert task.step(Steps())["reason"] == "collecting_recovery_samples"
    env.refresh(6)
    assert task.step(Steps())["reason"] == "collecting_recovery_samples"


@pytest.mark.parametrize("change,reason", [
    (lambda env: env.lab.nodes["jetson-a"].update(freshness="stale"), "report_not_fresh"),
    (lambda env: env.resources["jetson-a"].update(gpu_percent=None), "resources_missing"),
])
def test_failback_requires_fresh_complete_target_resources(automatic, change, reason):
    env = automatic
    task = enable_failback(env)
    change(env)
    env.refresh()
    assert task.step(Steps())["reason"] == reason
    assert env.lab.store.camera("camera_1")["active"]["node_id"] == "jetson-b"


@pytest.mark.parametrize("failback", [
    None, {}, {"preferred_node": "unknown", "sustain_seconds": 30},
    {"preferred_node": "jetson-a", "sustain_seconds": 0},
    {"preferred_node": "jetson-a", "sustain_seconds": True},
    {"preferred_node": "jetson-a", "sustain_seconds": float("nan")},
    {"preferred_node": "jetson-a", "sustain_seconds": 601},
    {"preferred_node": "jetson-a", "sustain_seconds": 30, "extra": 1},
])
def test_failback_policy_rejects_unsafe_values(automatic, failback):
    policy = copy.deepcopy(automatic.policy)
    policy["failback"] = failback
    with pytest.raises((ValueError, RegistryError)):
        validate_policy(policy)
