"""물리 리소스 경합·FIFO·종료 증거와 저장/재시작 경계를 확인한다."""

import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from src.config.event_type_map import event_type_map
from src.services.action_bridge import ActionBridge
from tests.test_action_expiry import TOPIC, accept, advance_to, make_bridge, plan
from tests.test_action_mqtt_intake import wait_until


def make_queue(tmp_path):
    bridge = make_bridge(tmp_path, queue=True)
    config = bridge._mqtt_intake.expiry_policy.config
    known = {rule["event_type"] for rule in config["rules"]}
    config["rules"].extend(
        [
            {**config["rules"][0], "event_type": event_type}
            for event_type in ["head", "zone_dwelling", "person", "not-registered"]
            if event_type not in known
        ]
    )
    return make_bridge(tmp_path, config=config, queue=True)


def enqueue(bridge, event_id, **changes):
    key, raw = accept(bridge, event_id=event_id, **changes)
    bridge._mqtt_intake._process(key)
    return key, raw


def rows(bridge):
    with bridge._mqtt_intake.store._connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM action_device_queue ORDER BY accepted_sequence"
            )
        ]


def queue(bridge):
    return bridge._mqtt_intake.device_queue


def emitted(bridge):
    return [
        call.args[1]["event_id"]
        for call in bridge._executor._publish_edgex_command.call_args_list
    ]


def acknowledge(bridge, row, **changes):
    payload = {
        "request_id": row["request_id"],
        "device_id": row["resource_key"],
        "event_id": row["event_id"],
        "status": "acknowledged",
    }
    payload.update(changes)
    bridge._handle_edgex_result("edgex/results/cctv/" + row["device_type"], payload)


def finish(bridge, row):
    acknowledge(bridge, row, execution_completed=True, resource_idle=True)


def test_priority_uses_server_risk_before_type_and_ignores_payload_rank(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(
        bridge,
        "warning",
        type="head",
        severity="warning",
        priority=-999,
        risk_rank=-999,
    )
    enqueue(bridge, "critical", type="zone_dwelling", severity="critical", priority=999)
    assert emitted(bridge) == []
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["critical"]
    assert [row["risk_rank"] for row in rows(bridge)] == [1, 0]
    finish(bridge, rows(bridge)[1])
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["critical", "warning"]


def test_equal_rank_fifo_does_not_depend_on_event_clock_or_id(tmp_path):
    bridge = make_queue(tmp_path)
    same_time = time.time() - 1
    enqueue(bridge, "z-first", occurred_at=same_time)
    enqueue(bridge, "a-second", occurred_at=same_time)
    for row in rows(bridge):
        queue(bridge).dispatch_ready()
        assert emitted(bridge)[-1] == row["event_id"]
        finish(bridge, row)
    assert emitted(bridge) == ["z-first", "a-second"]


def test_same_camera_memory_cooldown_does_not_discard_queued_event(tmp_path):
    bridge = make_queue(tmp_path)
    bridge._alarm.try_acquire_slot = Mock(return_value=False)
    enqueue(bridge, "one", type="head", severity="warning")
    enqueue(bridge, "two", type="head", severity="warning")
    assert len(rows(bridge)) == 2
    bridge._alarm.try_acquire_slot.assert_not_called()


def test_same_device_across_cameras_has_only_one_owner(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "cam-one", camera_id="cam1")
    enqueue(bridge, "cam-two", camera_id="cam2")
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["cam-one"]
    assert [row["state"] for row in rows(bridge)] == ["sent", "queued"]


def test_busy_device_does_not_block_another_physical_device(tmp_path):
    bridge = make_queue(tmp_path)
    bridge._resolve_edgex_device_ids = lambda device, camera: ["board-" + camera]
    enqueue(bridge, "one", camera_id="cam1")
    enqueue(bridge, "two", camera_id="cam2")
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["one", "two"]
    assert all(row["state"] == "sent" for row in rows(bridge))


@pytest.mark.parametrize(
    "proof",
    [
        {},
        {"execution_completed": True},
        {"resource_idle": True},
        {"execution_completed": "true", "resource_idle": True},
    ],
)
def test_ack_without_both_boolean_completion_proofs_keeps_owner(tmp_path, proof):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    queue(bridge).dispatch_ready()
    acknowledge(bridge, rows(bridge)[0], **proof)
    queue(bridge).dispatch_ready()
    assert rows(bridge)[0]["state"] == "acknowledged"
    assert emitted(bridge) == ["one"]


@pytest.mark.parametrize("status", ["failed", "uncertain", "simulated"])
def test_non_success_result_never_frees_real_resource(tmp_path, status):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    queue(bridge).dispatch_ready()
    acknowledge(
        bridge,
        rows(bridge)[0],
        status=status,
        execution_completed=True,
        resource_idle=True,
    )
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["one"]
    assert rows(bridge)[0]["state"] == "uncertain"


def test_late_publish_return_does_not_revive_completed_owner(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    bridge._executor._publish_edgex_command.side_effect = lambda *args: (
        finish(bridge, rows(bridge)[0]),
        True,
    )[1]
    queue(bridge).dispatch_ready()
    assert rows(bridge)[0]["state"] == "complete"
    acknowledge(bridge, rows(bridge)[0], status="failed")
    assert rows(bridge)[0]["state"] == "complete"


def test_expired_waiter_is_preserved_without_reviving_busy_owner(tmp_path):
    bridge = make_queue(tmp_path)
    key, _ = enqueue(bridge, "one")
    enqueue(bridge, "two")
    queue(bridge).dispatch_ready()
    advance_to(bridge, plan(bridge, key)["deadlines"]["signboard"] + 1)
    queue(bridge).dispatch_ready()
    assert [row["state"] for row in rows(bridge)] == ["sent", "expired"]
    assert emitted(bridge) == ["one"]


def test_registry_delay_after_claim_checks_expiry_again(tmp_path):
    bridge = make_queue(tmp_path)
    key, _ = enqueue(bridge, "one")
    original = queue(bridge)._sync

    def sync(command_id):
        result = original(command_id)
        advance_to(bridge, plan(bridge, key)["deadlines"]["signboard"])
        return result

    queue(bridge)._sync = sync
    queue(bridge).dispatch_ready()
    assert rows(bridge)[0]["state"] == "expired"
    assert emitted(bridge) == []


def test_queued_commands_do_not_start_response_timeout_before_publish(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    with sqlite3.connect(bridge._repo.db_path) as connection:
        status, deadline = connection.execute(
            "SELECT status,response_deadline FROM action_commands"
        ).fetchone()
    assert status == "queued" and deadline is None
    queue(bridge).dispatch_ready()
    with sqlite3.connect(bridge._repo.db_path) as connection:
        status, deadline = connection.execute(
            "SELECT status,response_deadline FROM action_commands"
        ).fetchone()
    assert status == "pending" and deadline is not None


def test_unpublished_queue_resumes_after_restart_without_new_sequence(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "z-first")
    enqueue(bridge, "a-second")
    before = [(r["request_id"], r["accepted_sequence"]) for r in rows(bridge)]
    restored = make_queue(tmp_path)
    restored._mqtt_intake.start()
    try:
        wait_until(lambda: emitted(restored) == ["z-first"])
        assert [
            (r["request_id"], r["accepted_sequence"]) for r in rows(restored)
        ] == before
    finally:
        restored._mqtt_intake.stop()


def test_restart_retains_sent_resource_and_blocks_next_event(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    queue(bridge).dispatch_ready()
    restored = make_queue(tmp_path)
    queue(restored).recover()
    queue(restored).dispatch_ready()
    assert rows(restored)[0]["state"] == "uncertain"
    assert emitted(restored) == []
    finish(restored, rows(restored)[0])
    queue(restored).dispatch_ready()
    assert emitted(restored) == ["two"]


def test_changed_priority_policy_holds_existing_queue(tmp_path, monkeypatch):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    monkeypatch.setitem(event_type_map._routing_levels, "fall_detected", 8)
    queue(bridge).dispatch_ready()
    assert rows(bridge)[0]["reason"] == "priority_policy_changed"
    assert emitted(bridge) == []


def test_duplicate_receipt_does_not_append_or_reorder_queue(tmp_path):
    bridge = make_queue(tmp_path)
    _, raw = enqueue(bridge, "one")
    before = rows(bridge)
    key = bridge._mqtt_intake.accept(TOPIC, raw)
    bridge._mqtt_intake._process(key)
    assert rows(bridge) == before


def test_unknown_type_and_normal_observation_never_enter_queue(tmp_path):
    bridge = make_queue(tmp_path)
    key, _ = enqueue(bridge, "unknown", type="not-registered", severity="critical")
    assert (
        bridge._mqtt_intake.store.get(key)["last_error"] == "held:priority_policy_unset"
    )
    enqueue(bridge, "normal", type="person", severity="normal")
    assert rows(bridge) == [] and emitted(bridge) == []


def test_partial_target_enqueue_rolls_back_all_targets_on_storage_failure(tmp_path):
    bridge = make_queue(tmp_path)
    bridge._resolve_edgex_device_ids = lambda *args: ["board-1", "board-2"]
    with bridge._mqtt_intake.store._connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_second BEFORE INSERT ON action_device_queue WHEN NEW.resource_key='board-2' BEGIN SELECT RAISE(ABORT,'full'); END"
        )
    key, _ = enqueue(bridge, "one")
    assert rows(bridge) == [] and emitted(bridge) == []
    assert bridge._mqtt_intake.store.get(key)["stage"] == "quarantined"


def test_queued_part_of_interrupted_event_is_not_dispatched(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    with bridge._mqtt_intake.store._connect() as connection:
        connection.execute("UPDATE action_execution_windows SET state='uncertain'")
    queue(bridge).dispatch_ready()
    assert rows(bridge)[0]["reason"] == "intake_unconfirmed"
    assert emitted(bridge) == []


def test_result_store_failure_before_publish_keeps_queue_retryable(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    with sqlite3.connect(bridge._repo.db_path) as connection:
        connection.execute("CREATE TRIGGER fail_result BEFORE UPDATE ON action_commands BEGIN SELECT RAISE(ABORT,'full'); END")
    queue(bridge).dispatch_ready()
    assert rows(bridge)[0]["state"] == "queued"
    assert emitted(bridge) == []
    with sqlite3.connect(bridge._repo.db_path) as connection:
        status, deadline = connection.execute("SELECT status,response_deadline FROM action_commands").fetchone()
        assert status == "queued" and deadline is None
        connection.execute("DROP TRIGGER fail_result")
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["one"]


def test_concurrent_claims_and_new_higher_priority_cannot_invert_order(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "low", type="head", severity="warning")
    enqueue(bridge, "high")
    items = rows(bridge)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(queue(bridge)._claim, [r["request_id"] for r in items]))
    assert claimed == [False, True]
    assert [r["state"] for r in rows(bridge)] == ["queued", "dispatching"]


def test_killed_publisher_keeps_resource_occupied_after_restart(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    marker = tmp_path / "published"
    code = """
import sys,time
from pathlib import Path
from tests.test_action_device_queue import make_queue
b=make_queue(Path(sys.argv[1]))
def publish(*args):
 Path(sys.argv[2]).write_text('sent')
 time.sleep(60)
b._executor._publish_edgex_command=publish
b._mqtt_intake.start()
time.sleep(60)
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(marker)])
    try:
        wait_until(marker.exists)
    finally:
        child.kill()
        child.wait(timeout=5)
    restored = make_queue(tmp_path)
    queue(restored).recover()
    queue(restored).dispatch_ready()
    assert rows(restored)[0]["state"] == "uncertain"
    assert emitted(restored) == []


def test_queue_requires_expiry_and_runner_passes_option(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="만료 정책"):
        ActionBridge(db_path=str(tmp_path / "bad.db"), physical_queue_enabled=True)
    from runners import run_action_bridge as runner

    monkeypatch.setenv("ACTION_PHYSICAL_QUEUE_ENABLED", "true")
    config = runner.parse_runtime_config([])
    constructor = Mock()
    monkeypatch.setattr(runner, "ActionBridge", constructor)
    runner.create_action_bridge(config)
    assert constructor.call_args.kwargs["physical_queue_enabled"] is True


def test_same_risk_orders_by_type_before_fifo(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "zone-first", type="zone_dwelling", severity="critical")
    enqueue(bridge, "fall-second", type="fall_detected", severity="critical")
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["fall-second"]


def test_higher_priority_waits_for_active_output_then_runs_next(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "active", type="head", severity="warning")
    queue(bridge).dispatch_ready()
    enqueue(bridge, "low", type="head", severity="warning")
    enqueue(bridge, "high", severity="critical")
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["active"]
    finish(bridge, rows(bridge)[0])
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["active", "high"]


def test_one_busy_target_does_not_delay_free_target_of_same_command(tmp_path):
    bridge = make_queue(tmp_path)
    bridge._resolve_edgex_device_ids = lambda device, camera: (
        ["board-1"] if camera == "cam1" else ["board-1", "board-2"]
    )
    enqueue(bridge, "active", camera_id="cam1")
    queue(bridge).dispatch_ready()
    enqueue(bridge, "both", camera_id="cam2")
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["active", "both"]
    assert [(r["resource_key"], r["state"]) for r in rows(bridge)[1:]] == [
        ("board-1", "queued"),
        ("board-2", "sent"),
    ]


@pytest.mark.parametrize(
    "field,value", [("request_id", "bad"), ("event_id", "bad"), ("device_id", "bad")]
)
def test_mismatched_completion_cannot_release_owner(tmp_path, field, value):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    queue(bridge).dispatch_ready()
    acknowledge(
        bridge,
        rows(bridge)[0],
        execution_completed=True,
        resource_idle=True,
        **{field: value},
    )
    assert rows(bridge)[0]["state"] == "sent"


def test_failed_queue_claim_cannot_publish(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    with bridge._mqtt_intake.store._connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_claim BEFORE UPDATE ON action_device_queue WHEN NEW.state='dispatching' BEGIN SELECT RAISE(ABORT,'full'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        queue(bridge).dispatch_ready()
    assert emitted(bridge) == [] and rows(bridge)[0]["state"] == "queued"


def test_post_publish_commit_failure_never_replays_or_frees_owner(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    with bridge._mqtt_intake.store._connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_sent BEFORE UPDATE ON action_device_queue WHEN NEW.state='sent' BEGIN SELECT RAISE(ABORT,'full'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["one"]
    restored = make_queue(tmp_path)
    queue(restored).recover()
    queue(restored).dispatch_ready()
    assert rows(restored)[0]["state"] == "uncertain"
    assert emitted(restored) == []


@pytest.mark.parametrize("result", [False, RuntimeError("send failed")])
def test_unconfirmed_publish_keeps_resource_busy(tmp_path, result):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    publish = bridge._executor._publish_edgex_command
    if isinstance(result, Exception):
        publish.side_effect = result
    else:
        publish.return_value = result
    queue(bridge).dispatch_ready()
    publish.assert_called_once()
    assert rows(bridge)[0]["state"] == "uncertain"


def test_disabling_queue_option_cannot_bypass_persisted_owner(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    queue(bridge).dispatch_ready()
    disabled = make_bridge(tmp_path, queue=False)
    with pytest.raises(RuntimeError, match="큐 옵션"):
        disabled._mqtt_intake.start()
    assert disabled._mqtt_intake._lease is None


def test_timeout_does_not_release_resource_but_late_completion_does(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    queue(bridge).dispatch_ready()
    with sqlite3.connect(bridge._repo.db_path) as connection:
        connection.execute(
            "UPDATE action_commands SET response_deadline=1 WHERE command_id='one:signboard'"
        )
    bridge._expire_command_responses()
    assert rows(bridge)[0]["state"] == "uncertain"
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["one"]
    finish(bridge, rows(bridge)[0])
    queue(bridge).dispatch_ready()
    assert emitted(bridge) == ["one", "two"]
