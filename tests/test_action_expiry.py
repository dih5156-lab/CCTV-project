"""경보 기한 고정·재시작 선점·발행 직전 검사를 실제 SQLite 경계에서 확인한다."""

import json
import sqlite3
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from src.services.action_bridge import ActionBridge, AlarmDevice, ControlMode
from tests.test_action_mqtt_intake import wait_until

TOPIC = "cctv/ai/events/cam1/fall"


def make_bridge(tmp_path, *, capture=False, config=None, queue=False):
    path = tmp_path / "expiry.json"
    if config is not None or not path.exists():
        path.write_text(
            json.dumps(
                config
                or {
                    "schema_version": 1,
                    "policy_version": "test-only",
                    "clock_verified": True,
                    "rules": [
                        {
                            "topic": TOPIC,
                            "event_type": "fall_detected",
                            "time_basis": "occurred_at",
                            "source_time_verified": True,
                            "ttl_seconds": {"signboard": 10},
                        }
                    ],
                }
            )
        )
    bridge = ActionBridge(
        db_path=str(tmp_path / "events.db"),
        mqtt_intake_db=str(tmp_path / "input.db"),
        mqtt_client_id="expiry-test",
        mqtt_capture_only=capture,
        edgex_command_mode="edgex",
        action_expiry_policy_path=str(path),
        physical_queue_enabled=queue,
        default_mode=ControlMode.AUTO,
    )
    bridge._repo.init()
    bridge._sites.resolve_alarm_devices = lambda camera: {AlarmDevice.SIGNBOARD}
    bridge._resolve_edgex_device_ids = lambda device, camera: ["board-1"]
    bridge._resolve_devices = lambda *args, **kwargs: {AlarmDevice.SIGNBOARD}
    bridge._executor._resolve_edgex_device_ids = bridge._resolve_edgex_device_ids
    bridge._executor._publish_edgex_command = Mock(return_value=True)
    bridge._publish_status = Mock()
    bridge._executor._publish_status = bridge._publish_status
    return bridge


def accept(bridge, **changes):
    payload = {
        "event_id": "event-1",
        "camera_id": "cam1",
        "type": "fall_detected",
        "severity": "critical",
        "occurred_at": time.time() - 1,
    }
    payload.update(changes)
    raw = json.dumps(payload).encode()
    key = bridge._mqtt_intake.accept(TOPIC, raw)
    return key, raw


def plan(bridge, key):
    return json.loads(bridge._mqtt_intake.store.get(key)["projection_json"])


def advance_to(bridge, value):
    policy = bridge._mqtt_intake.expiry_policy
    policy.clock = lambda: value
    policy.monotonic = lambda: policy._monotonic + value - policy._wall


def window(bridge):
    with bridge._mqtt_intake.store._connect() as connection:
        return dict(
            connection.execute(
                "SELECT * FROM action_execution_windows WHERE event_id='event-1'"
            ).fetchone()
        )


@pytest.mark.parametrize("offset", [0, 0.001, 3600])
def test_expiry_boundary_preserves_raw_and_never_dispatches(tmp_path, offset):
    bridge = make_bridge(tmp_path)
    key, raw = accept(bridge)
    advance_to(bridge, plan(bridge, key)["deadlines"]["signboard"] + offset)
    bridge._mqtt_intake._process(key)
    receipt = bridge._mqtt_intake.store.get(key)
    assert receipt["last_error"] == "expired:expired"
    assert window(bridge)["state"] == "expired"
    assert json.loads(receipt["raw_payload"])["payload_base64"]
    bridge._executor._publish_edgex_command.assert_not_called()


@pytest.mark.parametrize(
    "occurred", [None, "bad", "2026-10-13T10:00:00", True, -1, float("inf")]
)
def test_unverified_event_time_is_held(tmp_path, occurred):
    bridge = make_bridge(tmp_path)
    key, _ = accept(bridge, occurred_at=occurred)
    bridge._mqtt_intake._process(key)
    assert bridge._mqtt_intake.store.get(key)["last_error"] in {
        "held:time_unverified",
        "held:invalid_payload",
    }
    bridge._executor._publish_edgex_command.assert_not_called()


def test_future_time_is_not_replaced_with_receive_time(tmp_path):
    bridge = make_bridge(tmp_path)
    key, _ = accept(bridge, occurred_at=time.time() + 20)
    bridge._mqtt_intake._process(key)
    assert bridge._mqtt_intake.store.get(key)["last_error"] == "held:time_unverified"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda c: c.update(rules=[]), "policy_unset"),
        (lambda c: c.update(clock_verified=False), "time_unverified"),
        (lambda c: c["rules"][0].update(source_time_verified=False), "time_unverified"),
    ],
)
def test_missing_policy_or_trust_is_held(tmp_path, mutation, reason):
    bridge = make_bridge(tmp_path)
    config = bridge._mqtt_intake.expiry_policy.config
    mutation(config)
    bridge = make_bridge(tmp_path, config=config)
    key, _ = accept(bridge)
    bridge._mqtt_intake._process(key)
    assert bridge._mqtt_intake.store.get(key)["last_error"] == "held:" + reason


def test_received_time_policy_keeps_first_deadline_on_redelivery(tmp_path):
    bridge = make_bridge(tmp_path)
    config = bridge._mqtt_intake.expiry_policy.config
    config["rules"][0]["time_basis"] = "received_at"
    bridge = make_bridge(tmp_path, config=config)
    key, raw = accept(bridge, occurred_at=None)
    first = plan(bridge, key)
    repeated = bridge._mqtt_intake.accept(TOPIC, raw)
    assert plan(bridge, repeated)["deadlines"] == first["deadlines"]
    advance_to(bridge, first["deadlines"]["signboard"])
    bridge._mqtt_intake._process(repeated)
    bridge._executor._publish_edgex_command.assert_not_called()


def test_unpublished_valid_event_resumes_once_after_restart(tmp_path):
    bridge = make_bridge(tmp_path, capture=True)
    bridge._mqtt_intake.start()
    key, raw = accept(bridge)
    frozen = plan(bridge, key)
    bridge._mqtt_intake.stop()
    restored = make_bridge(tmp_path)
    restored._mqtt_intake.start()
    try:
        wait_until(lambda: restored._mqtt_intake.store.get(key)["stage"] == "complete")
        assert plan(restored, key)["deadlines"] == frozen["deadlines"]
        restored._executor._publish_edgex_command.assert_called_once()
        repeated = restored._mqtt_intake.accept(TOPIC, raw)
        wait_until(
            lambda: restored._mqtt_intake.store.get(repeated)["stage"] == "quarantined"
        )
        assert (
            restored._mqtt_intake.store.get(repeated)["last_error"]
            == "held:duplicate_completed"
        )
        restored._executor._publish_edgex_command.assert_called_once()
    finally:
        restored._mqtt_intake.stop()


def test_expired_backlog_does_not_resume(tmp_path):
    bridge = make_bridge(tmp_path, capture=True)
    bridge._mqtt_intake.start()
    key, _ = accept(bridge, occurred_at=time.time() - 30)
    assert window(bridge)["state"] == "expired"
    bridge._mqtt_intake.stop()
    restored = make_bridge(tmp_path)
    restored._mqtt_intake.start()
    try:
        wait_until(
            lambda: restored._mqtt_intake.store.get(key)["stage"] == "quarantined"
        )
        assert restored._mqtt_intake.store.get(key)["last_error"] == "expired:expired"
        restored._executor._publish_edgex_command.assert_not_called()
    finally:
        restored._mqtt_intake.stop()


@pytest.mark.parametrize(
    "change,reason",
    [
        ("policy", "policy_changed"),
        ("target", "target_changed"),
        ("mode", "manual_approval_required"),
    ],
)
def test_changed_execution_conditions_block_resume(tmp_path, change, reason):
    bridge = make_bridge(tmp_path, capture=True)
    bridge._mqtt_intake.start()
    key, _ = accept(bridge)
    bridge._mqtt_intake.stop()
    config = bridge._mqtt_intake.expiry_policy.config
    if change == "policy":
        config["rules"][0]["ttl_seconds"]["signboard"] = 300
    restored = make_bridge(tmp_path, config=config)
    if change == "target":
        restored._resolve_edgex_device_ids = lambda *args: ["new-board"]
    if change == "mode":
        restored._sites.resolve_mode = lambda camera: (ControlMode.MANUAL, None)
    restored._mqtt_intake.start()
    try:
        wait_until(
            lambda: restored._mqtt_intake.store.get(key)["stage"] == "quarantined"
        )
        assert restored._mqtt_intake.store.get(key)["last_error"] == "held:" + reason
        restored._executor._publish_edgex_command.assert_not_called()
    finally:
        restored._mqtt_intake.stop()


def test_clock_step_is_held_instead_of_extending_deadline(tmp_path):
    bridge = make_bridge(tmp_path)
    key, _ = accept(bridge)
    bridge._mqtt_intake.expiry_policy.clock = lambda: time.time() - 10
    bridge._mqtt_intake._process(key)
    assert bridge._mqtt_intake.store.get(key)["last_error"] == "held:time_unverified"


def test_conflicting_redelivery_preserves_original_plan(tmp_path):
    bridge = make_bridge(tmp_path)
    first, _ = accept(bridge)
    original = window(bridge)
    conflicting, _ = accept(bridge, occurred_at=time.time() - 2)
    bridge._mqtt_intake._process(conflicting)
    assert (
        bridge._mqtt_intake.store.get(conflicting)["last_error"]
        == "held:event_content_conflict"
    )
    assert window(bridge) == original
    bridge._mqtt_intake._process(first)
    bridge._executor._publish_edgex_command.assert_called_once()


def test_legacy_backlog_without_snapshot_never_adopts_new_policy(tmp_path):
    bridge = make_bridge(tmp_path)
    key = bridge._mqtt_intake.store.accept_mqtt(
        "action_mqtt", TOPIC, b'{"event_id":"legacy"}'
    )
    bridge._mqtt_intake.start()
    try:
        assert (
            bridge._mqtt_intake.store.get(key)["last_error"]
            == "recovery_requires_review"
        )
        bridge._executor._publish_edgex_command.assert_not_called()
    finally:
        bridge._mqtt_intake.stop()


def test_second_physical_target_expires_after_first_publish_delay(tmp_path):
    bridge = make_bridge(tmp_path)
    bridge._resolve_edgex_device_ids = lambda *args: ["board-1", "board-2"]
    bridge._executor._resolve_edgex_device_ids = bridge._resolve_edgex_device_ids
    key, _ = accept(bridge)
    deadline = plan(bridge, key)["deadlines"]["signboard"]
    publish = bridge._executor._publish_edgex_command
    publish.side_effect = lambda *args: (advance_to(bridge, deadline), True)[1]
    bridge._mqtt_intake._process(key)
    publish.assert_called_once()
    with sqlite3.connect(bridge._repo.db_path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM action_commands WHERE command_id='event-1:signboard'"
            ).fetchone()[0]
        )
    assert [row["status"] for row in payload["device_results"]] == [
        "accepted",
        "expired",
    ]
    assert payload["device_results"][1]["error_code"] == "expired"


def test_expiry_after_resolving_devices_is_checked_before_dispatch(tmp_path):
    bridge = make_bridge(tmp_path)
    key, _ = accept(bridge)
    deadline = plan(bridge, key)["deadlines"]["signboard"]
    bridge._resolve_devices = lambda *args, **kwargs: (
        advance_to(bridge, deadline),
        {AlarmDevice.SIGNBOARD},
    )[1]
    bridge._mqtt_intake._process(key)
    bridge._executor._publish_edgex_command.assert_not_called()
    with sqlite3.connect(bridge._repo.db_path) as connection:
        status, raw = connection.execute(
            "SELECT status, payload_json FROM action_commands WHERE command_id='event-1:signboard'"
        ).fetchone()
    assert status == "expired"
    assert json.loads(raw)["action_decision"]["reason"] == "expired"


def test_external_payload_cannot_supply_execution_permission(tmp_path):
    bridge = make_bridge(tmp_path)
    bridge._handle_event(
        {
            "event_id": "forged",
            "camera_id": "cam1",
            "type": "fall_detected",
            "severity": "critical",
            "execution_context": {"expires_at": time.time() + 1000},
        }
    )
    bridge._executor._publish_edgex_command.assert_not_called()
    assert (
        bridge._publish_status.call_args.args[1]["reason"]
        == "execution_context_missing"
    )


def test_snapshot_commit_failure_prevents_ack_and_output(tmp_path):
    from types import SimpleNamespace

    bridge = make_bridge(tmp_path, capture=True)
    with bridge._mqtt_intake.store._connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_snapshot BEFORE UPDATE ON sensor_ingest_state BEGIN SELECT RAISE(ABORT,'full'); END"
        )
    subscriber = MqttTopicSubscriber(
        message_handler=bridge._mqtt_intake.accept, client_id="stable", durable=True
    )
    client = Mock()
    raw = json.dumps(
        {
            "event_id": "event-1",
            "type": "fall_detected",
            "camera_id": "cam1",
            "occurred_at": time.time() - 1,
        }
    ).encode()
    subscriber._on_message(
        client, None, SimpleNamespace(topic=TOPIC, payload=raw, qos=1, mid=1)
    )
    client.ack.assert_not_called()
    bridge._executor._publish_edgex_command.assert_not_called()


@pytest.mark.parametrize(
    "payload,reason",
    [
        (b'[{"event_id":"a"},{"event_id":"b"}]', "batch_requires_split"),
        (b"{}", "event_id_required"),
        (b"null", "invalid_payload"),
    ],
)
def test_unsupported_inputs_are_preserved_and_held(tmp_path, payload, reason):
    bridge = make_bridge(tmp_path)
    key = bridge._mqtt_intake.accept(TOPIC, payload)
    bridge._mqtt_intake._process(key)
    assert bridge._mqtt_intake.store.get(key)["last_error"] == "held:" + reason


def test_single_item_array_is_processed(tmp_path):
    bridge = make_bridge(tmp_path)
    raw = json.dumps(
        [
            {
                "event_id": "event-1",
                "type": "fall_detected",
                "camera_id": "cam1",
                "severity": "critical",
                "occurred_at": time.time() - 1,
            }
        ]
    ).encode()
    key = bridge._mqtt_intake.accept(TOPIC, raw)
    bridge._mqtt_intake._process(key)
    bridge._executor._publish_edgex_command.assert_called_once()


def test_control_backlog_is_not_automatically_reapproved(tmp_path):
    bridge = make_bridge(tmp_path, capture=True)
    bridge._mqtt_intake.start()
    key = bridge._mqtt_intake.accept("cctv/commands/approve", b'{"event_id":"event-1"}')
    bridge._mqtt_intake.stop()
    restored = make_bridge(tmp_path)
    restored._dispatch_command = Mock()
    restored._mqtt_intake.start()
    try:
        wait_until(
            lambda: restored._mqtt_intake.store.get(key)["stage"] == "quarantined"
        )
        restored._dispatch_command.assert_not_called()
    finally:
        restored._mqtt_intake.stop()


def test_killed_dispatch_is_not_replayed_even_by_a_duplicate_receipt(tmp_path):
    bridge = make_bridge(tmp_path)
    key, raw = accept(bridge)
    marker = tmp_path / "called"
    code = """
import sys,time
from pathlib import Path
from tests.test_action_expiry import make_bridge
b=make_bridge(Path(sys.argv[1]))
def call(*args,**kwargs):
    Path(sys.argv[2]).write_text('dispatching')
    time.sleep(60)
b._process_mqtt_payload=call
b._mqtt_intake.start()
time.sleep(60)
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(marker)])
    try:
        wait_until(marker.exists)
    finally:
        child.kill()
        child.wait(timeout=5)
    restored = make_bridge(tmp_path)
    restored._mqtt_intake.start()
    try:
        assert (
            restored._mqtt_intake.store.get(key)["last_error"]
            == "execution_unconfirmed"
        )
        duplicate = restored._mqtt_intake.accept(TOPIC, raw)
        wait_until(
            lambda: restored._mqtt_intake.store.get(duplicate)["stage"] == "quarantined"
        )
        assert window(restored)["state"] == "uncertain"
        restored._executor._publish_edgex_command.assert_not_called()
    finally:
        restored._mqtt_intake.stop()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "mqtt_intake_db": "input.db",
            "mqtt_client_id": "stable",
            "rest_enabled": True,
        },
    ],
)
def test_guard_cannot_be_enabled_without_durable_input_or_edgex(tmp_path, options):
    with pytest.raises(ValueError):
        ActionBridge(
            db_path=str(tmp_path / "events.db"),
            action_expiry_policy_path="config/action_expiry_policy.json",
            **options,
        )


def test_runner_passes_expiry_policy(monkeypatch):
    from runners import run_action_bridge as runner

    monkeypatch.setenv("ACTION_EXPIRY_POLICY_PATH", "config/action_expiry_policy.json")
    config = runner.parse_runtime_config([])
    constructor = Mock()
    monkeypatch.setattr(runner, "ActionBridge", constructor)
    runner.create_action_bridge(config)
    assert (
        constructor.call_args.kwargs["action_expiry_policy_path"]
        == "config/action_expiry_policy.json"
    )


@pytest.mark.parametrize(
    "value", [None, True, "2026-10-13T10:00:00Z", float("inf"), -1]
)
def test_corrupt_saved_deadline_is_held_without_retry_loop(tmp_path, value):
    bridge = make_bridge(tmp_path)
    key, _ = accept(bridge)
    frozen = plan(bridge, key)
    frozen["deadlines"]["signboard"] = value
    with bridge._mqtt_intake.store._connect() as connection:
        connection.execute(
            "UPDATE sensor_ingest_state SET projection_json=? WHERE ingest_id=?",
            (json.dumps(frozen), key),
        )
    bridge._mqtt_intake._process(key)
    assert (
        bridge._mqtt_intake.store.get(key)["last_error"]
        == "held:execution_context_missing"
    )
    bridge._executor._publish_edgex_command.assert_not_called()


def test_missing_canonical_id_is_not_replaced_during_execution(tmp_path):
    bridge = make_bridge(tmp_path)
    key, _ = accept(bridge, event_id=None, eventId="legacy-alias")
    bridge._mqtt_intake._process(key)
    assert bridge._mqtt_intake.store.get(key)["last_error"] == "held:event_id_required"
    bridge._executor._publish_edgex_command.assert_not_called()


def test_failed_execution_claim_never_calls_output(tmp_path):
    bridge = make_bridge(tmp_path)
    key, _ = accept(bridge)
    with bridge._mqtt_intake.store._connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_claim BEFORE UPDATE ON action_execution_windows BEGIN SELECT RAISE(ABORT,'full'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        bridge._mqtt_intake._process(key)
    assert bridge._mqtt_intake.store.get(key)["stage"] == "pending"
    assert window(bridge)["state"] == "queued"
    bridge._executor._publish_edgex_command.assert_not_called()


def test_result_commit_failure_cannot_replay_after_restart(tmp_path):
    bridge = make_bridge(tmp_path)
    key, raw = accept(bridge)
    with bridge._mqtt_intake.store._connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_finish BEFORE UPDATE ON action_execution_windows WHEN NEW.state='complete' BEGIN SELECT RAISE(ABORT,'full'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        bridge._mqtt_intake._process(key)
    bridge._executor._publish_edgex_command.assert_called_once()
    restored = make_bridge(tmp_path)
    restored._mqtt_intake.start()
    try:
        assert (
            restored._mqtt_intake.store.get(key)["last_error"]
            == "execution_unconfirmed"
        )
        duplicate = restored._mqtt_intake.accept(TOPIC, raw)
        wait_until(
            lambda: restored._mqtt_intake.store.get(duplicate)["stage"] == "quarantined"
        )
        assert window(restored)["state"] == "uncertain"
        restored._executor._publish_edgex_command.assert_not_called()
    finally:
        restored._mqtt_intake.stop()


def test_one_expired_device_does_not_discard_another_valid_device(tmp_path):
    bridge = make_bridge(tmp_path)
    config = bridge._mqtt_intake.expiry_policy.config
    config["rules"][0]["ttl_seconds"] = {"speaker": 1, "signboard": 10}
    bridge = make_bridge(tmp_path, config=config)
    bridge._sites.resolve_alarm_devices = lambda *args: {
        AlarmDevice.SPEAKER,
        AlarmDevice.SIGNBOARD,
    }
    bridge._resolve_devices = lambda *args, **kwargs: {
        AlarmDevice.SPEAKER,
        AlarmDevice.SIGNBOARD,
    }
    bridge._resolve_edgex_device_ids = lambda device, camera: [device + "-1"]
    bridge._executor._resolve_edgex_device_ids = bridge._resolve_edgex_device_ids
    key, _ = accept(bridge, occurred_at=time.time() - 2)
    bridge._mqtt_intake._process(key)
    bridge._executor._publish_edgex_command.assert_called_once()
    with sqlite3.connect(bridge._repo.db_path) as connection:
        rows = {
            row[0].rsplit(":", 1)[1]: row[1]
            for row in connection.execute(
                "SELECT command_id, status FROM action_commands"
            )
        }
    assert rows == {"speaker": "expired", "signboard": "pending"}


def test_ack_with_another_expired_target_is_partial(tmp_path):
    bridge = make_bridge(tmp_path)
    bridge._resolve_edgex_device_ids = lambda *args: ["board-1", "board-2"]
    bridge._executor._resolve_edgex_device_ids = bridge._resolve_edgex_device_ids
    key, _ = accept(bridge)
    deadline = plan(bridge, key)["deadlines"]["signboard"]
    bridge._executor._publish_edgex_command.side_effect = lambda *args: (
        advance_to(bridge, deadline),
        True,
    )[1]
    bridge._mqtt_intake._process(key)
    bridge._handle_edgex_result(
        "edgex/results/cctv/signboard",
        {
            "request_id": "event-1:signboard:board-1",
            "event_id": "event-1",
            "device_id": "board-1",
            "status": "acknowledged",
        },
    )
    with sqlite3.connect(bridge._repo.db_path) as connection:
        status, raw = connection.execute(
            "SELECT status, payload_json FROM action_commands"
        ).fetchone()
    assert status == "partial_failure"
    assert [item["status"] for item in json.loads(raw)["device_results"]] == [
        "acknowledged",
        "expired",
    ]
