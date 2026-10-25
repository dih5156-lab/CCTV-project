import base64
import json
import sqlite3
import subprocess
import sys
import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from src.services.action_bridge import ActionBridge


def make_bridge(tmp_path, **kwargs):
    return ActionBridge(
        db_path=str(tmp_path / "actions.db"),
        mqtt_intake_db=str(tmp_path / "input.db"),
        mqtt_client_id="action-test-stable",
        edgex_command_mode="edgex",
        **kwargs,
    )


def wait_until(check):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.01)
    assert check()


def test_capture_only_preserves_bytes_without_action(tmp_path):
    bridge = make_bridge(tmp_path, mqtt_capture_only=True)
    calls = []
    bridge._process_mqtt_payload = lambda *args: calls.append(args)
    intake = bridge._mqtt_intake
    intake.start()
    try:
        key = intake.accept("cctv/input", b"\xffinvalid-json")
        receipt = intake.store.get(key)
        assert base64.b64decode(receipt["raw"]["payload_base64"]) == b"\xffinvalid-json"
        assert receipt["stage"] == "pending"
        assert calls == []
    finally:
        intake.stop()


def test_slow_action_does_not_block_next_receipt(tmp_path):
    bridge = make_bridge(tmp_path)
    entered, release = Event(), Event()

    def slow(*args):
        entered.set()
        assert release.wait(3)

    bridge._process_mqtt_payload = slow
    intake = bridge._mqtt_intake
    intake.start()
    try:
        first = intake.accept("cctv/input", b'{"event_id":"first"}')
        assert entered.wait(2)
        second = intake.accept("cctv/input", b'{"event_id":"second"}')
        assert intake.store.get(second)["stage"] == "pending"
        assert intake.store.get(first)["stage"] == "ready"
        release.set()
        wait_until(lambda: intake.store.get(second)["stage"] == "complete")
    finally:
        release.set()
        intake.stop()


def test_restart_holds_backlog_instead_of_replaying_outputs(tmp_path):
    intake = make_bridge(tmp_path, mqtt_capture_only=True)._mqtt_intake
    intake.start()
    key = intake.accept("cctv/input", b'{"event_id":"old"}')
    intake.stop()
    bridge = make_bridge(tmp_path)
    calls = []
    bridge._process_mqtt_payload = lambda *args: calls.append(args)
    bridge._mqtt_intake.start()
    try:
        receipt = bridge._mqtt_intake.store.get(key)
        assert receipt["stage"] == "quarantined"
        assert receipt["last_error"] == "recovery_requires_review"
        assert calls == []
    finally:
        bridge._mqtt_intake.stop()


def test_storage_failure_does_not_ack_or_execute(tmp_path):
    bridge = make_bridge(tmp_path)
    bridge._process_mqtt_payload = Mock()
    intake = bridge._mqtt_intake
    intake.start()
    subscriber = MqttTopicSubscriber(
        message_handler=intake.accept, client_id="stable", durable=True,
    )
    client = Mock()
    try:
        with sqlite3.connect(intake.store.path) as connection:
            connection.execute("CREATE TRIGGER reject_input BEFORE INSERT ON ingest_receipts "
                               "BEGIN SELECT RAISE(ABORT, 'full'); END")
        subscriber._on_message(client, None, SimpleNamespace(
            topic="test", payload=b"data", qos=1, mid=1,
        ))
        client.ack.assert_not_called()
        client.disconnect.assert_called_once()
        bridge._process_mqtt_payload.assert_not_called()
        assert intake.store.pending(namespace="action_mqtt") == []
    finally:
        intake.stop()


def test_ack_observes_committed_raw_even_when_capture_paused(tmp_path):
    intake = make_bridge(tmp_path, mqtt_capture_only=True)._mqtt_intake
    intake.start()
    subscriber = MqttTopicSubscriber(message_handler=intake.accept, client_id="stable", durable=True)
    client = Mock()

    def ack(mid, qos):
        with sqlite3.connect(intake.store.path) as connection:
            assert connection.execute("SELECT count(*) FROM ingest_receipts").fetchone() == (1,)
        return 0

    client.ack.side_effect = ack
    try:
        subscriber._on_message(client, None, SimpleNamespace(
            topic="test", payload=b"raw", qos=1, mid=2,
        ))
        client.ack.assert_called_once_with(2, 1)
    finally:
        intake.stop()


def test_execution_error_is_held_without_retry_and_next_input_progresses(tmp_path):
    bridge = make_bridge(tmp_path)
    bridge._process_mqtt_payload = Mock(side_effect=[RuntimeError("device timeout"), None])
    intake = bridge._mqtt_intake
    intake.start()
    try:
        failed = intake.accept("test", b"first")
        following = intake.accept("test", b"second")
        wait_until(lambda: intake.store.get(following)["stage"] == "complete")
        assert intake.store.get(failed)["stage"] == "quarantined"
        assert intake.store.get(failed)["last_error"] == "execution_unconfirmed:RuntimeError"
        assert bridge._process_mqtt_payload.call_count == 2
    finally:
        intake.stop()


def test_result_commit_failure_keeps_claim_and_never_reexecutes(tmp_path, monkeypatch):
    bridge = make_bridge(tmp_path)
    bridge._process_mqtt_payload = Mock()
    intake = bridge._mqtt_intake
    monkeypatch.setattr(intake.store, "complete", Mock(side_effect=sqlite3.OperationalError("full")))
    intake.start()
    try:
        key = intake.accept("test", b"one")
        wait_until(lambda: intake.store.get(key)["stage"] == "quarantined")
        assert intake.store.get(key)["last_error"] == "execution_unconfirmed"
        bridge._process_mqtt_payload.assert_called_once_with("test", b"one")
    finally:
        intake.stop()


def test_claim_commit_failure_does_not_execute(tmp_path):
    bridge = make_bridge(tmp_path)
    bridge._process_mqtt_payload = Mock()
    intake = bridge._mqtt_intake
    intake.start()
    try:
        with sqlite3.connect(intake.store.path) as connection:
            connection.execute("CREATE TRIGGER reject_claim BEFORE UPDATE ON sensor_ingest_state "
                               "WHEN new.stage='ready' BEGIN SELECT RAISE(ABORT, 'full'); END")
        key = intake.accept("test", b"one")
        time.sleep(0.3)
        assert intake.store.get(key)["stage"] == "pending"
        bridge._process_mqtt_payload.assert_not_called()
    finally:
        intake.stop()


def test_second_process_cannot_take_active_inbox(tmp_path):
    intake = make_bridge(tmp_path, mqtt_capture_only=True)._mqtt_intake
    intake.start()
    code = """
import sys
from src.services.action_mqtt_intake import ActionMqttIntake
other = ActionMqttIntake(sys.argv[1], lambda *args: None)
try:
    other.start()
except RuntimeError:
    sys.exit(0)
other.stop()
sys.exit(3)
"""
    try:
        result = subprocess.run([sys.executable, "-c", code, str(intake.store.path)], timeout=5)
        assert result.returncode == 0
    finally:
        intake.stop()


def test_killed_execution_is_preserved_and_not_replayed(tmp_path):
    path = tmp_path / "killed.db"
    code = """
import sys,time
from pathlib import Path
from src.services.action_mqtt_intake import ActionMqttIntake
def execute(*args):
    Path(sys.argv[2]).write_text('executing')
    time.sleep(60)
intake = ActionMqttIntake(sys.argv[1], execute)
intake.start()
intake.accept('test', b'raw')
time.sleep(60)
"""
    marker = tmp_path / "executing"
    child = subprocess.Popen([sys.executable, "-c", code, str(path), str(marker)])
    try:
        # 출력 장치 진입 직후 프로세스를 종료하여 ready 선점의 디스크 보존을 검증한다.
        wait_until(marker.exists)
    finally:
        child.kill()
        child.wait(timeout=5)
    from src.services.action_mqtt_intake import ActionMqttIntake
    handler = Mock()
    restored = ActionMqttIntake(str(path), handler)
    key, = restored.store.pending(namespace="action_mqtt")
    restored.start()
    try:
        assert restored.store.get(key)["stage"] == "quarantined"
        assert restored.store.get(key)["last_error"] == "execution_unconfirmed"
        handler.assert_not_called()
    finally:
        restored.stop()


@pytest.mark.parametrize("options", [
    {"mqtt_intake_db": "new.db"},
    {"mqtt_capture_only": True},
    {"mqtt_intake_db": "new.db", "mqtt_client_id": "id", "edgex_command_mode": "direct"},
])
def test_unsafe_intake_configuration_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        ActionBridge(db_path=str(tmp_path / "event.db"), **options)


def test_publisher_uses_reconnected_subscriber_client():
    subscriber = MqttTopicSubscriber(message_handler=Mock())
    with pytest.raises(ConnectionError):
        subscriber.publish("test", "payload")
    old, new = Mock(), Mock()
    subscriber._client = old
    subscriber._client = new
    assert subscriber.publish("test", "payload", qos=1) is new.publish.return_value
    old.publish.assert_not_called()


def test_runner_passes_intake_settings(monkeypatch, tmp_path):
    from runners import run_action_bridge as runner
    monkeypatch.setenv("ACTION_MQTT_INTAKE_DB", str(tmp_path / "input.db"))
    monkeypatch.setenv("ACTION_MQTT_CLIENT_ID", "runner-stable")
    monkeypatch.setenv("ACTION_MQTT_CAPTURE_ONLY", "true")
    config = runner.parse_runtime_config(["--edgex-command-mode", "edgex"])
    constructor = Mock()
    monkeypatch.setattr(runner, "ActionBridge", constructor)
    runner.create_action_bridge(config)
    assert constructor.call_args.kwargs["mqtt_intake_db"] == str(tmp_path / "input.db")
    assert constructor.call_args.kwargs["mqtt_client_id"] == "runner-stable"
    assert constructor.call_args.kwargs["mqtt_capture_only"] is True


def test_connection_failure_releases_worker_and_forwarder(tmp_path, monkeypatch):
    import src.services.action_bridge as module
    bridge = make_bridge(tmp_path)
    bridge._forwarder = Mock()
    bridge._MQTT_MAX_ATTEMPTS = 1
    client = Mock()
    client.connect.return_value = False
    monkeypatch.setattr(module, "MqttTopicSubscriber", Mock(return_value=client))
    with pytest.raises(ConnectionError):
        bridge.start()
    assert bridge._mqtt_intake._lease is None
    assert bridge._mqtt_intake._worker is None
    client.disconnect.assert_called_once()
    bridge._forwarder.stop.assert_called_once()


def test_worker_start_failure_stops_started_forwarder(tmp_path, monkeypatch):
    bridge = make_bridge(tmp_path)
    bridge._forwarder = Mock()
    monkeypatch.setattr(bridge._mqtt_intake, "start", Mock(side_effect=RuntimeError("owned")))
    with pytest.raises(RuntimeError, match="owned"):
        bridge.start()
    bridge._forwarder.stop.assert_called_once()


def test_new_input_database_is_private(tmp_path):
    bridge = make_bridge(tmp_path)
    assert bridge._mqtt_intake.store.path.stat().st_mode & 0o077 == 0


def test_duplicate_delivery_preserves_request_id_and_device_executes_once(tmp_path):
    from src.devices.signboard import SignboardConfig
    from src.edgex.dabit_device_service import DabitDeviceService
    registry = tmp_path / "devices.json"
    registry.write_text(json.dumps({"devices": [{
        "device_id": "test-signboard", "device_type": "signboard",
        "camera_ids": ["camera_1"], "enabled": True,
    }]}))
    bridge = make_bridge(tmp_path, edgex_allowed_devices={"signboard"},
                         edgex_device_registry_path=str(registry),
                         signboard_config=SignboardConfig(host="192.0.2.1"))
    bridge._repo.init()
    requests = []
    bridge._mqtt_client = SimpleNamespace(publish=lambda topic, body, **kwargs:
        requests.append((topic, json.loads(body))) or SimpleNamespace(rc=0))
    intake = bridge._mqtt_intake
    intake.start()
    try:
        raw = json.dumps({"event_id": "same-event", "camera_id": "camera_1", "type": "fall_detected",
                          "confidence": 0.95, "severity": "critical"}).encode()
        first = intake.accept("cctv/ai/events/camera_1/fall_detected", raw)
        second = intake.accept("cctv/ai/events/camera_1/fall_detected", raw)
        wait_until(lambda: intake.store.get(second)["stage"] == "complete")
        assert intake.store.get(first)["stage"] == "complete"
        commands = [body for topic, body in requests if topic.startswith("edgex/commands/")]
        assert commands
        assert {body["request_id"] for body in commands} == {"same-event:signboard:test-signboard"}
        device = Mock()
        device.display.return_value = True
        service = DabitDeviceService(device_id="test-signboard", config=SignboardConfig(),
                                     device=device, request_db_path=str(tmp_path / "device.sqlite3"))
        for body in commands + commands:
            assert service.execute_request(body).status == "acknowledged"
        device.display.assert_called_once()
    finally:
        intake.stop()
