"""물리 장치별 지연 격리·수신 보존·종료/재시작 경계를 확인한다."""

import importlib
import json
import sqlite3
import subprocess
import sys
import time
from threading import Event
from unittest.mock import Mock

import pytest

from src.devices.signboard import SignboardConfig
from src.devices.siren import SensorConfig
from src.devices.speaker import SpeakerConfig
from src.edgex.dabit_device_service import DabitDeviceService
from src.edgex.device_mqtt_workers import DeviceMqttWorkers
from src.edgex.siren_device_service import SirenDeviceService
from src.edgex.speaker_device_service import SpeakerDeviceService
from tests.test_action_device_queue import make_queue, queue, rows
from tests.test_action_expiry import accept
from tests.test_action_mqtt_intake import wait_until

KINDS = [
    ("signboard", DabitDeviceService, SignboardConfig, "display", "display"),
    ("speaker", SpeakerDeviceService, SpeakerConfig, "play", "play"),
    ("siren", SirenDeviceService, SensorConfig, "trigger", "trigger"),
]


def make_workers(tmp_path, kind=KINDS[0], timeout=30):
    name, cls, config, action, method = kind
    devices = {key: Mock() for key in ("slow", "fast")}
    service = cls(device_id="slow", config=config(), devices=devices,
                  request_db_path=tmp_path / "requests.db")
    publish = Mock(return_value=True)
    workers = DeviceMqttWorkers(service, name, publish, timeout_seconds=timeout)
    def request(device_id="slow", event_id="one", **changes):
        return {"request_id": f"{event_id}:{name}:{device_id}", "event_id": event_id,
                "device": name, "device_id": device_id, "action": action,
                "payload": {"text": "경보", "event_type": "fall_detected"}, **changes}
    return workers, devices, request, method


def submit(workers, request):
    raw = json.dumps(request).encode()
    workers.accept("edgex/commands/test", raw)
    return raw


def inbox(workers):
    with sqlite3.connect(workers.cache._database_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM device_mqtt_inbox ORDER BY sequence")]


@pytest.mark.parametrize("kind", KINDS)
def test_slow_device_does_not_block_same_kind_other_device_or_status(tmp_path, kind):
    workers, devices, request, method = make_workers(tmp_path, kind, timeout=.01)
    entered, release = Event(), Event()
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return True
    getattr(devices["slow"], method).side_effect = slow
    workers.start()
    try:
        raw = submit(workers, request())
        wait_until(entered.is_set)
        submit(workers, request("fast", "two"))
        submit(workers, request(action="request_status"))
        wait_until(lambda: getattr(devices["fast"], method).call_count == 1)
        wait_until(lambda: any(call.args[0]["request_id"] == request()["request_id"] for call in workers.publish.call_args_list))
        wait_until(lambda: any(row["device_id"] == "fast" and row["state"] == "complete" for row in inbox(workers)))
        assert inbox(workers)[0]["payload"] == raw
        wait_until(lambda: time.monotonic() - next(iter(workers._active.values()))[0] >= .01)
        workers.expire_active()
        workers.expire_active()
        timeouts = [call.args[0] for call in workers.publish.call_args_list
                    if call.args[0].get("error_code") == "device_worker_timeout"]
        assert len(timeouts) == 1 and timeouts[0]["status"] == "uncertain"
        assert inbox(workers)[0]["state"] == "running"
        assert workers.service.get_request_status(request()["request_id"])["error_code"] == "device_worker_timeout"
        assert getattr(devices["slow"], method).call_count == 1
    finally:
        release.set()
        assert workers.stop()


def test_same_device_calls_are_serial_and_duplicate_executes_once(tmp_path):
    workers, devices, request, method = make_workers(tmp_path)
    entered, release = Event(), Event()
    getattr(devices["slow"], method).side_effect = lambda **kwargs: entered.set() or release.wait(5)
    workers.start()
    try:
        submit(workers, request())
        wait_until(entered.is_set)
        submit(workers, request())
        submit(workers, request(event_id="two"))
        assert [row["state"] for row in inbox(workers)] == ["running", "pending", "pending"]
        assert getattr(devices["slow"], method).call_count == 1
        release.set()
        wait_until(lambda: all(row["state"] == "complete" for row in inbox(workers)))
        assert getattr(devices["slow"], method).call_count == 2
    finally:
        release.set()
        workers.stop()


@pytest.mark.parametrize("state", ["pending", "running"])
def test_restart_holds_old_output_and_redelivery_but_accepts_new_request(tmp_path, state):
    workers, devices, request, method = make_workers(tmp_path)
    submit(workers, request())
    with sqlite3.connect(workers.cache._database_path) as conn:
        conn.execute("UPDATE device_mqtt_inbox SET state=?", (state,))
    workers.start()
    try:
        submit(workers, request())
        submit(workers, request("fast", "new"))
        wait_until(lambda: getattr(devices["fast"], method).call_count == 1)
        assert [row["state"] for row in inbox(workers)][:2] == ["held", "held"]
        assert workers.service.get_request_status(request()["request_id"])["error_code"] == "worker_recovery_requires_review"
        getattr(devices["slow"], method).assert_not_called()
    finally:
        workers.stop()


def test_storage_failure_raises_before_acceptance_and_never_executes(tmp_path):
    workers, devices, request, method = make_workers(tmp_path)
    with sqlite3.connect(workers.cache._database_path) as conn:
        conn.execute("CREATE TRIGGER fail BEFORE INSERT ON device_mqtt_inbox BEGIN SELECT RAISE(ABORT,'full'); END")
    with pytest.raises(sqlite3.Error):
        submit(workers, request())
    assert inbox(workers) == []
    getattr(devices["slow"], method).assert_not_called()


def test_stop_keeps_lease_while_blocked_and_does_not_start_waiting_output(tmp_path):
    workers, devices, request, method = make_workers(tmp_path)
    entered, release = Event(), Event()
    getattr(devices["slow"], method).side_effect = lambda **kwargs: entered.set() or release.wait(5)
    workers.start()
    try:
        submit(workers, request())
        wait_until(entered.is_set)
        submit(workers, request(event_id="two"))
        started = time.monotonic()
        assert workers.stop(timeout=.02) is False
        assert time.monotonic() - started < .5
        other, _, _, _ = make_workers(tmp_path)
        with pytest.raises(OSError):
            other.start()
        assert getattr(devices["slow"], method).call_count == 1
    finally:
        release.set()
        assert workers.stop()
    assert inbox(workers)[1]["state"] == "pending"


@pytest.mark.parametrize("raw", [b"bad", b"[]", b"{}", b'{"device":"speaker"}', b'\xff'])
def test_invalid_raw_is_preserved_and_held(tmp_path, raw):
    workers, devices, _, method = make_workers(tmp_path)
    workers.accept("topic", raw)
    assert inbox(workers)[0]["payload"] == raw
    assert inbox(workers)[0]["state"] == "held"
    getattr(devices["slow"], method).assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(tmp_path, timeout):
    with pytest.raises(ValueError):
        make_workers(tmp_path, timeout=timeout)


def test_complete_request_is_not_reexecuted_after_worker_restart(tmp_path):
    workers, devices, request, method = make_workers(tmp_path)
    workers.start()
    submit(workers, request())
    wait_until(lambda: inbox(workers)[0]["state"] == "complete")
    assert workers.stop()
    restored, fresh, restored_request, _ = make_workers(tmp_path)
    restored.start()
    try:
        submit(restored, restored_request())
        wait_until(lambda: inbox(restored)[1]["state"] == "complete")
        getattr(fresh["slow"], method).assert_not_called()
    finally:
        restored.stop()


def test_action_slow_publish_preserves_intake_and_other_physical_target(tmp_path):
    bridge = make_queue(tmp_path)
    bridge._resolve_edgex_device_ids = lambda device, camera: ["slow" if camera == "cam1" else "fast"]
    entered, release, fast = Event(), Event(), Event()
    def publish(topic, payload):
        if payload["action"] == "request_status":
            return True
        if payload["device_id"] == "slow":
            entered.set()
            assert release.wait(5)
        else:
            fast.set()
        return True
    bridge._executor._publish_edgex_command.side_effect = publish
    bridge._mqtt_intake.start()
    try:
        accept(bridge, event_id="slow-one")
        wait_until(entered.is_set)
        receipt, _ = accept(bridge, event_id="fast-one", camera_id="cam2")
        wait_until(fast.is_set)
        wait_until(lambda: bridge._mqtt_intake.store.get(receipt)["stage"] == "complete")
        assert next(row for row in rows(bridge) if row["resource_key"] == "slow")["state"] == "dispatching"
        assert len(queue(bridge)._workers) == 2
        queue(bridge).refresh_workers()
        assert len(queue(bridge)._workers) == 2
    finally:
        release.set()
        bridge._mqtt_intake.stop()


def test_action_shutdown_keeps_single_owner_until_slow_publish_returns(tmp_path, monkeypatch):
    bridge = make_queue(tmp_path)
    entered, release = Event(), Event()
    bridge._executor._publish_edgex_command.side_effect = lambda *args: entered.set() or release.wait(5)
    original = queue(bridge).stop_workers
    monkeypatch.setattr(queue(bridge), "stop_workers", lambda timeout=5: original(.02))
    bridge._mqtt_intake.start()
    try:
        accept(bridge)
        wait_until(entered.is_set)
        with pytest.raises(RuntimeError, match="장치 작업자"):
            bridge._mqtt_intake.stop()
        assert bridge._mqtt_intake._lease is not None
        with pytest.raises(RuntimeError, match="다른 작업자"):
            make_queue(tmp_path)._mqtt_intake.start()
    finally:
        release.set()
        wait_until(lambda: all(not worker.is_alive() for worker in queue(bridge)._workers.values()))
        bridge._mqtt_intake.stop()
    assert bridge._mqtt_intake._lease is None


@pytest.mark.parametrize("kind", ["signboard", "speaker", "siren"])
def test_runner_enables_durable_intake_and_stops_workers(tmp_path, monkeypatch, kind):
    module = importlib.import_module("runners.run_" + kind + "_device_service")
    workers, _, _, _ = make_workers(tmp_path, next(row for row in KINDS if row[0] == kind))
    monkeypatch.setenv("DEVICE_WORKERS_ENABLED", "true")
    monkeypatch.setenv("EDGEX_JETSON_ID", "test-node")
    monkeypatch.setattr(module, "create_service", lambda: workers.service)
    stopped = Mock()
    stopped.wait.side_effect = [False, True]
    monkeypatch.setattr(module, "Event", lambda: stopped)
    monkeypatch.setattr(module, "ThreadingHTTPServer", Mock())
    monkeypatch.setattr(module, "start_validation_responder", Mock())
    monkeypatch.setattr(module.signal, "signal", Mock())
    client = Mock()
    client.connect.return_value = True
    subscriber = Mock(return_value=client)
    monkeypatch.setattr(module, "MqttTopicSubscriber", subscriber)
    module.main()
    assert subscriber.call_args.kwargs["durable"] is True
    assert subscriber.call_args.kwargs["client_id"] == f"cctv-test-node-{kind}"
    client.reconnect_if_needed.assert_called_once()
    client.disconnect.assert_called_once()
    assert not subscriber.call_args.kwargs["message_handler"].__self__._lease


def test_process_killed_during_execution_holds_redelivery(tmp_path):
    code = """
import os, sys, time
from pathlib import Path
from tests.test_device_workers import make_workers, submit
path = Path(sys.argv[1])
workers, devices, request, method = make_workers(path)
def crash(**kwargs):
    (path / 'physical-call.txt').write_text('one')
    os._exit(19)
getattr(devices['slow'], method).side_effect = crash
workers.start()
submit(workers, request())
time.sleep(10)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], timeout=8, capture_output=True)
    assert result.returncode == 19 and (tmp_path / "physical-call.txt").read_text() == "one"
    workers, devices, request, method = make_workers(tmp_path)
    workers.start()
    try:
        submit(workers, request())
        assert all(row["state"] == "held" for row in inbox(workers))
        getattr(devices["slow"], method).assert_not_called()
    finally:
        workers.stop()
