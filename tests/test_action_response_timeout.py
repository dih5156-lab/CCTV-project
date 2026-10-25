"""미응답 기한·늦은 응답·재시작이 중복 실행이나 거짓 완료로 이어지지 않는지 확인한다."""

import gc
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event
from unittest.mock import MagicMock

import pytest

from src.devices.speaker import SpeakerConfig
from src.edgex.request_idempotency import RequestIdempotencyCache
from src.edgex.speaker_device_service import SpeakerDeviceService
from src.services import _action_bridge_repo
from src.services._action_bridge_repo import _EventRepo
from src.services.action_bridge import ActionBridge
from src.services.edgex_result_handler import EdgeXResultHandler


@pytest.fixture
def clock(monkeypatch):
    value = [datetime(2026, 10, 12, 0, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(_action_bridge_repo, "now_kst_iso", lambda: value[0].isoformat())
    return value


def target(device="board-1", status="pending"):
    return {"request_id": f"event-1:signboard:{device}", "event_id": "event-1",
            "device_id": device, "status": status, "error_code": None}


def register(repo, devices=("board-1",)):
    targets = [target(device) for device in devices]
    repo.record_command("event-1:signboard", "device/signboard", "pending", {
        "event_id": "event-1", "target_request_ids": [row["request_id"] for row in targets],
        "device_results": targets,
    })


def test_missing_response_expires_at_boundary_and_survives_restart(tmp_path, clock):
    bridge = ActionBridge(db_path=str(tmp_path / "events.db"))
    bridge._repo.init()
    register(bridge._repo)
    clock[0] += timedelta(seconds=29.999)
    assert bridge.list_commands()[0]["status"] == "pending"
    clock[0] += timedelta(milliseconds=1)
    row = bridge.list_commands()[0]
    assert row["status"] == "uncertain"
    result = row["payload"]["device_results"][0]
    assert result["error_code"] == "response_timeout"
    assert result["retry_allowed"] is False
    restored = ActionBridge(db_path=str(bridge._repo.db_path))
    restored._repo.init()
    assert restored.list_commands()[0]["status"] == "uncertain"


def test_late_reply_updates_only_missing_target_and_never_reexecutes(tmp_path, clock):
    bridge = ActionBridge(db_path=str(tmp_path / "events.db"))
    bridge._repo.init()
    bridge._publish_edgex_command = MagicMock()
    register(bridge._repo, ("board-1", "board-2"))
    handler = EdgeXResultHandler(repository=bridge._repo, publish_status=lambda *_: None)
    handler.handle("edgex/results/cctv/jetson-01/signboard", target("board-1", "acknowledged"))
    clock[0] += timedelta(seconds=30)
    rows = bridge.list_commands()[0]["payload"]["device_results"]
    assert [row["status"] for row in rows] == ["acknowledged", "uncertain"]
    handler.handle("edgex/results/cctv/jetson-01/signboard", target("board-2", "acknowledged"))
    final = bridge.list_commands()[0]
    assert final["status"] == "acknowledged"
    assert final["payload"]["device_results"][1]["error_code"] is None
    assert final["payload"]["device_results"][1]["response_timed_out_at"]
    bridge._publish_edgex_command.assert_not_called()


def test_repeat_registration_does_not_extend_deadline(tmp_path, clock):
    bridge = ActionBridge(db_path=str(tmp_path / "events.db"))
    bridge._repo.init()
    register(bridge._repo)
    original = bridge.list_commands()[0]["response_deadline"]
    clock[0] += timedelta(seconds=29)
    register(bridge._repo)
    assert bridge.list_commands()[0]["response_deadline"] == original
    clock[0] += timedelta(seconds=1)
    assert bridge.list_commands()[0]["status"] == "uncertain"


def test_timeout_and_actual_reply_race_preserve_success(tmp_path, clock):
    bridge = ActionBridge(db_path=str(tmp_path / "events.db"))
    bridge._repo.init()
    register(bridge._repo)
    clock[0] += timedelta(seconds=31)
    handler = EdgeXResultHandler(repository=bridge._repo, publish_status=lambda *_: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(bridge.list_commands), pool.submit(handler.handle,
            "edgex/results/cctv/jetson-01/signboard", target(status="acknowledged"))]
        for call in calls:
            call.result(timeout=3)
    assert bridge.list_commands()[0]["status"] == "acknowledged"


def test_failed_timeout_commit_does_not_publish_state(tmp_path, clock):
    bridge = ActionBridge(db_path=str(tmp_path / "events.db"))
    bridge._repo.init()
    register(bridge._repo)
    bridge._publish_status = MagicMock()
    with sqlite3.connect(bridge._repo.db_path) as conn:
        conn.execute("CREATE TRIGGER deny_update BEFORE UPDATE ON action_commands "
                     "BEGIN SELECT RAISE(ABORT, '저장 실패'); END")
    clock[0] += timedelta(seconds=31)
    assert bridge._expire_command_responses() is None
    bridge._publish_status.assert_not_called()


def test_start_loop_runs_timeout_check_without_api_polling(tmp_path, monkeypatch):
    bridge = ActionBridge(db_path=str(tmp_path / "events.db"))
    bridge._forwarder = MagicMock()
    bridge._start_rest_action_worker = MagicMock()
    bridge.stop = MagicMock()
    bridge._expire_command_responses = MagicMock()
    monkeypatch.setattr("src.services.action_bridge.create_mqtt_client", lambda *_: MagicMock())
    monkeypatch.setattr("src.services.action_bridge.signal.signal", lambda *_: None)
    monkeypatch.setattr("src.services.action_bridge.time.sleep", lambda *_: setattr(bridge, "_running", False))
    bridge.start()
    bridge._expire_command_responses.assert_called_once()


def test_duplicate_wait_has_deadline_without_releasing_owner(tmp_path):
    device = MagicMock()
    entered, release = Event(), Event()

    def play(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return True

    device.play.side_effect = play
    service = SpeakerDeviceService(device_id="speaker-1", config=SpeakerConfig(), device=device,
                                   request_db_path=tmp_path / "requests.db")
    service._request_results._wait_timeout_seconds = 0.02
    request = {"request_id": "event-1:speaker:speaker-1", "event_id": "event-1", "device": "speaker",
               "device_id": "speaker-1", "action": "play", "payload": {"text": "경고"}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(service.execute_request, request)
        try:
            assert entered.wait(1)
            duplicate = pool.submit(service.execute_request, request).result(timeout=0.5)
            assert duplicate.status == "uncertain"
            assert duplicate.error_code == "response_wait_timeout"
            assert service.get_request_status(request["request_id"])["retry_allowed"] is False
        finally:
            release.set()
        assert owner.result(timeout=1).status == "acknowledged"
    assert service.get_request_status(request["request_id"])["status"] == "acknowledged"
    device.play.assert_called_once()


def test_existing_schema_migration_uses_original_time_and_keeps_records(tmp_path, clock):
    path = tmp_path / "legacy.db"
    payload = {"event_id": "event-1", "device_results": [target()]}
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE action_commands(command_id TEXT PRIMARY KEY, topic TEXT NOT NULL, "
                     "status TEXT NOT NULL, message TEXT, site_id TEXT, event_id TEXT, "
                     "requested_at TEXT NOT NULL, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
        conn.execute("INSERT INTO action_commands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     ("event-1:signboard", "device/signboard", "pending", "", "site-1", "event-1",
                      clock[0].isoformat(), clock[0].isoformat(), json.dumps(payload)))
    clock[0] += timedelta(seconds=31)
    bridge = ActionBridge(db_path=str(path))
    bridge._repo.init()
    row = bridge.list_commands()[0]
    assert row["status"] == "uncertain"
    assert row["site_id"] == "site-1"
    assert row["requested_at"] == "2026-10-12T00:00:00+00:00"
    assert row["payload"]["event_id"] == "event-1"
    bridge._repo.init()
    assert len(bridge.list_commands()) == 1
    assert bridge.list_commands()[0]["response_deadline"] is None


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_response_deadlines_are_rejected(tmp_path, seconds):
    with pytest.raises(ValueError, match="제한시간"):
        _EventRepo(tmp_path / "action.db", response_timeout_seconds=seconds)
    with pytest.raises(ValueError, match="제한시간"):
        RequestIdempotencyCache(wait_timeout_seconds=seconds)


def test_runner_timeout_setting_reaches_deadline_calculation(tmp_path, monkeypatch, clock):
    from runners.run_action_bridge import create_action_bridge, parse_runtime_config

    monkeypatch.setenv("COMMAND_RESPONSE_TIMEOUT_SECONDS", "12.5")
    config = parse_runtime_config(["--db-path", str(tmp_path / "events.db")])
    bridge = create_action_bridge(config)
    bridge._repo.init()
    register(bridge._repo)
    clock[0] += timedelta(seconds=12.5)
    assert bridge.list_commands()[0]["status"] == "uncertain"


def test_restart_before_deadline_preserves_configured_original_deadline(tmp_path, clock):
    path = tmp_path / "events.db"
    first = ActionBridge(db_path=str(path), command_response_timeout_seconds=12)
    first._repo.init()
    register(first._repo)
    clock[0] += timedelta(seconds=10)
    restored = ActionBridge(db_path=str(path), command_response_timeout_seconds=100)
    restored._repo.init()
    assert restored.list_commands()[0]["status"] == "pending"
    clock[0] += timedelta(seconds=2)
    assert restored.list_commands()[0]["status"] == "uncertain"


def test_periodic_response_check_closes_database_connections(tmp_path):
    repo = _EventRepo(tmp_path / "events.db")
    repo.init()
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        before = len(os.listdir("/proc/self/fd"))
        for _ in range(30):
            assert repo.expire_command_responses() == []
        assert len(os.listdir("/proc/self/fd")) <= before + 1
    finally:
        if enabled:
            gc.enable()
        gc.collect()
