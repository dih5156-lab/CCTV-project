"""과거 실행 이력의 빈 내용을 성공으로 꾸미거나 재출력하지 않는지 확인한다."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.ops.seed_legacy_device_requests import seed_legacy_requests
from src.devices.speaker import SpeakerConfig
from src.edgex.speaker_device_service import SpeakerDeviceService
from tests.test_device_request_durability import CASES, request
from tests.test_runtime_database_backup import run_backup


def source_database(path, rows):
    """실제 이관에 필요한 기존 Action 명령 열과 기록을 만든다."""
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE action_commands(command_id TEXT PRIMARY KEY, topic TEXT, "
                     "event_id TEXT, status TEXT, payload_json TEXT)")
        conn.executemany("INSERT INTO action_commands VALUES (?, ?, ?, ?, ?)", rows)
    return path


@pytest.mark.parametrize("kind,service_class,config,action,method", CASES)
def test_legacy_logical_id_blocks_new_physical_request_after_restart(
    tmp_path, kind, service_class, config, action, method,
):
    device = MagicMock()
    path = tmp_path / "requests.db"
    options = dict(device_id="device-1", config=config(), device=device, request_db_path=path)
    service_class(**options)
    # 빈 hash는 내용/실행 결과를 복원할 수 없는 과거 ID의 보류 표식이다.
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO device_requests(request_id, request_hash) VALUES (?, '')",
                     (f"event-1:{kind}",))
    restored = service_class(**options)
    result = restored.execute_request(request(kind, action))
    assert result.status == "uncertain"
    assert result.error_code == "legacy_execution_unconfirmed"
    getattr(device, method).assert_not_called()
    state = restored.get_request_status(result.request_id)
    assert state["error_code"] == "legacy_execution_unconfirmed"
    assert state["retry_allowed"] is False
    assert service_class(**options).execute_request(request(kind, action)).status == "uncertain"
    assert restored.execute_request(request(kind, action, request_id="new", event_id="new")).status == "acknowledged"
    getattr(device, method).assert_called_once()


@pytest.mark.parametrize("kind,service_class,config,action,method", CASES)
def test_migration_keeps_all_statuses_held_and_source_unchanged(
    tmp_path, kind, service_class, config, action, method,
):
    statuses = ["acknowledged", "failed", "simulated", "error", "sent", "pending", "uncertain"]
    rows = [(f"event-{i}:{kind}", f"device/{kind}", f"event-{i}", status, "{}")
            for i, status in enumerate(statuses)]
    rows.append(("mode", "cctv/commands/mode", None, "acknowledged", "{}"))
    source = source_database(tmp_path / "source.db", rows)
    before = source.read_bytes()
    destination = tmp_path / "requests.db"
    summary = seed_legacy_requests(source, destination, kind)
    assert summary["source_rows"] == 8
    assert summary["selected_commands"] == summary["held_identifiers"] == 7
    assert summary["legacy_statuses"] == dict.fromkeys(statuses, 1)
    assert source.read_bytes() == before
    assert destination.stat().st_mode & 0o077 == 0
    device = MagicMock()
    service = service_class(device_id="device-1", config=config(), device=device, request_db_path=destination)
    for i in range(7):
        result = service.execute_request(request(kind, action, request_id=f"event-{i}:{kind}:device-1",
                                                 event_id=f"event-{i}"))
        assert result.error_code == "legacy_execution_unconfirmed"
    getattr(device, method).assert_not_called()


@pytest.mark.parametrize("values", [
    {"request_id": "event-1:speaker"},
    {"request_id": "a-different-request-id"},
    {"event_id": "a-changed-event"},
    {"device_id": "site:device:2", "request_id": "event-1:speaker:site:device:2"},
])
def test_legacy_aliases_cannot_bypass_hold(tmp_path, values):
    source = source_database(tmp_path / "source.db", [
        ("event-1:speaker", "device/speaker", "event-1", "failed", "{}"),
    ])
    destination = tmp_path / "requests.db"
    seed_legacy_requests(source, destination, "speaker")
    device = MagicMock()
    service = SpeakerDeviceService(device_id="device-1", config=SpeakerConfig(), device=device,
                                   devices={"device-1": device, "site:device:2": device}, request_db_path=destination)
    result = service.execute_request(request(**values))
    assert result.error_code == "legacy_execution_unconfirmed"
    assert service.get_request_status(result.request_id)["status"] == "uncertain"
    device.play.assert_not_called()


def test_result_topic_physical_ids_and_event_aliases_are_held(tmp_path):
    source = source_database(tmp_path / "source.db", [
        ("old:speaker:site:device", "edgex/results/cctv/j1/speaker", "column-event", "acknowledged",
         json.dumps({"request_id": "old:speaker:site:device", "device_id": "site:device",
                     "event_id": "payload-event", "device_results": [
                         {"request_id": "alias:speaker:device-2", "device_id": "device-2"}]})),
    ])
    destination = tmp_path / "requests.db"
    seed_legacy_requests(source, destination, "speaker")
    with sqlite3.connect(destination) as conn:
        actual = conn.execute("SELECT request_id,request_hash,result_json FROM device_requests").fetchall()
    expected = {"old:speaker:site:device", "old:speaker", "column-event:speaker", "payload-event:speaker",
                "alias:speaker:device-2", "alias:speaker"}
    assert {r[0] for r in actual} == expected
    assert all(r[1:] == ("", None) for r in actual)


@pytest.mark.parametrize("payload", ["not-json", "[]", '{"device_results":{}}',
                                    '{"event_id":123}', '{"request_id":" bad-id"}'])
def test_invalid_history_does_not_publish_partial_journal(tmp_path, payload):
    source = source_database(tmp_path / "source.db", [
        ("ok:speaker", "device/speaker", "ok", "failed", "{}"),
        ("bad:speaker", "device/speaker", "bad", "failed", payload),
    ])
    destination = tmp_path / "requests.db"
    with pytest.raises(ValueError):
        seed_legacy_requests(source, destination, "speaker")
    assert not destination.exists()
    assert not list(tmp_path.glob(".legacy-*"))


def test_existing_journal_is_never_overwritten(tmp_path):
    source = source_database(tmp_path / "source.db", [])
    destination = tmp_path / "requests.db"
    destination.write_bytes(b"existing execution evidence")
    with pytest.raises(FileExistsError):
        seed_legacy_requests(source, destination, "speaker")
    assert destination.read_bytes() == b"existing execution evidence"


def test_missing_source_is_not_created(tmp_path):
    source, destination = tmp_path / "missing.db", tmp_path / "requests.db"
    with pytest.raises(sqlite3.Error):
        seed_legacy_requests(source, destination, "speaker")
    assert not source.exists() and not destination.exists()


def test_concurrent_destination_creation_is_preserved(tmp_path, monkeypatch):
    source = source_database(tmp_path / "source.db", [])
    destination = tmp_path / "requests.db"
    original = os.link

    def competing_link(source_path, destination_path):
        destination.write_bytes(b"another writer")
        original(source_path, destination_path)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError):
        seed_legacy_requests(source, destination, "speaker")
    assert destination.read_bytes() == b"another writer"
    assert not list(tmp_path.glob(".legacy-*"))


def test_held_request_commit_failure_never_executes(tmp_path):
    source = source_database(tmp_path / "source.db", [
        ("event-1:speaker", "device/speaker", "event-1", "failed", "{}"),
    ])
    destination = tmp_path / "requests.db"
    seed_legacy_requests(source, destination, "speaker")
    device = MagicMock()
    service = SpeakerDeviceService(device_id="device-1", config=SpeakerConfig(), device=device,
                                   request_db_path=destination)
    with sqlite3.connect(destination) as conn:
        conn.execute("CREATE TRIGGER fail_claim BEFORE INSERT ON device_requests "
                     "BEGIN SELECT RAISE(ABORT, 'cannot preserve held request'); END")
    result = service.execute_request(request())
    assert result.error_code == "request_store_unavailable"
    device.play.assert_not_called()


def test_backup_and_parallel_restart_keep_legacy_hold(tmp_path):
    source = source_database(tmp_path / "source.db", [
        ("event-1:speaker", "device/speaker", "event-1", "failed", "{}"),
    ])
    destination = tmp_path / "data/device_requests/speaker.sqlite3"
    seed_legacy_requests(source, destination, "speaker")
    restored = run_backup(tmp_path) / "speaker.sqlite3"
    device = MagicMock()
    services = [SpeakerDeviceService(device_id="device-1", config=SpeakerConfig(), device=device,
                                      request_db_path=restored) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: services[i % 2].execute_request(request()), range(20)))
    assert all(r.error_code == "legacy_execution_unconfirmed" for r in results)
    device.play.assert_not_called()


def test_cli_uses_read_only_source_and_creates_complete_journal(tmp_path):
    source = source_database(tmp_path / "source.db", [
        ("event-1:speaker", "device/speaker", "event-1", "failed", "{}"),
    ])
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "requests.db"
    script = Path(__file__).resolve().parents[1] / "scripts/ops/seed_legacy_device_requests.py"
    result = subprocess.run([sys.executable, str(script), "--source-db", str(source),
                             "--destination-db", str(destination), "--device", "speaker"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["held_identifiers"] == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
