"""실행 이력·내용 충돌·종료 경계가 물리 출력 중복을 막는지 확인한다."""

import importlib
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import pytest

from src.devices.signboard import SignboardConfig
from src.devices.siren import SensorConfig
from src.devices.speaker import SpeakerConfig
from src.edgex.dabit_device_service import DabitDeviceService
from src.edgex.siren_device_service import SirenDeviceService
from src.edgex.speaker_device_service import SpeakerDeviceService

CASES = [
    ("signboard", DabitDeviceService, SignboardConfig, "display", "display"),
    ("speaker", SpeakerDeviceService, SpeakerConfig, "play", "play"),
    ("siren", SirenDeviceService, SensorConfig, "trigger", "trigger"),
]


def request(kind="speaker", action="play", **values):
    return {"request_id": "event-1:" + kind + ":device-1", "event_id": "event-1",
            "device": kind, "device_id": "device-1", "action": action,
            "payload": {"text": "경고"}, **values}


@pytest.mark.parametrize("kind,service_class,config,action,method", CASES)
def test_changed_content_is_rejected_without_replacing_original(kind, service_class, config, action, method):
    device = MagicMock()
    service = service_class(device_id="device-1", config=config(), device=device)
    original = service.execute_request(request(kind, action))
    conflict = service.execute_request(request(kind, action, payload={"text": "변경"}))
    assert conflict.status == "failed"
    assert conflict.error_code == "request_id_conflict"
    assert service.execute_request(request(kind, action)) == original
    getattr(device, method).assert_called_once()


@pytest.mark.parametrize("kind,service_class,config,action,method", CASES)
def test_completed_result_survives_restart_and_memory_eviction(tmp_path, kind, service_class, config, action, method):
    device = MagicMock()
    options = dict(device_id="device-1", config=config(), device=device,
                   request_db_path=tmp_path / "requests.sqlite3")
    first = service_class(**options)
    first._request_results._max_entries = 1
    original = first.execute_request(request(kind, action))
    first.execute_request(request(kind, action, request_id="other"))
    duplicate = request(kind, action, issued_at="new-time", attempt_id="attempt-2", source="retry")
    assert first.execute_request(duplicate) == original
    restored = service_class(**options)
    assert restored.execute_request(duplicate) == original
    assert getattr(device, method).call_count == 2


@pytest.mark.parametrize("change", [
    {"event_id": "different"}, {"device_id": "device-2"}, {"action": "stop"},
    {"payload": {"text": "다른 경고"}},
])
def test_persisted_conflict_includes_event_target_action_and_payload(tmp_path, change):
    device = MagicMock()
    options = dict(device_id="device-1", config=SpeakerConfig(), device=device,
                   devices={"device-1": device, "device-2": device}, request_db_path=tmp_path / "requests.db")
    original = SpeakerDeviceService(**options).execute_request(request())
    restored = SpeakerDeviceService(**options)
    assert restored.execute_request(request(**change)).error_code == "request_id_conflict"
    assert restored.execute_request(request()) == original
    device.play.assert_called_once()
    device.stop.assert_not_called()


def test_inflight_conflict_and_other_service_instance_do_not_execute(tmp_path):
    entered, release = Event(), Event()
    device = MagicMock()

    def play(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return True

    device.play.side_effect = play
    options = dict(device_id="device-1", config=SpeakerConfig(), device=device,
                   request_db_path=tmp_path / "requests.db")
    first, second = SpeakerDeviceService(**options), SpeakerDeviceService(**options)
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(first.execute_request, request())
        try:
            assert entered.wait(2)
            assert first.execute_request(request(payload={"text": "변경"})).error_code == "request_id_conflict"
            assert second.execute_request(request()).status == "uncertain"
        finally:
            release.set()
        # 완료 commit을 먼저 기다려 비교 순서에 따른 실행 중 조회와 구분한다.
        completed = running.result(timeout=2)
        assert second.execute_request(request()) == completed
    device.play.assert_called_once()


@pytest.mark.parametrize("success", [False, RuntimeError("응답 유실")])
def test_physical_failure_is_not_blindly_retried_after_restart(tmp_path, success):
    device = MagicMock()
    if isinstance(success, Exception):
        device.play.side_effect = success
    else:
        device.play.return_value = success
    options = dict(device_id="device-1", config=SpeakerConfig(), device=device,
                   request_db_path=tmp_path / "requests.db")
    result = SpeakerDeviceService(**options).execute_request(request())
    assert result.status == "uncertain"
    assert SpeakerDeviceService(**options).execute_request(request()) == result
    device.play.assert_called_once()


def test_storage_failure_before_execution_blocks_output(tmp_path):
    device = MagicMock()
    path = tmp_path / "requests.db"
    service = SpeakerDeviceService(device_id="device-1", config=SpeakerConfig(), device=device,
                                   request_db_path=path)
    path.rename(tmp_path / "unavailable.db")
    path.mkdir()
    result = service.execute_request(request())
    assert result.status == "uncertain"
    assert result.error_code == "request_store_unavailable"
    device.play.assert_not_called()


def test_dry_run_result_cannot_hide_real_execution(tmp_path):
    device = MagicMock()
    options = dict(device_id="device-1", config=SpeakerConfig(), device=device,
                   request_db_path=tmp_path / "requests.db")
    assert SpeakerDeviceService(**options, dry_run=True).execute_request(request()).status == "simulated"
    assert SpeakerDeviceService(**options).execute_request(request()).error_code == "request_id_conflict"
    device.play.assert_not_called()


def test_result_commit_failure_keeps_unconfirmed_claim(tmp_path):
    device = MagicMock()
    path = tmp_path / "requests.db"
    options = dict(device_id="device-1", config=SpeakerConfig(), device=device, request_db_path=path)
    service = SpeakerDeviceService(**options)
    # 장치 호출 뒤 실제 SQLite UPDATE를 실패시켜 성공 응답과 재출력을 모두 막는다.
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TRIGGER fail_result BEFORE UPDATE ON device_requests "
                     "BEGIN SELECT RAISE(ABORT, '결과 저장 실패'); END")
    result = service.execute_request(request())
    assert result.status == "uncertain"
    assert result.error_code == "request_store_unavailable"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT result_json FROM device_requests").fetchone() == (None,)
        conn.execute("DROP TRIGGER fail_result")
    assert SpeakerDeviceService(**options).execute_request(request()).error_code == "execution_unconfirmed"
    device.play.assert_called_once()


def test_json_key_order_is_irrelevant_but_invalid_values_block_output(tmp_path):
    device = MagicMock()
    service = SpeakerDeviceService(device_id="device-1", config=SpeakerConfig(), device=device,
                                   request_db_path=tmp_path / "requests.db")
    original = service.execute_request(request(payload={"text": "경고", "severity": "critical"}))
    assert service.execute_request(request(payload={"severity": "critical", "text": "경고"})) == original
    assert service.execute_request(request(request_id="invalid", payload={"volume": float("nan")})).error_code == "invalid_payload"
    device.play.assert_called_once()


@pytest.mark.parametrize("boundary", ["before_output", "after_output", "after_commit"])
def test_process_exit_does_not_replay_unconfirmed_or_completed_output(tmp_path, boundary):
    # 실제 별도 프로세스를 종료해 메모리 객체를 공유하지 않는 복구를 확인한다.
    script = '''
import os, sys
from pathlib import Path
from src.devices.speaker import SpeakerConfig
from src.edgex.speaker_device_service import SpeakerDeviceService
root, boundary = Path(sys.argv[1]), sys.argv[2]
class Device:
    def play(self, *args, **kwargs):
        if boundary == "before_output": os._exit(31)
        with (root / "physical_calls").open("a") as f: f.write("output\\n")
        if boundary == "after_output": os._exit(32)
        return True
service = SpeakerDeviceService(device_id="device-1", config=SpeakerConfig(), device=Device(),
    request_db_path=root / "requests.db")
result = service.execute_request({"request_id":"event-1:speaker:device-1", "event_id":"event-1",
    "device":"speaker", "device_id":"device-1", "action":"play", "payload":{"text":"경고"}})
assert result.status == "acknowledged"
os._exit(33)
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), boundary],
                            cwd=Path(__file__).resolve().parents[1], timeout=15, capture_output=True)
    assert result.returncode == {"before_output": 31, "after_output": 32, "after_commit": 33}[boundary], result.stderr
    device = MagicMock()
    restored = SpeakerDeviceService(device_id="device-1", config=SpeakerConfig(), device=device,
                                     request_db_path=tmp_path / "requests.db")
    assert restored.execute_request(request()).status == ("acknowledged" if boundary == "after_commit" else "uncertain")
    device.play.assert_not_called()
    marker = tmp_path / "physical_calls"
    assert (marker.read_text().splitlines() if marker.exists() else []) == ([] if boundary == "before_output" else ["output"])


def test_independent_processes_share_atomic_request_claim(tmp_path):
    script = '''
import sys, time
from pathlib import Path
from src.devices.speaker import SpeakerConfig
from src.edgex.speaker_device_service import SpeakerDeviceService
root = Path(sys.argv[1])
class Device:
    def play(self, *args, **kwargs):
        with (root / "physical_calls").open("a") as f: f.write("output\\n")
        time.sleep(0.3)
        return True
service = SpeakerDeviceService(device_id="device-1", config=SpeakerConfig(), device=Device(),
    request_db_path=root / "requests.db")
print("ready", flush=True)
sys.stdin.readline()
result = service.execute_request({"request_id":"event-1:speaker:device-1", "event_id":"event-1",
    "device":"speaker", "device_id":"device-1", "action":"play", "payload":{"text":"경고"}})
print(result.status)
'''
    processes = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path)],
                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(2)]
    try:
        # 두 프로세스의 준비 이후 함께 입력을 보내 서로 다른 메모리에서 같은 DB를 선점한다.
        import select

        for process in processes:
            assert select.select([process.stdout], [], [], 10)[0]
            assert process.stdout.readline().strip() == "ready"
        for process in processes:
            process.stdin.write("start\n")
            process.stdin.flush()
        statuses = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            statuses.append(stdout.strip())
        assert "acknowledged" in statuses
        assert set(statuses) <= {"acknowledged", "uncertain"}
        assert (tmp_path / "physical_calls").read_text().splitlines() == ["output"]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


@pytest.mark.parametrize("kind,service_class,config,action,method", CASES)
def test_runner_uses_persistent_store_with_and_without_registry(tmp_path, monkeypatch, kind, service_class, config, action, method):
    monkeypatch.setenv("DEVICE_REQUEST_DB_PATH", str(tmp_path / "runner.db"))
    monkeypatch.setenv(f"{kind.upper()}_DEVICE_ID", "device-1")
    monkeypatch.setenv(f"{kind.upper()}_DRY_RUN", "true")
    monkeypatch.setenv("EDGEX_DEVICE_REGISTRY_PATH", "")
    module = importlib.import_module(f"runners.run_{kind}_device_service")
    first = module.create_service()
    try:
        assert first.execute_request(request(kind, action)).status == "simulated"
    finally:
        first.close()
    registry = tmp_path / "devices.json"
    registry.write_text(json.dumps({"devices": [{"device_id": "device-1", "device_type": kind,
                                                "connection": {"host": ""}, "enabled": True}]}))
    monkeypatch.setenv("EDGEX_DEVICE_REGISTRY_PATH", str(registry))
    restored = module.create_service()
    try:
        assert restored.execute_request(request(kind, action, payload={"text": "변경"})).error_code == "request_id_conflict"
    finally:
        restored.close()
