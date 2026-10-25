"""완료 증거의 요청 소유권·영속 저장·읽기 전용 상태 대조를 검증한다."""

import json
import sqlite3
from datetime import datetime
from unittest.mock import Mock

import pytest

from src.devices.signboard import SignboardConfig, SignboardDevice, _DabitClient
from src.devices.siren import SensorConfig
from src.devices.speaker import SpeakerConfig
from src.edgex.command_contract import handle_mqtt_request
from src.edgex.dabit_device_service import DabitDeviceService
from src.edgex.siren_device_service import SirenDeviceService
from src.edgex.speaker_device_service import SpeakerDeviceService
from tests.test_action_device_queue import acknowledge, enqueue, make_queue, queue, rows


def make_service(tmp_path):
    config = SignboardConfig(host="192.0.2.1", idle_enabled=False)
    device = SignboardDevice(config)
    device._client = Mock()
    for method in ("set_brightness", "send_title", "send_context"):
        getattr(device._client, method).return_value = True
    service = DabitDeviceService(device_id="board-1", config=config, device=device,
                                 request_db_path=tmp_path / "requests.db")
    request = {"request_id": "one:signboard:board-1", "event_id": "one", "device": "signboard",
               "device_id": "board-1", "action": "display", "payload": {"text": "경보"}}
    service.on_completion = Mock(return_value=True)
    return service, device, request


def idle(device):
    device._last_event_ts -= device.config.display_time + 1
    device._send_idle_frame(device._client, datetime.now())


def test_completion_is_saved_before_publish_and_survives_restart(tmp_path):
    service, device, request = make_service(tmp_path)
    result = service.execute_request(request)
    assert result.status == "acknowledged" and not result.execution_completed
    assert not service.get_request_status(request["request_id"])["resource_idle"]
    def publish(result):
        assert service.get_request_status(result["request_id"])["execution_completed"] is True
        return False  # 발행 유실 후에도 저장 결과로만 대조한다.
    service.on_completion.side_effect = publish
    idle(device)
    assert service.on_completion.call_count == 1
    restarted, _, _ = make_service(tmp_path)
    restored = restarted.execute_request(request)
    assert restored.execution_completed is True and restored.resource_idle is True
    assert restored.completion_evidence == "dabit_idle_frame_ack"
    restarted._device._client.send_context.assert_not_called()
    status = handle_mqtt_request(restarted, {**request, "action": "request_status"}, "signboard")
    assert status["execution_completed"] is True and status["durable"] is True


@pytest.mark.parametrize("response", [b"", b"OK!", b"![000F!]", b"![0000!", b"![0000!]junk"])
def test_unverified_protocol_response_never_proves_completion(response):
    client = _DabitClient("192.0.2.1", 5000, 1)
    client._send = Mock(return_value=response)
    try:
        assert client.send_context("경보", 2, 1, 0, 5) is False
    except RuntimeError:
        assert response == b"![000F!]"


def test_recorded_positive_protocol_frames_are_recognized():
    client = _DabitClient("192.0.2.1", 5000, 1)
    client._send = Mock(side_effect=[b"![00500!]", b"![0000!]", b"![0000!]"])
    assert client.set_brightness(10) is True
    assert client.send_title("제목") is True
    assert client.send_context("본문", 2, 1, 0, 5) is True


@pytest.mark.parametrize("phase,method", [(phase, method) for phase in ("display", "idle")
                                         for method in ("set_brightness", "send_title", "send_context")])
def test_every_frame_requires_positive_ack(tmp_path, phase, method):
    service, device, request = make_service(tmp_path)
    if phase == "display":
        getattr(device._client, method).return_value = False
    service.execute_request(request)
    getattr(device._client, method).return_value = phase != "idle"
    idle(device)
    service.on_completion.assert_not_called()
    assert not service.get_request_status(request["request_id"])["execution_completed"]


def test_timer_alone_does_not_complete_and_callback_is_one_shot(tmp_path):
    service, device, request = make_service(tmp_path)
    service.execute_request(request)
    device._send_idle_frame(device._client, datetime.now())
    service.on_completion.assert_not_called()
    idle(device)
    idle(device)
    assert service.on_completion.call_count == 1


@pytest.mark.parametrize("interruption", ["display", "failed_display", "clear", "power_off"])
def test_other_output_cannot_complete_original_request(tmp_path, interruption):
    service, device, request = make_service(tmp_path)
    service.execute_request(request)
    if interruption in {"display", "failed_display"}:
        if interruption == "failed_display":
            device._client.send_title.side_effect = OSError("실패")
        device.display("다른 경보")
        device._client.send_title.side_effect = None
    else:
        getattr(device, interruption)()
    idle(device)
    service.on_completion.assert_not_called()


def test_completion_commit_failure_keeps_owner_and_retries_without_alert_replay(tmp_path):
    service, device, request = make_service(tmp_path)
    service.execute_request(request)
    with sqlite3.connect(tmp_path / "requests.db") as conn:
        conn.execute("CREATE TRIGGER reject_completion BEFORE UPDATE ON device_requests BEGIN SELECT RAISE(ABORT, 'full'); END")
    idle(device)
    service.on_completion.assert_not_called()
    assert not service.get_request_status(request["request_id"])["execution_completed"]
    with sqlite3.connect(tmp_path / "requests.db") as conn:
        conn.execute("DROP TRIGGER reject_completion")
    idle(device)
    assert service.on_completion.call_count == 1
    assert sum(call.args[0] == "경보" for call in device._client.send_context.call_args_list) == 1


def test_early_idle_before_first_result_commit_waits_for_next_observation(tmp_path, monkeypatch):
    service, device, request = make_service(tmp_path)
    finish = service._request_results._finish
    def early(*args):
        idle(device)
        service.on_completion.assert_not_called()
        return finish(*args)
    monkeypatch.setattr(service._request_results, "_finish", early)
    service.execute_request(request)
    idle(device)
    assert service.on_completion.call_count == 1


def test_restart_before_idle_does_not_guess_completion(tmp_path):
    service, device, request = make_service(tmp_path)
    service.execute_request(request)
    restarted, fresh_device, _ = make_service(tmp_path)
    idle(fresh_device)
    assert not restarted.get_request_status(request["request_id"])["execution_completed"]
    restarted.on_completion.assert_not_called()


@pytest.mark.parametrize("kind,cls,config,action", [
    ("signboard", DabitDeviceService, SignboardConfig, "display"),
    ("speaker", SpeakerDeviceService, SpeakerConfig, "play"),
    ("siren", SirenDeviceService, SensorConfig, "trigger"),
])
def test_status_query_is_read_only_and_ack_does_not_prove_idle(tmp_path, kind, cls, config, action):
    device = Mock()
    service = cls(device_id="d", config=config(), device=device, request_db_path=tmp_path / "requests.db")
    request = {"request_id": f"e:{kind}:d", "event_id": "e", "device": kind,
               "device_id": "d", "action": action, "payload": {"text": "경보"}}
    service.execute_request(request)
    before = list(device.mock_calls)
    raw = (tmp_path / "requests.db").read_bytes()
    status = handle_mqtt_request(service, {**request, "action": "request_status"}, kind)
    assert status["status"] == "acknowledged" and status.get("resource_idle") is not True
    assert device.mock_calls == before and (tmp_path / "requests.db").read_bytes() == raw


@pytest.mark.parametrize("changes", [{"request_id": ""}, {"device_id": "other"}, {"device": "speaker"},
                                      {"event_id": "other"}, {"request_id": []}])
def test_invalid_status_query_cannot_execute(tmp_path, changes):
    service, device, request = make_service(tmp_path)
    with pytest.raises(ValueError):
        handle_mqtt_request(service, {**request, "action": "request_status", **changes}, "signboard")
    device._client.assert_not_called()
    assert device._client.mock_calls == []


def test_unknown_query_creates_no_execution_record(tmp_path):
    service, device, request = make_service(tmp_path)
    result = handle_mqtt_request(service, {**request, "action": "request_status"}, "signboard")
    assert result["status"] == "unknown" and result["retry_allowed"] is False
    with sqlite3.connect(tmp_path / "requests.db") as conn:
        assert conn.execute("SELECT count(*) FROM device_requests").fetchone()[0] == 0
    assert device._client.mock_calls == []


def test_reconcile_repairs_queue_after_action_commit_and_callback_failure(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    queue(bridge).dispatch_ready()
    owner = rows(bridge)[0]
    bridge._edgex_result_handler.on_result = Mock(side_effect=sqlite3.OperationalError("full"))
    with pytest.raises(sqlite3.OperationalError):
        acknowledge(bridge, owner, execution_completed=True, resource_idle=True)
    queue(bridge).recover()
    queue(bridge).reconcile()
    assert rows(bridge)[0]["state"] == "complete"
    queue(bridge).dispatch_ready()
    assert rows(bridge)[1]["state"] == "sent"


def test_reconcile_only_publishes_throttled_status_query_for_active_owner(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    queue(bridge).dispatch_ready()
    queue(bridge).recover()
    publish = bridge._executor._publish_edgex_command
    publish.reset_mock()
    queue(bridge).reconcile()
    queue(bridge).reconcile()
    assert publish.call_count == 1
    request = publish.call_args.args[1]
    assert request["action"] == "request_status" and request["payload"] == {}
    assert request["request_id"] == rows(bridge)[0]["request_id"]
    assert [row["state"] for row in rows(bridge)] == ["uncertain", "queued"]


def test_missing_action_database_is_not_recreated_by_reconciliation(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    queue(bridge).dispatch_ready()
    path = bridge._repo.db_path
    path.rename(tmp_path / "preserved-events.db")
    bridge._executor._publish_edgex_command.reset_mock()
    with pytest.raises(sqlite3.OperationalError):
        queue(bridge).reconcile()
    assert not path.exists() and rows(bridge)[0]["state"] == "sent"
    bridge._executor._publish_edgex_command.assert_not_called()


def test_device_completion_lookup_releases_queue_without_replaying_alert(tmp_path):
    bridge = make_queue(tmp_path)
    enqueue(bridge, "one")
    enqueue(bridge, "two")
    queue(bridge).dispatch_ready()
    row = rows(bridge)[0]
    service, device, _ = make_service(tmp_path)
    request = json.loads(row["wire_json"])
    service.execute_request(request)
    service.on_completion.return_value = False
    idle(device)
    queue(bridge).recover()
    def publish(topic, request):
        assert request["action"] == "request_status"
        result = handle_mqtt_request(service, request, "signboard")
        bridge._handle_edgex_result("edgex/results/cctv/signboard", result)
        return True
    queue(bridge).publish = publish
    queue(bridge).reconcile()
    assert rows(bridge)[0]["state"] == "complete"
    assert sum(call.args[0] == request["payload"]["text"] for call in device._client.send_context.call_args_list) == 1
