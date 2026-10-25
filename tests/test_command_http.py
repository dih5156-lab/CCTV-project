from unittest.mock import MagicMock

import pytest

from src.devices.speaker import SpeakerConfig
from src.edgex.command_http import handle_command_request
from src.edgex.speaker_device_service import SpeakerCommandResult, SpeakerDeviceService


class _FakeService:
    def execute_request(self, request):
        return SpeakerCommandResult(
            request_id=request["request_id"],
            event_id=request["event_id"],
            device_id="cctv-speaker-01",
            status="simulated",
        )


def test_core_command_path_is_converted_to_common_request():
    status_code, result = handle_command_request(
        _FakeService(),
        "cctv-speaker-01",
        "/api/v3/device/name/cctv-speaker-01/play",
        {"event_id": "event-1", "text": "안내"},
        "cmd-1",
    )

    assert status_code == 200
    assert result["apiVersion"] == "v3"
    assert result["statusCode"] == 200
    assert result["request_id"] == "cmd-1"
    assert result["status"] == "simulated"


def test_core_command_path_rejects_other_device():
    status_code, result = handle_command_request(
        _FakeService(),
        "cctv-speaker-01",
        "/api/v3/device/name/other-speaker/play",
        {"event_id": "event-1"},
        "cmd-2",
    )

    assert status_code == 404
    assert result["error_code"] == "device_not_found"


def test_core_command_path_supports_siren_device_type():
    status_code, result = handle_command_request(
        _FakeService(),
        "cctv-siren-01",
        "/api/v3/device/name/cctv-siren-01/trigger",
        {"event_id": "event-3"},
        "cmd-3",
        device_type="siren",
    )

    assert status_code == 200
    assert result["status"] == "simulated"


def test_core_command_path_accepts_registered_multiple_devices():
    """다중 장치 서비스가 등록된 장치 ID만 허용하는지 확인한다."""
    status_code, result = handle_command_request(
        _FakeService(),
        ("cctv-speaker-01", "cctv-speaker-02"),
        "/api/v3/device/name/cctv-speaker-02/play",
        {"event_id": "event-4"},
        "cmd-4",
    )

    assert status_code == 200
    assert result["status"] == "simulated"


def test_core_command_path_rejects_unregistered_multiple_device():
    """다중 장치 목록에 없는 물리 장치 ID를 차단하는지 확인한다."""
    status_code, result = handle_command_request(
        _FakeService(),
        ("cctv-speaker-01", "cctv-speaker-02"),
        "/api/v3/device/name/unknown-speaker/play",
        {"event_id": "event-5"},
        "cmd-5",
    )

    assert status_code == 404
    assert result["error_code"] == "device_not_found"


def test_http_command_executes_selected_physical_device():
    first, second = MagicMock(), MagicMock()
    second.play.return_value = True
    service = SpeakerDeviceService(device_id="speaker-1", config=SpeakerConfig(),
        device=first, devices={"speaker-1": first, "speaker-2": second})
    status_code, result = handle_command_request(service, service.device_ids,
        "/api/v3/device/name/speaker-2/play", {"event_id": "event-1", "text": "안내"},
        "event-1:speaker:speaker-2")
    assert status_code == 200
    assert result["device_id"] == "speaker-2"
    first.play.assert_not_called()
    second.play.assert_called_once()


@pytest.mark.parametrize("command_id", ["", " "])
def test_http_command_requires_explicit_id(command_id):
    service = MagicMock()
    status, result = handle_command_request(service, "speaker-1", "/api/v3/device/name/speaker-1/play",
                                            {"text": "안내"}, command_id)
    assert status == 400
    assert result["error_code"] == "missing_request_id"
    service.execute_request.assert_not_called()


def test_http_and_mqtt_share_identity_and_report_conflict(tmp_path):
    device = MagicMock()
    service = SpeakerDeviceService(device_id="speaker-1", config=SpeakerConfig(), device=device,
                                   request_db_path=tmp_path / "requests.db")
    request_id = "event-1:speaker:speaker-1"
    service.execute_request({"request_id": request_id, "event_id": "event-1", "device": "speaker",
                             "device_id": "speaker-1", "action": "play", "payload": {"text": "안내"}})
    path = "/api/v3/device/name/speaker-1/play"
    assert handle_command_request(service, service.device_ids, path,
                                  {"event_id": "event-1", "text": "안내"}, request_id)[0] == 200
    status, result = handle_command_request(service, service.device_ids, path,
                                            {"event_id": "event-1", "text": "변경"}, request_id)
    assert status == 409
    assert result["error_code"] == "request_id_conflict"
    device.play.assert_called_once()
