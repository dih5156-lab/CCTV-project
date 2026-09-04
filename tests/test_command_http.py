from src.edgex.command_http import handle_command_request
from src.edgex.speaker_device_service import SpeakerCommandResult


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
