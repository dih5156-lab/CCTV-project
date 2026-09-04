from unittest.mock import MagicMock

from src.devices.speaker import SpeakerConfig
from src.edgex.speaker_device_service import SpeakerDeviceService


def _service():
    device = MagicMock()
    return SpeakerDeviceService(
        device_id="cctv-speaker-01",
        config=SpeakerConfig(),
        device=device,
    ), device


def test_play_command_is_translated_to_speaker_device_call():
    service, device = _service()
    device.play.return_value = True

    result = service.execute_request(
        {
            "request_id": "req-1",
            "event_id": "event-1",
            "device": "speaker",
            "action": "play",
            "payload": {
                "event_type": "fall_detected",
                "severity": "critical",
                "camera_id": "cam-01",
                "text": "낙상 위험이 감지되었습니다.",
            },
        }
    )

    assert result.status == "acknowledged"
    assert result.request_id == "req-1"
    device.play.assert_called_once_with(
        "fall_detected",
        "critical",
        "cam-01",
        text="낙상 위험이 감지되었습니다.",
    )


def test_unsupported_command_returns_failed_result_without_device_call():
    service, device = _service()

    result = service.execute_request(
        {
            "request_id": "req-2",
            "event_id": "event-2",
            "device": "speaker",
            "action": "display",
            "payload": {},
        }
    )

    assert result.status == "failed"
    assert result.error_code == "unsupported_command"
    device.play.assert_not_called()


def test_failed_interm_response_is_converted_to_device_unreachable():
    service, device = _service()
    device.stop.return_value = False

    result = service.execute_request(
        {
            "request_id": "req-3",
            "event_id": "event-3",
            "device": "speaker",
            "action": "stop",
            "payload": {},
        }
    )

    assert result.status == "failed"
    assert result.error_code == "device_unreachable"


def test_command_result_can_be_published_as_json_ready_dictionary(tmp_path):
    service, device = _service()
    device.power_on.return_value = False

    result = service.execute_request(
        {
            "request_id": "req-4",
            "event_id": "event-4",
            "device": "speaker",
            "action": "power_on",
            "payload": {},
        }
    )

    assert result.to_dict() == {
        "request_id": "req-4",
        "event_id": "event-4",
        "device_id": "cctv-speaker-01",
        "status": "failed",
        "error_code": "device_unreachable",
    }


def test_dry_run_returns_simulated_without_calling_unconnected_speaker(tmp_path):
    device = MagicMock()
    service = SpeakerDeviceService(
        device_id="cctv-speaker-01",
        config=SpeakerConfig(),
        device=device,
        dry_run=True,
    )

    result = service.execute_request(
        {
            "request_id": "req-5",
            "event_id": "event-5",
            "device": "speaker",
            "action": "play",
            "payload": {"text": "연결 확인"},
        }
    )

    assert result.status == "simulated"
    assert result.error_code is None
    device.play.assert_not_called()


def test_multi_device_pool_routes_to_requested_speaker():
    first_device = MagicMock()
    second_device = MagicMock()
    second_device.play.return_value = True
    service = SpeakerDeviceService(
        device_id="cctv-speaker-01",
        config=SpeakerConfig(),
        devices={"cctv-speaker-01": first_device, "cctv-speaker-02": second_device},
    )

    result = service.execute_request(
        {
            "request_id": "req-pool",
            "event_id": "event-pool",
            "device": "speaker",
            "device_id": "cctv-speaker-02",
            "action": "play",
            "payload": {"text": "두 번째 스피커"},
        }
    )

    assert result.device_id == "cctv-speaker-02"
    assert result.status == "acknowledged"
    first_device.play.assert_not_called()
    second_device.play.assert_called_once()
