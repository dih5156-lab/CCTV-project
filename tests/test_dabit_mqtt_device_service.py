from unittest.mock import MagicMock

from src.devices.signboard import SignboardConfig
from src.edgex.dabit_device_service import DabitDeviceService


def _service(*, dry_run=False):
    device = MagicMock()
    return DabitDeviceService(
        device_id="cctv-signboard-01",
        config=SignboardConfig(),
        device=device,
        dry_run=dry_run,
    ), device


def test_display_command_is_translated_to_dabit_device_call():
    service, device = _service()
    device.display.return_value = True

    result = service.execute_request(
        {
            "request_id": "req-1",
            "event_id": "event-1",
            "device": "signboard",
            "action": "display",
            "payload": {"text": "위험", "title": "경고!"},
        }
    )

    assert result.status == "acknowledged"
    display_kwargs = device.display.call_args.kwargs
    assert display_kwargs["text"] == "위험"
    assert display_kwargs["title"] == "경고!"


def test_display_command_accepts_profile_display_text_field():
    service, device = _service()
    device.display.return_value = True

    result = service.execute_request(
        {
            "request_id": "req-profile",
            "event_id": "event-profile",
            "device": "signboard",
            "action": "display",
            "payload": {"display_text": "프로파일 문구"},
        }
    )

    assert result.status == "acknowledged"
    assert device.display.call_args.kwargs["text"] == "프로파일 문구"


def test_dry_run_does_not_call_unconnected_dabit_device():
    service, device = _service(dry_run=True)

    result = service.execute_request(
        {
            "request_id": "req-2",
            "event_id": "event-2",
            "device": "signboard",
            "action": "clear",
            "payload": {},
        }
    )

    assert result.status == "simulated"
    device.clear.assert_not_called()


def test_multi_device_pool_routes_to_requested_signboard():
    first_device = MagicMock()
    second_device = MagicMock()
    second_device.display.return_value = True
    service = DabitDeviceService(
        device_id="cctv-signboard-01",
        config=SignboardConfig(),
        devices={"cctv-signboard-01": first_device, "cctv-signboard-02": second_device},
    )

    result = service.execute_request(
        {
            "request_id": "req-pool",
            "event_id": "event-pool",
            "device": "signboard",
            "device_id": "cctv-signboard-02",
            "action": "display",
            "payload": {"text": "두 번째 전광판"},
        }
    )

    assert result.device_id == "cctv-signboard-02"
    assert result.status == "acknowledged"
    first_device.display.assert_not_called()
    second_device.display.assert_called_once()
