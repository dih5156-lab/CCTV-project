from unittest.mock import MagicMock

from src.devices.signboard import SignboardConfig
from src.edgex.dabit_device_service import DabitDeviceService


def test_dabit_commands_map_to_signboard_methods():
    service = DabitDeviceService(device_id="signboard-01", config=SignboardConfig())
    service._device = MagicMock()
    service._device.display.return_value = True

    result = service.execute("cmd-1", "display", {"display_text": "테스트", "display_color": 1})

    assert result.status == "acknowledged"
    assert result.request_id == "cmd-1"
    service._device.display.assert_called_once()
    assert service._device.display.call_args.kwargs["text_color"] == 1


def test_dabit_unknown_command_is_rejected():
    service = DabitDeviceService(device_id="signboard-01", config=SignboardConfig())
    result = service.execute("cmd-2", "unknown", {})
    assert result.status == "failed"
    assert result.error_code == "unsupported_command"


def test_dabit_duplicate_request_id_does_not_repeat_physical_display():
    device = MagicMock()
    device.display.return_value = True
    service = DabitDeviceService(
        device_id="signboard-01",
        config=SignboardConfig(),
        device=device,
    )
    request = {
        "request_id": "req-duplicate",
        "event_id": "event-duplicate",
        "device": "signboard",
        "device_id": "signboard-01",
        "action": "display",
        "payload": {"text": "중복 방지"},
    }

    first = service.execute_request(request)
    second = service.execute_request(request)

    assert first.status == second.status == "acknowledged"
    device.display.assert_called_once()
