from unittest.mock import MagicMock

from src.devices.signboard import SignboardConfig
from src.edgex.dabit_device_service import DabitDeviceService


def test_dabit_commands_map_to_signboard_methods():
    service = DabitDeviceService(device_id="signboard-01", config=SignboardConfig())
    service._device = MagicMock()
    service._device.display.return_value = True

    result = service.execute("cmd-1", "display", {"display_text": "테스트", "display_color": 1})

    assert result.status == "acknowledged"
    service._device.display.assert_called_once()
    assert service._device.display.call_args.kwargs["text_color"] == 1


def test_dabit_unknown_command_is_rejected():
    service = DabitDeviceService(device_id="signboard-01", config=SignboardConfig())
    result = service.execute("cmd-2", "unknown", {})
    assert result.status == "failed"
    assert result.error_code == "unsupported_command"
