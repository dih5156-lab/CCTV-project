"""EdgeX 운용 시 전광판 유휴 화면의 쓰기 주체가 한 곳인지 검증한다."""

from runners import run_signboard_device_service
from src.devices.signboard import SignboardConfig, SignboardDevice
from src.services.action_bridge import ActionBridge, AlarmDevice


def test_edgex_stack_has_one_idle_writer(monkeypatch, tmp_path):
    starts = []
    monkeypatch.setattr(SignboardDevice, "_start_idle_thread", lambda device: starts.append(device))
    monkeypatch.setenv("SIGNBOARD_HOST", "192.0.2.1")
    monkeypatch.delenv("EDGEX_DEVICE_REGISTRY_PATH", raising=False)
    bridge = ActionBridge(
        edgex_command_mode="edgex", signboard_config=SignboardConfig(host="192.0.2.1"),
        db_path=str(tmp_path / "events.db"),
    )
    service = run_signboard_device_service.create_service()
    assert starts == [service._device]
    assert bridge._device_is_available(AlarmDevice.SIGNBOARD)
    service.close()


def test_direct_mode_keeps_idle_display(monkeypatch, tmp_path):
    starts = []
    monkeypatch.setattr(SignboardDevice, "_start_idle_thread", lambda device: starts.append(device))
    config = SignboardConfig(host="192.0.2.1")
    bridge = ActionBridge(edgex_command_mode="direct", signboard_config=config, db_path=str(tmp_path / "events.db"))
    assert starts == [bridge._signboard]
    assert config.host == "192.0.2.1"
