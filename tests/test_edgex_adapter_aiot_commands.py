import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from runners.run_edgex_adapter import configure_aiot_commands
from src.edgex.adapter_service import EdgeXDeviceAdapterService
from src.services.aiot_command_service import AiotCommandService


def test_aiot_commands_disabled_by_default():
    adapter = EdgeXDeviceAdapterService()
    assert adapter.aiot_commands_enabled is False
    assert adapter.aiot_command_topic == "edgex/commands/cctv/jetson-01/#"


def test_aiot_command_message_routes_to_service():
    command_service = Mock()
    adapter = EdgeXDeviceAdapterService(
        aiot_commands_enabled=True,
        aiot_jetson_id="edge-01",
        aiot_command_service=command_service,
    )
    message = SimpleNamespace(
        payload=json.dumps({"message_type": "ai_query_request", "request_id": "q-1"}).encode()
    )
    adapter._on_aiot_message(None, None, message)
    command_service.handle.assert_called_once_with(
        {"message_type": "ai_query_request", "request_id": "q-1"}
    )


def test_malformed_aiot_command_is_ignored():
    command_service = Mock()
    adapter = EdgeXDeviceAdapterService(
        aiot_commands_enabled=True, aiot_command_service=command_service
    )
    adapter._on_aiot_message(None, None, SimpleNamespace(payload=b"not-json"))
    command_service.handle.assert_not_called()


def test_configure_aiot_commands_builds_service_from_environment(tmp_path, monkeypatch):
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    monkeypatch.setenv("AIOT_COMMANDS_ENABLED", "true")
    monkeypatch.setenv("AIOT_COMMAND_DB", str(tmp_path / "commands.db"))
    monkeypatch.setenv("APPEARANCES_DB", str(tmp_path / "appearances.db"))
    monkeypatch.setenv("APPEARANCE_CROP_DIR", str(crop_dir))
    monkeypatch.setenv("AIOT_ALLOWED_UPLOAD_HOSTS", "uploads.example.com")
    adapter = EdgeXDeviceAdapterService()
    configure_aiot_commands(adapter)
    assert adapter.aiot_commands_enabled is True
    assert isinstance(adapter.aiot_command_service, AiotCommandService)


def test_jetson_compose_keeps_aiot_disabled_and_mounts_runtime_data():
    compose = (Path(__file__).resolve().parents[1] / "docker-compose.jetson.yml").read_text()
    adapter_section = compose.split("  cctv-edgex-adapter:", 1)[1].split(
        "  aiot-parser:", 1
    )[0]
    assert "AIOT_COMMANDS_ENABLED: ${AIOT_COMMANDS_ENABLED:-false}" in adapter_section
    assert "target: /app/data" in adapter_section
