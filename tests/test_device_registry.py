from src.edgex.device_registry import DeviceRegistry


def test_registry_resolves_devices_by_type_and_camera():
    registry = DeviceRegistry.from_file("config/output_devices.json")

    targets = registry.resolve("speaker", camera_id="cam-01")

    assert [target.device_id for target in targets] == ["cctv-speaker-01"]


def test_registry_does_not_route_a_device_to_another_camera():
    registry = DeviceRegistry.from_file("config/output_devices.json")

    assert registry.resolve("siren", camera_id="cam-99") == []
