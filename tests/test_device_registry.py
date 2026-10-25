from src.edgex.device_registry import DeviceRegistry


def test_registry_resolves_devices_by_type_and_camera():
    registry = DeviceRegistry.from_file("config/output_devices.json")

    targets = registry.resolve("speaker", camera_id="cam-01")

    assert [target.device_id for target in targets] == ["cctv-speaker-01"]


def test_registry_does_not_route_a_device_to_another_camera():
    registry = DeviceRegistry.from_file("config/output_devices.json")

    assert registry.resolve("siren", camera_id="cam-99") == []


def test_live_camera_routes_only_to_verified_signboard():
    # 실물 확인된 전광판만 연결하고 미가동 스피커·미보유 사이렌으로 확대하지 않는다.
    registry = DeviceRegistry.from_file("config/output_devices.json")

    assert [target.device_id for target in registry.resolve("signboard", camera_id="camera_1")] == [
        "cctv-signboard-01"
    ]
    assert registry.resolve("speaker", camera_id="camera_1") == []
    assert registry.resolve("siren", camera_id="camera_1") == []
    assert registry.resolve("signboard", camera_id="camera_other") == []
