from scripts.health.check_cameras_config import validate_output_mappings


def test_validate_output_mappings_accepts_enabled_camera_with_enabled_target():
    cameras = [{"id": "sample_eval", "enabled": True}]
    devices = [
        {
            "device_id": "signboard-1",
            "device_type": "signboard",
            "enabled": True,
            "camera_ids": ["sample_eval"],
        }
    ]

    assert validate_output_mappings(cameras, devices) == []


def test_validate_output_mappings_reports_enabled_camera_without_target():
    cameras = [{"id": "sample_eval", "enabled": True}]
    devices = [
        {
            "device_id": "signboard-1",
            "device_type": "signboard",
            "enabled": True,
            "camera_ids": ["other-camera"],
        }
    ]

    assert validate_output_mappings(cameras, devices) == [
        "camera[sample_eval] has no enabled output device mapping"
    ]
