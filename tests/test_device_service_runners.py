import importlib
import json

import pytest


@pytest.mark.parametrize(
    ("module_name", "device_type", "prefix"),
    [
        ("runners.run_speaker_device_service", "speaker", "SPEAKER"),
        ("runners.run_siren_device_service", "siren", "SIREN"),
        ("runners.run_signboard_device_service", "signboard", "SIGNBOARD"),
    ],
)
def test_runner_builds_client_pool_from_registry(tmp_path, monkeypatch, module_name, device_type, prefix):
    """레지스트리의 여러 장치가 러너 클라이언트 풀로 생성되는지 확인한다."""
    registry_path = tmp_path / "output_devices.json"
    registry_path.write_text(
        json.dumps(
            {
                "devices": [
                    {
                        "device_id": f"cctv-{device_type}-01",
                        "device_type": device_type,
                        "connection": {"host": "", "port": 80},
                        "enabled": True,
                    },
                    {
                        "device_id": f"cctv-{device_type}-02",
                        "device_type": device_type,
                        "connection": {"host": "", "port": 81},
                        "enabled": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EDGEX_DEVICE_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv(f"{prefix}_DRY_RUN", "true")

    module = importlib.import_module(module_name)
    service = module.create_service()
    try:
        assert service.device_ids == (
            f"cctv-{device_type}-01",
            f"cctv-{device_type}-02",
        )
    finally:
        service.close()
