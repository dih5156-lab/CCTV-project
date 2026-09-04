from unittest.mock import MagicMock

from src.devices.siren import SensorConfig
from src.edgex.siren_device_service import SirenDeviceService


def _service(*, dry_run=False):
    device = MagicMock()
    return SirenDeviceService(
        device_id="cctv-siren-01",
        config=SensorConfig(),
        device=device,
        dry_run=dry_run,
    ), device


def test_trigger_command_is_translated_to_siren_device_call():
    service, device = _service()
    device.trigger.return_value = True

    result = service.execute_request(
        {
            "request_id": "req-1",
            "event_id": "event-1",
            "device": "siren",
            "action": "trigger",
            "payload": {"event_type": "intrusion", "camera_id": "cam-01"},
        }
    )

    assert result.status == "acknowledged"
    device.trigger.assert_called_once_with("intrusion", "cam-01")


def test_stop_failure_is_reported_as_unreachable():
    service, device = _service()
    device.stop.return_value = False

    result = service.execute_request(
        {
            "request_id": "req-2",
            "event_id": "event-2",
            "device": "siren",
            "action": "stop",
            "payload": {},
        }
    )

    assert result.status == "failed"
    assert result.error_code == "device_unreachable"


def test_dry_run_does_not_call_unconnected_siren():
    service, device = _service(dry_run=True)

    result = service.execute_request(
        {
            "request_id": "req-3",
            "event_id": "event-3",
            "device": "siren",
            "action": "trigger",
            "payload": {},
        }
    )

    assert result.status == "simulated"
    device.trigger.assert_not_called()


def test_multi_device_pool_routes_to_requested_siren():
    first_device = MagicMock()
    second_device = MagicMock()
    second_device.trigger.return_value = True
    service = SirenDeviceService(
        device_id="cctv-siren-01",
        config=SensorConfig(),
        devices={"cctv-siren-01": first_device, "cctv-siren-02": second_device},
    )

    result = service.execute_request(
        {
            "request_id": "req-pool",
            "event_id": "event-pool",
            "device": "siren",
            "device_id": "cctv-siren-02",
            "action": "trigger",
            "payload": {},
        }
    )

    assert result.device_id == "cctv-siren-02"
    assert result.status == "acknowledged"
    first_device.trigger.assert_not_called()
    second_device.trigger.assert_called_once()
