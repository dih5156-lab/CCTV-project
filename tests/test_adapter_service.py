import pytest
from unittest.mock import MagicMock

pytest.importorskip("redis")

from src.edgex.adapter_service import EdgeXDeviceAdapterService


def test_replay_outbox_once_replays_pending_events():
    service = EdgeXDeviceAdapterService()
    service.edgex_service = MagicMock()
    service.edgex_service.get_pending_detection_events.return_value = [
        {
            "id": 1,
            "_table": "detection_outbox",
            "camera_id": "cam1",
            "event_data": {
                "camera_id": "cam1",
                "type": "fall_detected",
                "source": "rtsp://camera-1",
            },
        }
    ]
    service.edgex_service.replay_detection_event.return_value = "replay-coro"
    service._ensure_camera_registered = MagicMock(return_value=True)
    service._run_coro = MagicMock(return_value=True)

    service._replay_outbox_once()

    service._ensure_camera_registered.assert_called_once_with(
        "cam1",
        rtsp_source="rtsp://camera-1",
    )
    service.edgex_service.replay_detection_event.assert_called_once_with(
        ("detection_outbox", 1),
        "cam1",
        {
            "camera_id": "cam1",
            "type": "fall_detected",
            "source": "rtsp://camera-1",
        },
    )
    service._run_coro.assert_called_once_with("replay-coro")


def test_on_message_handles_list_payload():
    """Kuiper 센서 이벤트는 JSON 배열로 발행됨 → 개별 처리"""
    import json

    service = EdgeXDeviceAdapterService()
    service.edgex_service = MagicMock()
    service._ensure_camera_registered = MagicMock(return_value=True)
    service._run_coro = MagicMock(return_value=True)

    msg = MagicMock()
    msg.topic = "aiot/rules/sensor/tilt"
    msg.payload = json.dumps([
        {"device_id": "factory-21", "type": "tilt_alert", "angle_x": 89.3},
        {"device_id": "factory-22", "type": "tilt_alert", "angle_x": 12.5},
    ]).encode("utf-8")

    service._on_message(None, None, msg)

    assert service._run_coro.call_count == 2
    assert service._ensure_camera_registered.call_count == 2


def test_on_message_handles_dict_payload():
    """AI 이벤트는 JSON dict 로 발행됨 → 정상 처리"""
    import json

    service = EdgeXDeviceAdapterService()
    service.edgex_service = MagicMock()
    service._ensure_camera_registered = MagicMock(return_value=True)
    service._run_coro = MagicMock(return_value=True)

    msg = MagicMock()
    msg.topic = "cctv/ai/events/camera_1/person"
    msg.payload = json.dumps(
        {"camera_id": "camera_1", "type": "person"}
    ).encode("utf-8")

    service._on_message(None, None, msg)

    service._run_coro.assert_called_once()
