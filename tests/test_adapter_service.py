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
        1,
        "cam1",
        {
            "camera_id": "cam1",
            "type": "fall_detected",
            "source": "rtsp://camera-1",
        },
    )
    service._run_coro.assert_called_once_with("replay-coro")
