from unittest.mock import MagicMock

from src.core.sensor_detection import SensorEventDetector
from src.protocols.mqtt import MqttEventPublisher
from src.services.sensor_bridge import SensorBridgeService


def test_process_decoded_uplink_publishes_sensor_event():
    publisher = MqttEventPublisher()
    publisher.publish_event = MagicMock(return_value=True)
    bridge = SensorBridgeService(
        publisher=publisher,
        detector=SensorEventDetector(),
    )

    uplink = {
        "app_eui": "a000000000000001",
        "dev_eui": "0080e11505c9e23c",
        "f_port": 2,
        "f_cnt_up": 75,
        "rx_metadata": [
            {
                "channel": 6,
                "frequency": 923100000,
                "rssi": -68,
                "snr": 13.8,
                "time": 1774938420097,
            }
        ],
    }
    decoded = {
        "tableName": "t34957",
        "data": {
            "tableName": "t34957",
            "temperature": 27.859375,
            "angle_x": 88.327866,
            "angle_y": 2.1585798,
        },
    }

    published = bridge.process_decoded_uplink(uplink, decoded)

    assert len(published) == 1
    assert published[0]["camera_id"] == "0080e11505c9e23c"
    assert published[0]["type"] == "tilt_alert"
    publisher.publish_event.assert_called_once()
