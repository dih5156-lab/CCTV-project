import json
import sys
from pathlib import Path
from types import SimpleNamespace

PARSER_ROOT = Path(__file__).resolve().parents[1]
if str(PARSER_ROOT) not in sys.path:
    sys.path.insert(0, str(PARSER_ROOT))

from mqtt.manager import Manager


class RecordingProcessor:
    def __init__(self):
        self.calls = []

    def process_sensor_data(self, **kwargs):
        self.calls.append(kwargs)


def test_dcalpwan_uplink_envelope_is_forwarded_to_tlv_processor():
    processor = RecordingProcessor()
    manager = Manager(configs=None)
    manager.set_processor(processor)
    message = {
        "message_id": "messageID_uplink",
        "f_port": 1,
        "payload": "SGkhCg==",
        "is_confirmed": False,
        "is_ack": False,
        "f_cnt_up": 21,
        "rx_metadata": [{
            "gateway_info": {"gw_id": "0A1B2C3D4E5F6789"},
            "data_rate": "SF7BW125",
            "channel": 3.9225,
            "time": 1562746105470,
            "rssi": -56,
            "snr": 6.8,
        }],
    }

    manager._message_handler(
        None,
        None,
        SimpleNamespace(
            topic="0000AAAA0000AAAA/0D0D33330D0D3333/up",
            payload=json.dumps(message).encode(),
        ),
    )

    assert processor.calls == [{
        "app_id": "0000AAAA0000AAAA",
        "dev_eui": "0D0D33330D0D3333",
        "payload": "SGkhCg==",
        "channel": 3,
        "frequency": 0,
        "received_at": 1562746105470,
        "uplink_metadata": {
            "message_id": "messageID_uplink",
            "f_port": 1,
            "f_cnt_up": 21,
            "is_confirmed": False,
            "is_ack": False,
            "radio": {
                "gateway_id": "0A1B2C3D4E5F6789",
                "data_rate": "SF7BW125",
                "channel": 3.9225,
                "frequency": 0,
                "rssi": -56,
                "snr": 6.8,
            },
        },
    }]


def test_non_object_json_and_non_uplink_topic_are_ignored():
    processor = RecordingProcessor()
    manager = Manager(configs=None)
    manager.set_processor(processor)

    manager._message_handler(
        None, None, SimpleNamespace(topic="app/dev/up", payload=b"[]")
    )
    manager._message_handler(
        None,
        None,
        SimpleNamespace(topic="app/dev/join", payload=b'{"payload":"SGkhCg=="}'),
    )

    assert processor.calls == []
