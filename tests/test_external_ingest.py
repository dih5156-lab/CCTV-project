import json
import sqlite3

from src.config import AppConfig
from src.services.external_ingest import ExternalIngestService, normalize_external_event


def test_normalize_external_event_maps_common_fields():
    raw = {
        "source_id": "cam-01",
        "event_type": "helmet_missing",
        "timestamp": "2026-03-31T01:23:45Z",
        "score": "0.87",
        "spec": {"temperature": 27},
        "image_path": "frames/cam-01.jpg",
    }

    event = normalize_external_event(raw, "factory/line1")

    assert event["camera_id"] == "cam-01"
    assert event["type"] == "helmet_missing"
    assert event["timestamp"] == "2026-03-31T01:23:45Z"
    assert event["confidence"] == 0.87
    assert event["metadata"]["topic"] == "factory/line1"
    assert event["metadata"]["image_path"] == "frames/cam-01.jpg"


def test_normalize_external_event_maps_lora_payload():
    raw = {
        "message_id": "up-1774916301346",
        "app_eui": "a000000000000001",
        "dev_eui": "0080e11505c9e523",
        "payload": "AAHBEACIjQDEAEhJEADEAUK1pnTEAr8XrPbkFY5pyxLe",
        "is_confirmed": False,
        "f_port": 2,
        "f_cnt_up": 468,
        "rx_metadata": [
            {
                "gateway_info": {
                    "gw_eui": "0016c001f153806a",
                    "latitude": 37.44001,
                    "longitude": 127.17662,
                    "altitude": 167,
                    "channel_plan": "KR920",
                },
                "modulation": "LORA",
                "data_rate": "SF7BW125",
                "coding_rate": "4/5",
                "timestamp": 1779712380,
                "time": 1774916300100,
                "gps_time": 1458951518100,
                "channel": 7,
                "frequency": 923300000,
                "rssi": -68,
                "snr": 13.8,
                "gw_recv_time": "2026-03-31T00:18:21.111Z",
            }
        ],
    }

    event = normalize_external_event(raw, "lora/up")

    assert event["camera_id"] == "0080e11505c9e523"
    assert event["type"] == "lora_uplink"
    assert event["timestamp"] == "2026-03-31T00:18:21.111Z"
    assert event["metadata"]["sensor_type"] == "lora"
    assert event["metadata"]["source_id"] == "0080e11505c9e523"
    assert event["metadata"]["spec"]["app_eui"] == "a000000000000001"
    assert event["metadata"]["spec"]["f_port"] == 2
    assert event["metadata"]["telemetry"]["rssi"] == -68
    assert event["metadata"]["telemetry"]["snr"] == 13.8
    assert event["metadata"]["spec"]["gateway"]["gw_eui"] == "0016c001f153806a"
    assert event["metadata"]["payload_base64"] == raw["payload"]


def test_service_saves_normalized_event_to_sqlite(tmp_path):
    cfg = AppConfig()
    cfg.external_ingest.db_path = str(tmp_path / "ingest_events.db")
    cfg.external_ingest.republish_enabled = False

    service = ExternalIngestService.from_app_config(cfg)
    service._repo.init()
    service.handle_message(
        "factory/raw",
        json.dumps({"camera_id": "camera-a", "type": "telemetry"}).encode("utf-8"),
    )

    with sqlite3.connect(cfg.external_ingest.db_path) as conn:
        row = conn.execute(
            "SELECT topic, raw_payload, normalized_payload, republished FROM ingest_events"
        ).fetchone()

    assert row is not None
    assert row[0] == "factory/raw"
    assert json.loads(row[1])["camera_id"] == "camera-a"
    assert json.loads(row[2])["type"] == "telemetry"
    assert row[3] == 0


def test_service_republishes_when_enabled(tmp_path):
    cfg = AppConfig()
    cfg.external_ingest.db_path = str(tmp_path / "ingest_events.db")
    cfg.external_ingest.republish_enabled = True

    service = ExternalIngestService.from_app_config(cfg)
    service._repo.init()

    published = []

    class StubPublisher:
        def publish_event(self, event):
            published.append(event)
            return True

        def disconnect(self):
            return None

    service._publisher = StubPublisher()
    service.handle_message(
        "factory/raw",
        json.dumps({"camera_id": "camera-b", "type": "image_ready"}).encode("utf-8"),
    )

    assert len(published) == 1
    assert published[0]["camera_id"] == "camera-b"
    assert service.get_stats()["republish_count"] == 1


def test_invalid_json_increments_parse_fail_count(tmp_path):
    cfg = AppConfig()
    cfg.external_ingest.db_path = str(tmp_path / "ingest_events.db")

    service = ExternalIngestService.from_app_config(cfg)
    service._repo.init()
    service.handle_message("factory/raw", b"{bad json")

    assert service.get_stats()["parse_fail_count"] == 1
