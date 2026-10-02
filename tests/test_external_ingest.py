import json
import sqlite3
from base64 import b64decode
from unittest.mock import Mock

import pytest

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
    assert event["schema_version"] == "1.0"
    assert event["event_id"].startswith("evt_")
    assert event["message_type"] == "external_event"
    assert event["occurred_at"] == "2026-03-31T01:23:45Z"
    assert event["device"]["camera_id"] == "cam-01"
    assert event["event"]["event_type"] == "helmet_missing"
    assert event["event"]["source"] == "external_mqtt"


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
    assert event["message_id"] == "up-1774916301346"
    assert event["occurred_at"] == "2026-03-31T00:18:21.111Z"
    assert event["device"]["dev_eui"] == "0080e11505c9e523"
    assert event["gateway"]["gw_eui"] == "0016c001f153806a"
    assert event["gateway"]["received_at"] == "2026-03-31T00:18:21.111Z"
    assert event["raw"]["payload_base64"] == raw["payload"]
    assert event["raw"]["rx_metadata"] == raw["rx_metadata"]


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
            "SELECT event_id, topic, raw_payload, normalized_payload, republished FROM ingest_events"
        ).fetchone()

    assert row is not None
    assert row[0].startswith("evt_")
    assert row[1] == "factory/raw"
    assert json.loads(row[2])["camera_id"] == "camera-a"
    assert json.loads(row[3])["type"] == "telemetry"
    assert row[4] == 0


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


@pytest.fixture
def ingest_service(tmp_path):
    cfg = AppConfig()
    cfg.external_ingest.db_path = str(tmp_path / "ingest.db")
    service = ExternalIngestService.from_app_config(cfg)
    service._repo.init()
    service._publisher = Mock()
    service._publisher.publish_event.return_value = True
    return service


@pytest.mark.parametrize(
    "changes,field,code",
    [
        ({"timestamp": "not-a-time"}, "occurred_at", "invalid_timestamp"),
        ({"timestamp": "2026-09-18T10:00:00"}, "occurred_at", "timezone_required"),
        ({"timestamp": "2026-02-30T10:00:00Z"}, "occurred_at", "invalid_timestamp"),
        ({"timestamp": float("inf")}, "occurred_at", "invalid_timestamp"),
        ({"timestamp": None}, "occurred_at", "invalid_timestamp"),
        ({"timestamp": True}, "occurred_at", "invalid_timestamp"),
        ({"timestamp": ""}, "occurred_at", "invalid_timestamp"),
        ({"timestamp": 1e100}, "occurred_at", "invalid_timestamp"),
        ({"camera_id": None}, "device_id", "invalid_device_id"),
        ({"camera_id": "   "}, "device_id", "invalid_device_id"),
        ({"camera_id": "unknown"}, "device_id", "invalid_device_id"),
        ({"camera_id": {"id": "cam-1"}}, "device_id", "invalid_device_id"),
        ({"event_id": ""}, "event_id", "invalid_event_id"),
        ({"event_id": None}, "event_id", "invalid_event_id"),
        ({"schema_version": "99"}, "schema_version", "unsupported_schema_version"),
        ({"decoded": [1, 2]}, "decoded", "mapping_required"),
    ],
)
def test_invalid_input_is_saved_with_reason_and_not_republished(
    ingest_service, changes, field, code
):
    raw = {"camera_id": "cam-1", "type": "fall_detected", "timestamp": 1700000000}
    raw.update(changes)

    ingest_service.handle_message("factory/raw", json.dumps(raw).encode())

    ingest_service._publisher.publish_event.assert_not_called()
    with sqlite3.connect(ingest_service._repo.db_path) as conn:
        row = conn.execute(
            "SELECT raw_payload, normalized_payload, republished FROM ingest_events"
        ).fetchone()
    assert json.loads(row[0]) == raw
    rejected = json.loads(row[1])
    assert rejected["event_id"].startswith("rejected_")
    assert rejected["validation"]["status"] == "invalid"
    assert {"field": field, "code": code} in rejected["validation"]["errors"]
    assert row[2] == 0
    assert ingest_service.get_stats()["quarantined_count"] == 1


@pytest.mark.parametrize("raw", [[], None, "hello", 5, {"type": "fall_detected"}])
def test_invalid_shape_or_missing_device_is_quarantined(ingest_service, raw):
    ingest_service.handle_message("factory/raw", json.dumps(raw).encode())

    ingest_service._publisher.publish_event.assert_not_called()
    assert ingest_service.get_stats()["quarantined_count"] == 1


def test_invalid_json_preserves_original_bytes(ingest_service):
    raw_bytes = b"{bad json\xff"
    ingest_service.handle_message("factory/raw", raw_bytes)

    with sqlite3.connect(ingest_service._repo.db_path) as conn:
        row = conn.execute("SELECT raw_payload FROM ingest_events").fetchone()
    assert b64decode(json.loads(row[0])["payload_base64"]) == raw_bytes
    ingest_service._publisher.publish_event.assert_not_called()
    assert ingest_service.get_stats()["parse_fail_count"] == 1
    assert ingest_service.get_stats()["quarantined_count"] == 1


def test_quarantine_save_failure_is_not_reported_as_success(ingest_service):
    ingest_service._repo.save = Mock(side_effect=sqlite3.OperationalError("disk full"))

    with pytest.raises(sqlite3.OperationalError, match="disk full"):
        ingest_service.handle_message("factory/raw", b'{"camera_id": ""}')

    ingest_service._publisher.publish_event.assert_not_called()
    assert ingest_service.get_stats()["quarantined_count"] == 0


def test_rejected_source_id_does_not_hide_other_invalid_inputs(ingest_service):
    for timestamp in ("bad-time-one", "bad-time-two"):
        raw = {
            "camera_id": "cam-1",
            "event_id": "same-source-id",
            "timestamp": timestamp,
        }
        ingest_service.handle_message("factory/raw", json.dumps(raw).encode())

    with sqlite3.connect(ingest_service._repo.db_path) as conn:
        rows = conn.execute("SELECT event_id FROM ingest_events").fetchall()
    assert len(rows) == 2
    assert rows[0] != rows[1]


def test_valid_input_after_invalid_input_keeps_existing_id_contract(ingest_service):
    ingest_service.handle_message("factory/raw", b'{"camera_id": ""}')
    valid = {"camera_id": "cam-1", "event_id": "source-17", "timestamp": 1700000000}
    ingest_service.handle_message("factory/raw", json.dumps(valid).encode())

    ingest_service._publisher.publish_event.assert_called_once()
    published = ingest_service._publisher.publish_event.call_args.args[0]
    # 변경 전 같은 정상 입력의 ID를 유지한다.
    assert published["event_id"] == "evt_c70145bb36ab8625"
    assert published["raw"]["raw_payload"]["event_id"] == "source-17"
    assert published["occurred_at_source"] == "timestamp"
    assert ingest_service.get_stats()["quarantined_count"] == 1


def test_missing_time_uses_explicit_receipt_time_and_preserves_source_record():
    event = normalize_external_event(
        {"device_id": "sensor-1", "event_id": "source-17"},
        "factory/raw",
        received_at="2026-09-18T11:00:00+09:00",
    )
    assert event["occurred_at"] == "2026-09-18T11:00:00+09:00"
    assert event["received_at"] == event["occurred_at"]
    assert event["occurred_at_source"] == "received_at"
    assert event["event_id"].startswith("evt_")
    assert event["raw"]["raw_payload"]["event_id"] == "source-17"


@pytest.mark.parametrize(
    "timestamp",
    [
        0,
        1700000000,
        1700000000000,
        1700000000000000,
        "2026-09-18T02:00:00Z",
        "2026-09-18T11:00:00+09:00",
    ],
)
def test_valid_timestamp_forms_keep_source_value(timestamp):
    event = normalize_external_event(
        {"camera_id": "cam-1", "timestamp": timestamp}, "factory/raw"
    )
    assert event["timestamp"] == str(timestamp)
    assert event["occurred_at_source"] == "timestamp"


@pytest.mark.parametrize(
    "metadata",
    [
        {"gw_recv_time": "bad-time"},
        {"time": None, "timestamp": 1700000000},
        {"time": True},
    ],
)
def test_invalid_gateway_time_is_not_hidden_by_fallback(ingest_service, metadata):
    raw = {"dev_eui": "sensor-1", "rx_metadata": [metadata]}
    ingest_service.handle_message("lora/up", json.dumps(raw).encode())

    ingest_service._publisher.publish_event.assert_not_called()
    assert ingest_service.get_stats()["quarantined_count"] == 1


def test_gateway_epoch_zero_keeps_time_and_origin():
    event = normalize_external_event(
        {"dev_eui": "sensor-1", "rx_metadata": [{"time": 0}]}, "lora/up"
    )
    assert event["occurred_at"] == "1970-01-01T09:00:00+09:00"
    assert event["occurred_at_source"] == "rx_metadata[0]"


def test_tlv_observation_is_persisted_with_quality_and_source_record(ingest_service):
    raw = {
        "device_id": "sensor-1",
        "table": "t34954",
        "data": {"temperature_c": 0, "humidity_pct": None, "created_at": 0},
        "received_at": 1700000000000,
    }
    ingest_service.handle_message(
        "aiot/sensors/sensor-1/t34954", json.dumps(raw).encode()
    )

    published = ingest_service._publisher.publish_event.call_args.args[0]
    observation = published["observation"]
    assert observation["source_record_id"] == published["event_id"]
    assert observation["measurements"]["temperature"]["value"] == 0.0
    assert observation["measurements"]["humidity"]["quality"] == "missing"
    assert observation["observed_at"] == "1970-01-01T09:00:00+09:00"
    with sqlite3.connect(ingest_service._repo.db_path) as conn:
        row = conn.execute(
            "SELECT raw_payload, normalized_payload FROM ingest_events"
        ).fetchone()
    assert json.loads(row[0]) == raw
    assert json.loads(row[1])["observation"] == observation


@pytest.mark.parametrize(
    "raw",
    [
        {"camera_id": "cam-1", "type": "fall_detected"},
        {"device_id": "sensor-1", "table": "unknown", "data": {"temperature": 20}},
    ],
)
def test_unmapped_input_does_not_get_sensor_observation(raw):
    event = normalize_external_event(raw, "factory/raw")
    assert "observation" not in event
