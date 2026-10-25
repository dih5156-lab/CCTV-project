"""추적 ID의 내부 정수·외부 문자열 계약과 재전송 호환을 검증한다."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.canonical_event import canonicalize_event_payload
from src.core._event_publish import normalize_event_queue_item
from src.core.events import DetectionEvent, EventType
from src.edgex._payload_mixin import _PayloadMixin
from src.protocols.mqtt_publisher import MqttEventPublisher
from src.video_observation import build_video_observation


@pytest.mark.parametrize("track_id", [0, 39, 2**53 - 1, 2**53, 2**63 - 1,
                                    2**63, 12123278610165399591,
                                    12123278610165399592, 2**64 - 2])
def test_queue_event_preserves_exact_id_at_all_wire_paths(track_id):
    event = DetectionEvent(EventType.PERSON, 1, 2, 30, 40, 0.9, 1700000000,
                           object_id=track_id,
                           metadata={"camera_id": "cam1", "generation": 2,
                                     "person_object_id": track_id})
    payload, _, _ = normalize_event_queue_item(event)
    assert event.object_id == track_id
    assert event.metadata["person_object_id"] == track_id
    assert payload["object_id"] == str(track_id)
    assert payload["raw"]["object_id"] == str(track_id)
    assert payload["observation"]["track_id"]["value"] == str(track_id)
    assert payload["metadata"]["person_object_id"] == str(track_id)
    assert payload["raw"]["metadata"]["person_object_id"] == str(track_id)
    assert payload["metadata"]["generation"] == 2
    assert canonicalize_event_payload(payload) == payload
    assert json.loads(json.dumps(payload))["object_id"] == str(track_id)


@pytest.mark.parametrize("value,quality", [
    ("0", "valid"), ("12123278610165399591", "valid"),
    (None, "missing"), (-1, "unknown"), (2**64 - 1, "unknown"),
    (str(2**64 - 1), "unknown"), (True, "invalid"), (3.0, "invalid"),
    (float(2**63), "invalid"), (2**64, "invalid"), ("01", "invalid"),
    ("+1", "invalid"), (" 1", "invalid"), ("1e3", "invalid"),
    ("１２", "invalid"), ("9" * 5000, "invalid"),
])
def test_observation_accepts_decimal_text_without_lossy_coercion(value, quality):
    observation = build_video_observation(
        {"object_id": value}, namespace="video_detection", source_record_id="event-1",
        device_id="cam1", received_at="2026-10-12T13:00:00+09:00",
    )
    assert observation["track_id"]["quality"] == quality
    assert observation["track_id"]["value"] == (value if quality == "valid" else None)


def test_existing_event_and_observation_identity_survive_wire_conversion():
    payload = {
        "event_id": "existing-event", "camera_id": "cam1", "type": "person",
        "timestamp": 1700000000, "object_id": 12123278610165399591,
        "observation": {"observation_id": "existing-observation",
                        "source_record_id": "existing-event",
                        "track_id": {"value": 12123278610165399591,
                                     "quality": "valid", "source_field": "object_id"}},
        "metadata": {"matches": [{"person_object_id": 12123278610165399592}]},
    }
    original = deepcopy(payload)
    normalized = canonicalize_event_payload(payload)
    assert payload == original
    assert normalized["event_id"] == payload["event_id"]
    assert normalized["observation"]["observation_id"] == "existing-observation"
    assert normalized["observation"]["track_id"]["value"] == "12123278610165399591"
    assert normalized["metadata"]["matches"][0]["person_object_id"] == "12123278610165399592"


def test_old_mqtt_outbox_replay_serializes_ids_without_rewriting_identity(tmp_path):
    with patch.object(MqttEventPublisher, "_start_outbox_replay_worker"):
        publisher = MqttEventPublisher(outbox_db_path=str(tmp_path / "outbox.db"))
    publisher._client = MagicMock()
    publisher._client.publish.return_value.rc = 0
    publisher._connected = True
    legacy = {"event_id": "old-event", "object_id": 12123278610165399591,
              "raw": {"object_id": 12123278610165399591},
              "observation": {"track_id": {"value": 12123278610165399591,
                                           "quality": "valid"}}}
    try:
        publisher._outbox.save_pending("cctv/ai/events/cam1/person", legacy)
        assert publisher.replay_pending_once() == 1
        wire = json.loads(publisher._client.publish.call_args.args[1])
        assert wire["object_id"] == "12123278610165399591"
        assert wire["raw"]["object_id"] == "12123278610165399591"
        assert wire["observation"]["track_id"]["value"] == "12123278610165399591"
        assert wire["event_id"] == "old-event"
        assert publisher._outbox.get_pending() == []
    finally:
        publisher.disconnect()


def test_edgex_detection_uses_same_track_id_contract():
    builder = _PayloadMixin()
    builder.PROFILE_NAME = "cctv"
    builder.service_name = "test"
    payload = builder._build_value_payload(
        "person", 0.9, 1, 2, 3, 4, 12123278610165399591,
        "2026-10-12T13:00:00+09:00", "cam1", "person_detection",
    )
    assert payload["object_id"] == "12123278610165399591"


def test_kuiper_stream_declares_track_id_as_string():
    path = Path(__file__).resolve().parents[1] / "kuiper/rules/cctv_intrusion_rules.json"
    sql = json.loads(path.read_text())["stream"]["sql"]
    assert "object_id STRING" in sql


@pytest.mark.parametrize("quality,expected", [("valid", "missing"), ("unknown", "unknown"),
                                             ("invalid", "invalid"), ("missing", "missing")])
def test_null_observation_keeps_unknown_reason_but_cannot_be_valid(quality, expected):
    payload = canonicalize_event_payload({
        "camera_id": "cam1", "type": "person", "timestamp": 1700000000,
        "observation": {"track_id": {"value": None, "quality": quality}},
    })
    assert payload["observation"]["track_id"] == {"value": None, "quality": expected}


@pytest.mark.parametrize("track_id,event_id", [(0, "evt_b97ea6dfcfa21ef1"),
                                              (12123278610165399591, "evt_0cc11b2572b1c4d2")])
def test_generated_event_id_matches_pre_contract_version(track_id, event_id):
    # 수정 전 생성값을 고정해 문자열 전환이 저장·중복 제거 키를 바꾸지 않게 한다.
    payload = canonicalize_event_payload({"camera_id": "cam1", "type": "person",
                                          "timestamp": 1700000000, "object_id": track_id})
    assert payload["event_id"] == event_id
