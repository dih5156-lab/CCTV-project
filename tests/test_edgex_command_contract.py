from src.edgex.command_contract import build_command_request, build_command_topic


def test_command_topic_contains_jetson_and_device_path():
    topic = build_command_topic("edgex/commands/cctv", "jetson-01", "speaker")

    assert topic == "edgex/commands/cctv/jetson-01/speaker"


def test_command_topic_can_include_physical_device_id():
    topic = build_command_topic(
        "edgex/commands/cctv", "jetson-01", "speaker", device_id="cctv-speaker-02"
    )

    assert topic == "edgex/commands/cctv/jetson-01/speaker/cctv-speaker-02"


def test_command_request_has_traceable_event_and_request_ids():
    request = build_command_request(
        event_id="event-123",
        device="speaker",
        action="play",
        payload={"text": "위험 상황입니다."},
        request_id="request-456",
        issued_at="2026-09-03T10:00:00+09:00",
    )

    assert request == {
        "version": "1",
        "request_id": "request-456",
        "event_id": "event-123",
        "source": "cctv-action-layer",
        "device": "speaker",
        "action": "play",
        "issued_at": "2026-09-03T10:00:00+09:00",
        "payload": {"text": "위험 상황입니다."},
    }


def test_command_request_can_target_a_physical_device():
    request = build_command_request(
        event_id="event-456",
        device="speaker",
        device_id="cctv-speaker-02",
        action="play",
        payload={"text": "두 번째 장치"},
    )

    assert request["device_id"] == "cctv-speaker-02"
