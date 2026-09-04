from types import SimpleNamespace

from src.services.device_command_transport import (
    DeviceCommand,
    DirectDeviceCommandTransport,
    EdgeXCommandTransport,
)


def test_direct_transport_executes_speaker_command_and_returns_result():
    calls = []

    speaker = SimpleNamespace(
        play=lambda *args, **kwargs: calls.append((args, kwargs)) or True
    )
    transport = DirectDeviceCommandTransport(
        speaker=speaker,
        signboard=SimpleNamespace(),
        siren=SimpleNamespace(),
    )

    result = transport.send(
        DeviceCommand(
            device="speaker",
            action="play",
            payload={"event_type": "fall_detected", "severity": "critical", "text": "낙상"},
            command_id="event-1:speaker",
            event_id="event-1",
            camera_id="camera-1",
        )
    )

    assert result.ok is True
    assert result.status == "acknowledged"
    assert calls == [
        (("fall_detected", "critical", "camera-1"), {"text": "낙상"})
    ]


def test_direct_transport_executes_signboard_and_siren_commands():
    calls = []
    signboard = SimpleNamespace(
        display=lambda **kwargs: calls.append(("signboard", kwargs)) or True
    )
    siren = SimpleNamespace(
        trigger=lambda *args: calls.append(("siren", args)) or False
    )
    transport = DirectDeviceCommandTransport(
        speaker=SimpleNamespace(),
        signboard=signboard,
        siren=siren,
    )

    signboard_result = transport.send(
        DeviceCommand(
            device="signboard",
            action="display",
            payload={"text": "위험", "title": "경고!", "class_name": "fall_detected"},
            command_id="event-1:signboard",
            event_id="event-1",
            camera_id="camera-1",
        )
    )
    siren_result = transport.send(
        DeviceCommand(
            device="siren",
            action="trigger",
            payload={"event_type": "fall_detected"},
            command_id="event-1:siren",
            event_id="event-1",
            camera_id="camera-1",
        )
    )

    assert signboard_result.status == "acknowledged"
    assert siren_result.status == "failed"
    assert calls == [
        (
            "signboard",
            {"text": "위험", "title": "경고!", "class_name": "fall_detected"},
        ),
        ("siren", ("fall_detected", "camera-1")),
    ]


def test_direct_transport_rejects_unknown_device_command():
    transport = DirectDeviceCommandTransport(
        speaker=SimpleNamespace(),
        signboard=SimpleNamespace(),
        siren=SimpleNamespace(),
    )

    result = transport.send(
        DeviceCommand(
            device="unknown",
            action="run",
            payload={},
            command_id="event-1:unknown",
            event_id="event-1",
            camera_id="camera-1",
        )
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.error == "지원하지 않는 장치: unknown"


def test_edgex_transport_publishes_one_command_per_registered_device():
    published = []
    transport = EdgeXCommandTransport(
        publish=lambda topic, payload: published.append((topic, payload)) or True,
        resolve_device_ids=lambda device, camera_id: ["speaker-1", "speaker-2"],
        jetson_id="jetson-01",
        topic_prefix="edgex/commands/cctv",
    )

    result = transport.send(
        DeviceCommand(
            device="speaker",
            action="play",
            payload={"event_type": "fall_detected", "text": "낙상"},
            command_id="event-1:speaker",
            event_id="event-1",
            camera_id="camera-1",
        )
    )

    assert result.ok is True
    assert result.status == "acknowledged"
    assert [topic for topic, _ in published] == [
        "edgex/commands/cctv/jetson-01/speaker/speaker-1",
        "edgex/commands/cctv/jetson-01/speaker/speaker-2",
    ]
    assert [payload["request_id"] for _, payload in published] == [
        "event-1:speaker:speaker-1",
        "event-1:speaker:speaker-2",
    ]


def test_edgex_transport_returns_failed_when_one_device_publish_fails():
    published = []
    transport = EdgeXCommandTransport(
        publish=lambda topic, payload: published.append(topic) or topic.endswith("speaker-1"),
        resolve_device_ids=lambda device, camera_id: ["speaker-1", "speaker-2"],
        jetson_id="jetson-01",
        topic_prefix="edgex/commands/cctv",
    )

    result = transport.send(
        DeviceCommand(
            device="speaker",
            action="play",
            payload={"event_type": "fall_detected"},
            command_id="event-1:speaker",
            event_id="event-1",
            camera_id="camera-1",
        )
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.error == "EdgeX 명령 발행 실패: speaker-2"
    assert len(published) == 2
