from src.services.action_command_components import CommandBuilder, DeviceResolver
from src.services.device_command_transport import DeviceCommandMode


def test_device_resolver_filters_disabled_and_unavailable_devices():
    available = {"signboard"}
    resolver = DeviceResolver(
        resolve_candidates=lambda camera_id: ["speaker", "signboard", "siren"],
        is_available=lambda device, force_refresh=False: device in available,
        command_mode=DeviceCommandMode.DIRECT,
    )

    assert resolver.resolve("camera-1") == ["signboard"]


def test_device_resolver_applies_edgex_allowed_device_filter():
    resolver = DeviceResolver(
        resolve_candidates=lambda camera_id: ["speaker", "signboard"],
        is_available=lambda device, force_refresh=False: True,
        command_mode=DeviceCommandMode.EDGEX,
        allowed_devices={"signboard"},
    )

    assert resolver.resolve("camera-1") == ["signboard"]


def test_command_builder_creates_common_command_without_mutating_payload():
    payload = {"text": "낙상 감지"}
    builder = CommandBuilder()

    command = builder.build(
        device="signboard",
        action="display",
        payload=payload,
        command_id="event-1:signboard",
        event_id="event-1",
        camera_id="camera-1",
    )

    assert command.device == "signboard"
    assert command.action == "display"
    assert command.payload == payload
    assert command.payload is not payload
    assert payload == {"text": "낙상 감지"}


def test_command_builder_creates_speaker_alarm_command():
    command = CommandBuilder().build_alarm_command(
        device="speaker",
        action="play",
        event_type="fall_detected",
        severity="critical",
        camera_id="camera-1",
        event_id="event-1",
        command_id="event-1:speaker",
        text="낙상이 감지되었습니다.",
    )

    assert command.action == "play"
    assert command.payload == {
        "event_type": "fall_detected",
        "severity": "critical",
        "camera_id": "camera-1",
        "text": "낙상이 감지되었습니다.",
    }


def test_command_builder_creates_signboard_alarm_command():
    command = CommandBuilder().build_alarm_command(
        device="signboard",
        action="display",
        event_type="fall_detected",
        severity="critical",
        camera_id="camera-1",
        event_id="event-1",
        command_id="event-1:signboard",
        text="낙상 감지",
    )

    assert command.payload == {
        "event_type": "fall_detected",
        "severity": "critical",
        "camera_id": "camera-1",
        "text": "낙상 감지",
        "title": "경고!",
        "class_name": "fall_detected",
    }


def test_command_builder_creates_siren_alarm_command():
    command = CommandBuilder().build_alarm_command(
        device="siren",
        action="trigger",
        event_type="fall_detected",
        severity="critical",
        camera_id="camera-1",
        event_id="event-1",
        command_id="event-1:siren",
    )

    assert command.payload == {
        "event_type": "fall_detected",
        "camera_id": "camera-1",
    }
