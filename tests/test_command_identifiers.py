import pytest

from src.services.command_identifiers import (
    build_device_command_id,
    build_transport_request_id,
    resolve_logical_command_id,
)


def test_build_device_command_id_uses_event_and_device():
    assert build_device_command_id("event-1", "signboard") == "event-1:signboard"


def test_build_transport_request_id_adds_physical_device_id():
    assert (
        build_transport_request_id("event-1:signboard", "signboard-1")
        == "event-1:signboard:signboard-1"
    )


def test_build_transport_request_id_keeps_logical_id_without_physical_id():
    assert build_transport_request_id("event-1:signboard", "") == "event-1:signboard"


def test_resolve_logical_command_id_removes_only_matching_device_suffix():
    assert (
        resolve_logical_command_id("event-1:signboard:signboard-1", "signboard-1")
        == "event-1:signboard"
    )


def test_resolve_logical_command_id_keeps_id_when_suffix_does_not_match():
    assert (
        resolve_logical_command_id("event-1:signboard", "other-1")
        == "event-1:signboard"
    )


@pytest.mark.parametrize(
    "builder,args",
    [
        (build_device_command_id, ("", "speaker")),
        (build_device_command_id, ("event-1", "")),
        (build_transport_request_id, ("", "device-1")),
    ],
)
def test_identifier_builders_reject_empty_required_values(builder, args):
    with pytest.raises(ValueError):
        builder(*args)
