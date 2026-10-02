from dataclasses import dataclass, field

from src.services.command_result_collector import CommandResultCollector
from src.services.device_command_transport import DeviceCommand, DeviceCommandResult


@dataclass
class FakeRepository:
    records: list = field(default_factory=list)

    def record_command(self, *args):
        self.records.append(args)


class FakeMetric:
    def __init__(self):
        self.labels_called = []

    def labels(self, **labels):
        self.labels_called.append(labels)
        return self

    def inc(self):
        return None


def test_collector_records_acknowledged_device_result():
    repository = FakeRepository()
    command_metric = FakeMetric()
    result_metric = FakeMetric()
    collector = CommandResultCollector(
        repository=repository,
        device_commands_metric=command_metric,
        device_results_metric=result_metric,
    )
    command = DeviceCommand(
        device="signboard",
        action="display",
        payload={"text": "주의"},
        command_id="event-1:signboard",
        event_id="event-1",
        camera_id="camera-1",
    )
    result = DeviceCommandResult(
        device="signboard",
        command_id=command.command_id,
        ok=True,
        status="acknowledged",
    )

    collected = collector.collect(command, result)

    assert collected == {
        "device": "signboard",
        "command_id": "event-1:signboard",
        "status": "acknowledged",
        "event_id": "event-1",
        "device_ids": [],
        "request_ids": [],
    }
    assert repository.records == [
        (
            "event-1:signboard",
            "device/signboard",
            "acknowledged",
            {"text": "주의"},
        )
    ]
    assert command_metric.labels_called == [{"device": "signboard", "status": "ok"}]
    assert result_metric.labels_called == [
        {"device": "signboard", "status": "acknowledged"}
    ]


def test_collector_includes_error_when_device_command_fails():
    collector = CommandResultCollector(
        repository=FakeRepository(),
        device_commands_metric=FakeMetric(),
        device_results_metric=FakeMetric(),
    )
    command = DeviceCommand(
        device="speaker",
        action="play",
        payload={"text": "경고"},
        command_id="event-2:speaker",
        event_id="event-2",
        camera_id="camera-1",
    )
    result = DeviceCommandResult(
        device="speaker",
        command_id=command.command_id,
        ok=False,
        status="failed",
        error="연결 실패",
    )

    assert collector.collect(command, result)["error"] == "연결 실패"


def test_collector_preserves_edgex_physical_trace_ids():
    collector = CommandResultCollector(
        repository=FakeRepository(),
        device_commands_metric=FakeMetric(),
        device_results_metric=FakeMetric(),
    )
    command = DeviceCommand(
        device="signboard",
        action="display",
        payload={},
        command_id="event-3:signboard",
        event_id="event-3",
        camera_id="camera-1",
    )
    result = DeviceCommandResult(
        device="signboard",
        command_id=command.command_id,
        ok=True,
        status="acknowledged",
        device_ids=("signboard-1", "signboard-2"),
        request_ids=("event-3:signboard:signboard-1", "event-3:signboard:signboard-2"),
    )

    collected = collector.collect(command, result)

    assert collected["event_id"] == "event-3"
    assert collected["device_ids"] == ["signboard-1", "signboard-2"]
    assert collected["request_ids"] == [
        "event-3:signboard:signboard-1",
        "event-3:signboard:signboard-2",
    ]
