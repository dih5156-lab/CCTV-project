from dataclasses import dataclass, field

from src.services.command_event_handler import CommandEventHandler


@dataclass
class FakeRepository:
    records: list = field(default_factory=list)

    def record_command(self, *args):
        self.records.append(args)


def test_handler_changes_mode_and_records_success():
    repository = FakeRepository()
    published = []
    modes = []
    handler = CommandEventHandler(
        repository=repository,
        publish_status=lambda topic, payload: published.append((topic, payload)),
        set_mode=lambda mode, site_id=None: modes.append((mode, site_id)),
        approve_event=lambda event_id: (True, "승인 완료"),
        reject_event=lambda event_id: (True, "거부 완료"),
        control_mode_factory=lambda value: value,
        mode_topic="commands/mode",
        approve_topic="commands/approve",
        reject_topic="commands/reject",
    )

    handler.dispatch(
        "commands/mode",
        {"command_id": "command-1", "mode": "manual", "site_id": "site-1"},
    )

    assert modes == [("manual", "site-1")]
    assert published[-1][0] == "commands/result"
    assert published[-1][1]["status"] == "success"
    assert repository.records[-1][0] == "command-1"
    assert repository.records[-1][2] == "success"


def test_handler_records_error_for_invalid_mode():
    repository = FakeRepository()
    published = []

    def invalid_mode(_value):
        raise ValueError("잘못된 모드")

    handler = CommandEventHandler(
        repository=repository,
        publish_status=lambda topic, payload: published.append((topic, payload)),
        set_mode=lambda mode, site_id=None: None,
        approve_event=lambda event_id: (True, "승인 완료"),
        reject_event=lambda event_id: (True, "거부 완료"),
        control_mode_factory=invalid_mode,
        mode_topic="commands/mode",
        approve_topic="commands/approve",
        reject_topic="commands/reject",
    )

    handler.dispatch("commands/mode", {"command_id": "command-2", "mode": "bad"})

    assert published[-1][1]["status"] == "error"
    assert repository.records[-1][2] == "error"
