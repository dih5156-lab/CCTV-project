from dataclasses import dataclass, field

from src.services.edgex_result_handler import EdgeXResultHandler


@dataclass
class FakeRepository:
    records: list = field(default_factory=list)

    def record_command(self, *args):
        self.records.append(args)


def test_handler_records_and_publishes_edgex_result():
    repository = FakeRepository()
    published = []
    handler = EdgeXResultHandler(
        repository=repository,
        publish_status=lambda topic, payload: published.append((topic, payload)),
    )

    handler.handle(
        "edgex/results/cctv/signboard",
        {
            "request_id": "event-1:signboard:signboard-1",
            "event_id": "event-1",
            "device_id": "signboard-1",
            "status": "acknowledged",
        },
    )

    assert repository.records[0][0] == "event-1:signboard"
    assert repository.records[0][2] == "acknowledged"
    assert published[0][0] == "devices/result"
    assert published[0][1]["status"] == "acknowledged"
    assert published[0][1]["command_id"] == "event-1:signboard"


def test_handler_ignores_result_without_request_id():
    repository = FakeRepository()
    published = []
    handler = EdgeXResultHandler(
        repository=repository,
        publish_status=lambda topic, payload: published.append((topic, payload)),
    )

    handler.handle("edgex/results/cctv/signboard", {"status": "failed"})

    assert repository.records == []
    assert published == []
