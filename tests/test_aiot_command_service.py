from datetime import datetime, timedelta, timezone

from prometheus_client import CollectorRegistry, generate_latest

from src.aiot.command_store import CommandStore
from src.aiot.metrics import AiotMetrics
from src.edgex._outbox_mixin import _OutboxMixin
from src.services.aiot_command_service import AiotCommandService


def _query_payload():
    return {
        "schema_version": "1.0",
        "message_type": "ai_query_request",
        "request_id": "q-1",
        "target": {"jetson_id": "edge-01", "camera_ids": ["camera-1"]},
        "search_mode": "history",
        "filters": {"gender": "female", "has_handbag": True},
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(minutes=5)
        ).isoformat(),
    }


class FakeQueryService:
    def __init__(self):
        self.calls = 0

    def search(self, request):
        self.calls += 1
        return [{"match_id": "event-1", "camera_id": "camera-1"}]


class FakePublisher:
    def __init__(self, succeeds=True):
        self.succeeds = succeeds
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return self.succeeds


class FakeOutbox:
    def __init__(self):
        self.items = []

    def store_result(self, request_id, payload, last_error):
        self.items.append((request_id, payload, last_error))


def _service(tmp_path, publisher=None, metrics=None):
    query = FakeQueryService()
    output = publisher or FakePublisher()
    outbox = FakeOutbox()
    service = AiotCommandService(
        command_store=CommandStore(tmp_path / "commands.db"),
        query_service=query,
        media_uploader=None,
        resolve_match=lambda _: None,
        publish_result=output,
        result_outbox=outbox,
        max_results=20,
        metrics=metrics,
    )
    return service, query, output, outbox


def test_query_publishes_accepted_running_completed(tmp_path):
    service, _, publisher, _ = _service(tmp_path)
    service.handle(_query_payload())
    assert [item["status"] for item in publisher.payloads] == [
        "accepted",
        "running",
        "completed",
    ]
    assert publisher.payloads[-1]["matches"][0]["match_id"] == "event-1"


def test_duplicate_republishes_saved_result_without_search(tmp_path):
    service, query, publisher, _ = _service(tmp_path)
    service.handle(_query_payload())
    service.handle(_query_payload())
    assert query.calls == 1
    assert publisher.payloads[-1]["status"] == "completed"


def test_failed_publish_is_written_to_outbox(tmp_path):
    service, _, _, outbox = _service(tmp_path, FakePublisher(succeeds=False))
    service.handle(_query_payload())
    assert outbox.items
    assert outbox.items[-1][0] == "q-1"


def test_edgex_outbox_adapter_builds_stable_result_event_id():
    class CapturingOutbox(_OutboxMixin):
        def __init__(self):
            self.saved = []

        def _store_failed_detection_event(self, camera_id, event_data, last_error):
            self.saved.append((camera_id, event_data, last_error))

    outbox = CapturingOutbox()
    payload = build_result_payload("q-1", "completed")
    outbox.store_result("q-1", payload, "offline")
    assert outbox.saved[0][1]["event_id"] == "aiot:q-1:completed"
    assert outbox.saved[0][1]["type"] == "aiot_command_result"


def build_result_payload(request_id, status):
    return {
        "schema_version": "1.0",
        "message_type": "ai_command_result",
        "request_id": request_id,
        "status": status,
    }


def test_query_records_bounded_metrics(tmp_path):
    registry = CollectorRegistry()
    metrics = AiotMetrics(registry)
    service, _, _, _ = _service(tmp_path, metrics=metrics)
    service.handle(_query_payload())
    output = generate_latest(registry).decode()
    assert 'message_type="ai_query_request"' in output
    assert 'search_mode="history"' in output
