from datetime import datetime, timedelta, timezone

from src.aiot.contracts import AiQueryRequest
from src.aiot.query_service import AiQueryService, RecentAppearanceLiveProvider


def _request(mode="history", filters=None):
    return AiQueryRequest(
        request_id="q-1",
        jetson_id="edge-01",
        camera_ids=("camera-1",),
        search_mode=mode,
        filters=filters or {},
        query_text=None,
        time_from=None,
        time_to=None,
        limit=20,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


ROW = {
    "event_id": "event-1",
    "timestamp": 100.0,
    "camera_id": "camera-1",
    "upper_color": "red",
    "lower_color": "black",
    "has_handbag": True,
    "has_backpack": False,
    "has_suitcase": False,
    "has_helmet": False,
    "gender": "female",
    "crop_path": "data/appearance_crops/event-1.jpg",
    "attribute_metadata": {"confidence": 0.91},
}


class FakeAppearanceLog:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.rows)


class FakeLiveProvider:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def search(self, filters, camera_ids, limit):
        self.calls += 1
        return list(self.rows)


def test_history_maps_handbag_and_gender_filters():
    log = FakeAppearanceLog([ROW])
    service = AiQueryService(log, FakeLiveProvider([]))

    matches = service.search(
        _request(filters={"gender": "female", "has_handbag": True})
    )

    assert log.calls[0]["gender"] == "female"
    assert log.calls[0]["has_handbag"] is True
    assert matches[0]["attributes"]["gender"] == "female"
    assert "crop_path" not in matches[0]


def test_both_deduplicates_same_event():
    service = AiQueryService(FakeAppearanceLog([ROW]), FakeLiveProvider([ROW]))
    matches = service.search(_request(mode="both"))
    assert len(matches) == 1
    assert matches[0]["match_id"] == "event-1"


def test_live_mode_does_not_query_history():
    log = FakeAppearanceLog([ROW])
    live = FakeLiveProvider([ROW])
    matches = AiQueryService(log, live).search(_request(mode="live"))
    assert len(matches) == 1
    assert not log.calls
    assert live.calls == 1


def test_query_service_resolves_media_without_exposing_path():
    service = AiQueryService(FakeAppearanceLog([ROW]), FakeLiveProvider([]))
    match = service.search(_request())[0]
    assert "crop_path" not in match
    assert str(service.resolve_media("event-1")).endswith("event-1.jpg")


def test_recent_live_provider_limits_search_to_recent_window():
    log = FakeAppearanceLog([ROW])
    provider = RecentAppearanceLiveProvider(log, window_seconds=30, now=lambda: 100.0)
    provider.search({"gender": "female"}, ("camera-1",), 5)
    assert log.calls[0]["time_from"] == 70.0
    assert log.calls[0]["time_to"] == 100.0
