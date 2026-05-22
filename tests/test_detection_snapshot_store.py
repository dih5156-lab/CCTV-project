"""DetectionSnapshotStore 단위 테스트."""

from src.core.detection_snapshot_store import DetectionSnapshotStore
from src.core.events import DetectionEvent, EventType


def test_record_and_snapshot_returns_latest_detections():
    store = DetectionSnapshotStore()
    event = DetectionEvent(
        event_type=EventType.PERSON,
        x=1,
        y=2,
        width=3,
        height=4,
        confidence=0.9,
        timestamp=123.0,
    )

    store.record("cam01", [event])
    snapshot = store.snapshot()

    assert "cam01" in snapshot
    assert snapshot["cam01"]["detections"][0]["type"] == "person"
    assert snapshot["cam01"]["timestamp"] > 0
