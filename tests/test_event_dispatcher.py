"""EventDispatcher 단위 테스트."""

from queue import Queue
from unittest.mock import MagicMock

from src.core.event_dispatcher import EventDispatcher


def test_enqueue_puts_event_and_increments_detected():
    event_queue = Queue(maxsize=1)
    backup_store = MagicMock()
    increment_stat = MagicMock()
    dispatcher = EventDispatcher(event_queue, backup_store, increment_stat)

    ok = dispatcher.enqueue("cam01", {"type": "person", "camera_id": "cam01"})

    assert ok is True
    assert event_queue.get_nowait()["type"] == "person"
    increment_stat.assert_called_once_with("events_detected")
    backup_store.save_locally.assert_not_called()


def test_enqueue_saves_locally_when_queue_is_full():
    event_queue = Queue(maxsize=1)
    event_queue.put_nowait({"type": "existing"})
    backup_store = MagicMock()
    increment_stat = MagicMock()
    dispatcher = EventDispatcher(event_queue, backup_store, increment_stat)
    dropped_event = {"type": "head", "camera_id": "cam01"}

    ok = dispatcher.enqueue("cam01", dropped_event)

    assert ok is False
    increment_stat.assert_called_once_with("events_dropped")
    backup_store.save_locally.assert_called_once_with(dropped_event)
