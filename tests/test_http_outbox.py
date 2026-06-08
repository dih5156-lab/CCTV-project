"""HTTP event outbox 단위 테스트."""

from src.protocols.http_outbox import HttpEventOutbox


def test_save_pending_deduplicates_by_event_and_target(tmp_path):
    outbox = HttpEventOutbox(tmp_path / "http_outbox.db")
    body = {
        "topic": "cctv/ai/events/cam01/head",
        "event": {"event_id": "evt-http-1", "camera_id": "cam01", "type": "head"},
    }

    first_id = outbox.save_pending("alert-api", "http://example.com/alerts", body)
    second_id = outbox.save_pending("alert-api", "http://example.com/alerts", body)

    assert first_id == second_id
    assert outbox.pending_count() == 1


def test_mark_sent_removes_http_outbox_pending(tmp_path):
    outbox = HttpEventOutbox(tmp_path / "http_outbox.db")
    row_id = outbox.save_pending(
        "alert-api",
        "http://example.com/alerts",
        {
            "topic": "topic",
            "event": {"event_id": "evt-http-2", "camera_id": "cam01", "type": "head"},
        },
    )

    outbox.mark_sent(row_id)

    assert outbox.pending_count() == 0
    assert outbox.get_pending() == []
