"""MQTT event outbox 단위 테스트."""

import sqlite3

from src.protocols.mqtt_outbox import MqttEventOutbox


def test_save_pending_deduplicates_by_event_id_and_destination(tmp_path):
    outbox = MqttEventOutbox(tmp_path / "mqtt_outbox.db", destination_name="cctv/ai/events")
    payload = {"event_id": "evt-1", "camera_id": "cam01", "type": "head"}

    first_id = outbox.save_pending("cctv/ai/events/cam01/head", payload)
    second_id = outbox.save_pending("cctv/ai/events/cam01/head", payload)

    assert first_id == second_id
    assert outbox.pending_count() == 1


def test_mark_sent_removes_from_pending(tmp_path):
    outbox = MqttEventOutbox(tmp_path / "mqtt_outbox.db")
    row_id = outbox.save_pending(
        "cctv/ai/events/cam01/head",
        {"event_id": "evt-2", "camera_id": "cam01", "type": "head"},
    )

    outbox.mark_sent(row_id)

    assert outbox.pending_count() == 0
    assert outbox.get_pending() == []


def test_mark_retry_failed_increments_retry_count(tmp_path):
    db_path = tmp_path / "mqtt_outbox.db"
    outbox = MqttEventOutbox(db_path)
    row_id = outbox.save_pending(
        "cctv/ai/events/cam01/head",
        {"event_id": "evt-3", "camera_id": "cam01", "type": "head"},
    )

    outbox.mark_retry_failed(row_id, "broker unavailable")

    pending = outbox.get_pending()
    assert pending[0]["retry_count"] == 1
    with sqlite3.connect(db_path) as conn:
        error = conn.execute(
            "SELECT last_error FROM mqtt_event_outbox WHERE id = ?",
            (row_id,),
        ).fetchone()[0]
    assert error == "broker unavailable"
