from datetime import datetime, timedelta, timezone

from src.aiot.command_store import CommandStore

FUTURE = datetime.now(timezone.utc) + timedelta(minutes=5)


def test_claim_is_idempotent_across_reopen(tmp_path):
    path = tmp_path / "commands.db"
    first = CommandStore(path)
    assert first.claim("q-1", "ai_query_request", FUTURE).is_new
    first.close()

    reopened = CommandStore(path)
    assert not reopened.claim("q-1", "ai_query_request", FUTURE).is_new
    reopened.close()


def test_update_persists_completed_result(tmp_path):
    path = tmp_path / "commands.db"
    store = CommandStore(path)
    store.claim("q-1", "ai_query_request", FUTURE)
    store.update("q-1", "completed", {"matches": []})

    record = store.get("q-1")
    assert record is not None
    assert record.status == "completed"
    assert record.result_payload == {"matches": []}
    store.close()


def test_update_does_not_persist_upload_url(tmp_path):
    store = CommandStore(tmp_path / "commands.db")
    store.claim("m-1", "fetch_media_request", FUTURE)
    store.update(
        "m-1",
        "failed",
        {"upload_url": "https://server/upload?secret=token", "error": "expired"},
    )

    record = store.get("m-1")
    assert record is not None
    assert record.result_payload == {"error": "expired"}
    store.close()

