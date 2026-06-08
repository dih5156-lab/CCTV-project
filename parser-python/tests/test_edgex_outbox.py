"""
tests/test_edgex_outbox.py
===========================
EdgeXOutbox 와 EdgeXForwarder 아웃박스 통합 단위 테스트.
"""

import time

import pytest
from database.edgex_outbox import EdgeXOutbox

# ────────────────────────────────────────────
# Fixture: 임시 SQLite 파일
# ────────────────────────────────────────────

@pytest.fixture
def outbox(tmp_path):
    db_path = str(tmp_path / "test_outbox.db")
    box = EdgeXOutbox(db_path)
    yield box
    box.close()


# ────────────────────────────────────────────
# 기본 저장 / 상태 전환 테스트
# ────────────────────────────────────────────

def test_save_pending_and_get(outbox):
    row_id = outbox.save_pending(
        device_id="dev001",
        table_name="t34950",
        core_data_url="http://localhost:59880/api/v3/event/device-rest/aiot-t34950-river/aiot-dev001/t34950",
        edgex_event={"apiVersion": "v3", "event": {"id": "abc"}},
    )
    assert row_id > 0

    rows = outbox.get_pending()
    assert len(rows) == 1
    assert rows[0]["id"] == row_id
    assert rows[0]["device_id"] == "dev001"
    assert rows[0]["table_name"] == "t34950"
    assert rows[0]["retry_count"] == 0


def test_mark_sent(outbox):
    row_id = outbox.save_pending(
        device_id="dev002",
        table_name="t34955",
        core_data_url="http://localhost:59880/api/v3/event/device-rest/aiot-t34955/aiot-dev002/t34955",
        edgex_event={"apiVersion": "v3"},
    )
    outbox.mark_sent(row_id)

    rows = outbox.get_pending()
    assert rows == []  # sent 이므로 목록에 없어야 함


def test_increment_retry(outbox):
    row_id = outbox.save_pending(
        device_id="dev003",
        table_name="t34957",
        core_data_url="http://localhost:59880/",
        edgex_event={"apiVersion": "v3"},
    )
    outbox.increment_retry(row_id)
    outbox.increment_retry(row_id)

    rows = outbox.get_pending()
    assert rows[0]["retry_count"] == 2


def test_pending_count(outbox):
    outbox.save_pending("d1", "t34950", "http://x", {"a": 1})
    outbox.save_pending("d2", "t34950", "http://x", {"b": 2})
    assert outbox.pending_count() == 2


def test_expire_old_failed(outbox):
    """최대 재시도 초과 항목은 expire_old_failed 후 pending 목록에서 제거."""
    import database.edgex_outbox as module

    original_max = module._MAX_RETRY
    module._MAX_RETRY = 0  # 즉시 만료 조건

    try:
        row_id = outbox.save_pending("d1", "t34950", "http://x", {"a": 1})
        outbox.increment_retry(row_id)  # retry_count=1 >= _MAX_RETRY=0+1? No, 0 is max
        expired = outbox.expire_old_failed()
        # retry_count(1) >= _MAX_RETRY(0) → expired
        assert expired >= 1
        assert outbox.get_pending() == []
    finally:
        module._MAX_RETRY = original_max


def test_get_pending_excludes_expired_rows(outbox):
    """TTL이 지난 행은 expire_old_failed 실행 전이라도 재전송 대상에서 제외한다."""
    row_id = outbox.save_pending("d1", "t34950", "http://x", {"a": 1})
    expired_at_ms = int((time.time() - 1) * 1000)

    with outbox._lock:
        outbox._conn.execute(
            "UPDATE event_outbox SET expire_at_ms=? WHERE id=?",
            (expired_at_ms, row_id),
        )
        outbox._conn.commit()

    assert outbox.get_pending() == []


def test_multiple_pending_ordered(outbox):
    """pending 항목은 생성 순(오래된 것부터)으로 반환되어야 한다."""
    for i in range(5):
        outbox.save_pending(f"dev{i}", "t34950", "http://x", {"i": i})
        time.sleep(0.01)

    rows = outbox.get_pending(limit=5)
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids)  # 오름차순
