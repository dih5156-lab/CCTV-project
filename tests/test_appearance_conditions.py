"""외형 조건 저장소 테스트."""

from __future__ import annotations

from src.services.appearance_conditions import AppearanceConditionStore


def test_appearance_condition_store_create_list_delete(tmp_path):
    store = AppearanceConditionStore(tmp_path / "appearance.db")
    payload = {
        "upper_color": "black",
        "lower_color": None,
        "has_helmet": True,
        "helmet_color": "yellow",
        "has_backpack": None,
        "has_handbag": None,
        "has_suitcase": None,
        "threshold": 0.8,
        "cameras": ["cam01"],
    }

    entry = store.create(
        condition_id="cond01",
        name="test-condition",
        payload=payload,
        enabled=True,
    )

    assert entry["id"] == "cond01"
    assert entry["name"] == "test-condition"
    assert entry["upper_color"] == "black"
    assert store.list_all() == [entry]
    assert store.delete("cond01") is True
    assert store.delete("cond01") is False
    assert store.list_all() == []


def test_appearance_condition_store_skips_invalid_payload(tmp_path):
    store = AppearanceConditionStore(tmp_path / "appearance.db")
    store.create(
        condition_id="valid",
        name="valid-condition",
        payload={"upper_color": "black", "threshold": 0.8},
        enabled=True,
    )
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO search_conditions (id, name, payload, enabled, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("broken", "broken-condition", "{not-json", 1, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()

    conditions = store.list_all()

    assert [condition["id"] for condition in conditions] == ["valid"]
