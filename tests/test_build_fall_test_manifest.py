from scripts.datasets.build_fall_test_manifest import select_test_rows


def test_select_test_rows_balances_directions_and_scene_groups():
    rows = [
        {"scene_id": "f1_C1", "scene_group": "f1", "scene_category": "전면낙상", "is_fall": True},
        {"scene_id": "s1_C1", "scene_group": "s1", "scene_category": "측면낙상", "is_fall": True},
        {"scene_id": "b1_C1", "scene_group": "b1", "scene_category": "후면낙상", "is_fall": True},
        {"scene_id": "n1_C1", "scene_group": "n1", "is_fall": False},
        {"scene_id": "f1_C2", "scene_group": "f1", "scene_category": "전면낙상", "is_fall": True},
    ]
    selected = select_test_rows(rows, per_group=5)
    assert len(selected) == 4
    assert {row["scene_group"] for row in selected} == {"f1", "s1", "b1", "n1"}
