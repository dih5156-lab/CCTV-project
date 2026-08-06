from scripts.datasets.train_fall_direction_rf import normalize_direction


def test_normalize_direction_uses_scene_category():
    assert normalize_direction({"scene_category": "후면낙상", "fall_type": "중심을 잃고 넘어짐"}) == "back"
    assert normalize_direction({"scene_category": "측면낙상"}) == "side"
    assert normalize_direction({"scene_category": "전면낙상"}) == "front"


def test_normalize_direction_falls_back_to_other():
    assert normalize_direction({"scene_category": "낙상", "fall_type": "미상"}) == "other"
