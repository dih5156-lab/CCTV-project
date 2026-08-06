from scripts.run_fall_training_pipeline import split_manifest


def test_split_manifest_keeps_scene_groups_together_and_classes_present():
    rows = [
        {"scene_id": "a_C1", "scene_group": "a", "label": "fall"},
        {"scene_id": "a_C2", "scene_group": "a", "label": "fall"},
        {"scene_id": "b_C1", "scene_group": "b", "label": "not_fall"},
        {"scene_id": "b_C2", "scene_group": "b", "label": "not_fall"},
        {"scene_id": "c_C1", "scene_group": "c", "label": "fall"},
        {"scene_id": "d_C1", "scene_group": "d", "label": "not_fall"},
    ]
    train, validation = split_manifest(rows, validation_ratio=0.5)
    train_groups = {row["scene_group"] for row in train}
    validation_groups = {row["scene_group"] for row in validation}
    assert train_groups.isdisjoint(validation_groups)
    assert {row["label"] for row in train} == {"fall", "not_fall"}
    assert {row["label"] for row in validation} == {"fall", "not_fall"}
