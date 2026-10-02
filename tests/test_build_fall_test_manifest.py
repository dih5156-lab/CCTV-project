import pytest

from scripts.datasets.build_fall_test_manifest import (
    rewrite_video_paths,
    select_test_rows,
    split_rows_by_scene_group,
)


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


def test_select_test_rows_can_keep_multiple_cameras_per_scene_group():
    rows = [
        {"scene_id": "f1_C1", "scene_group": "f1", "scene_category": "전면낙상", "is_fall": True},
        {"scene_id": "f1_C2", "scene_group": "f1", "scene_category": "전면낙상", "is_fall": True},
        {"scene_id": "s1_C1", "scene_group": "s1", "scene_category": "측면낙상", "is_fall": True},
        {"scene_id": "s1_C2", "scene_group": "s1", "scene_category": "측면낙상", "is_fall": True},
        {"scene_id": "b1_C1", "scene_group": "b1", "scene_category": "후면낙상", "is_fall": True},
        {"scene_id": "b1_C2", "scene_group": "b1", "scene_category": "후면낙상", "is_fall": True},
        {"scene_id": "n1_C1", "scene_group": "n1", "is_fall": False},
        {"scene_id": "n1_C2", "scene_group": "n1", "is_fall": False},
    ]

    selected = select_test_rows(rows, per_group=1, cameras_per_group=2)

    assert [row["scene_id"] for row in selected] == [
        "f1_C1",
        "f1_C2",
        "s1_C1",
        "s1_C2",
        "b1_C1",
        "b1_C2",
        "n1_C1",
        "n1_C2",
    ]


def test_split_rows_by_scene_group_keeps_each_group_in_one_split():
    rows = [
        {"scene_id": "f1_C1", "scene_group": "f1", "label": "fall", "is_fall": True},
        {"scene_id": "f1_C2", "scene_group": "f1", "label": "fall", "is_fall": True},
        {"scene_id": "f2_C1", "scene_group": "f2", "label": "fall", "is_fall": True},
        {"scene_id": "f3_C1", "scene_group": "f3", "label": "fall", "is_fall": True},
        {"scene_id": "n1_C1", "scene_group": "n1", "label": "not_fall", "is_fall": False},
        {"scene_id": "n2_C1", "scene_group": "n2", "label": "not_fall", "is_fall": False},
        {"scene_id": "n3_C1", "scene_group": "n3", "label": "not_fall", "is_fall": False},
    ]

    splits = split_rows_by_scene_group(rows, validation_ratio=0.25, test_ratio=0.25)

    group_splits = {}
    for split_name, split_rows in splits.items():
        for row in split_rows:
            group_splits.setdefault(row["scene_group"], set()).add(split_name)
    assert set(group_splits) == {"f1", "f2", "f3", "n1", "n2", "n3"}
    assert all(len(split_names) == 1 for split_names in group_splits.values())
    assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == len(rows)


def test_split_rows_by_scene_group_rejects_mixed_labels():
    rows = [
        {"scene_id": "same_C1", "scene_group": "same", "label": "fall", "is_fall": True},
        {"scene_id": "same_C2", "scene_group": "same", "label": "not_fall", "is_fall": False},
    ]

    with pytest.raises(ValueError, match="혼합된 라벨"):
        split_rows_by_scene_group(rows)


def test_split_rows_by_scene_group_rejects_invalid_ratios():
    with pytest.raises(ValueError, match="비율"):
        split_rows_by_scene_group([], validation_ratio=0.6, test_ratio=0.5)


def test_rewrite_video_paths_maps_only_the_configured_source_root():
    rows = [
        {"video_path": "/host/dataset/scene.mp4"},
        {"video_path": "/other/scene.mp4"},
        {"scene_id": "without_video"},
    ]

    rewritten = rewrite_video_paths(
        rows,
        source_root="/host/dataset",
        target_root="/app/dataset",
    )

    assert rewritten[0]["video_path"] == "/app/dataset/scene.mp4"
    assert rewritten[1]["video_path"] == "/other/scene.mp4"
    assert "video_path" not in rewritten[2]
