import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/datasets/build_hard_case_manifest.py"
    spec = importlib.util.spec_from_file_location("build_hard_case_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_rows_filters_existing_cache_and_balances_labels(tmp_path):
    module = _load_module()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "existing_uniform_max30_stride6.json").write_text("{}")
    rows = [
        {"scene_id": "existing", "camera": 2, "is_fall": True, "video_path": str(tmp_path / "a.mp4")},
        {"scene_id": "fall", "camera": 2, "is_fall": True, "video_path": str(tmp_path / "b.mp4")},
        {"scene_id": "notfall", "camera": 3, "is_fall": False, "video_path": str(tmp_path / "c.mp4")},
        {"scene_id": "camera1", "camera": 1, "is_fall": True, "video_path": str(tmp_path / "d.mp4")},
    ]
    for row in rows:
        Path(row["video_path"]).touch()

    selected = module.select_rows(
        rows,
        cache=cache,
        max_fall=1,
        max_notfall=1,
        min_camera=2,
        max_frames=30,
        frame_stride=6,
        seed=1,
    )

    assert {row["scene_id"] for row in selected} == {"fall", "notfall"}
