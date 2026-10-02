from __future__ import annotations

import json

from scripts.health import check_model_layout


def test_model_layout_includes_crop_pose_profile(capsys) -> None:
    assert check_model_layout.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["passed"] is True
    checks = payload["crop_pose_checks"]
    assert len(checks) == 4
    assert all(item["config_exists"] and item["model_exists"] for item in checks)
