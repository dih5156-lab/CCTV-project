from __future__ import annotations

import json

from scripts.health import check_crop_pose_gie_registry


def test_crop_pose_gie_registry_is_consistent(capsys) -> None:
    assert check_crop_pose_gie_registry.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["passed"] is True
    assert [row["configured_gie_id"] for row in payload["checks"]] == ["1", "2", "3", "4"]
    assert [row["operate_on_gie_id"] for row in payload["checks"]] == [None, "1", "1", "1"]
