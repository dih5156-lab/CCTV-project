"""DeepStream crop-Pose GIE ID와 SGIE 연결 설정을 검증한다."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    1: ("config/deepstream/config_infer_person_primary.txt", "models/person/yolov8n.engine", None),
    2: ("config/deepstream/config_infer_pose_sgie.txt", "models/fall/yolov8n-pose.engine", 1),
    3: ("config/deepstream/config_infer_helmet_sgie.txt", "models/head/helmet_model.engine", 1),
    4: ("config/deepstream/config_infer_pa100k_sgie.txt", "models/appearance/pa100k_resnet50_attr.engine", 1),
}


def _property(text: str, name: str) -> str | None:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\S+)\s*$", text)
    return match.group(1) if match else None


def main() -> int:
    errors: list[str] = []
    checks: list[dict[str, object]] = []
    for gie_id, (config_relative, model_relative, parent_id) in EXPECTED.items():
        config_path = ROOT / config_relative
        model_path = ROOT / model_relative
        text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
        actual_id = _property(text, "gie-unique-id")
        actual_parent = _property(text, "operate-on-gie-id")
        expected_parent = str(parent_id) if parent_id is not None else None
        row = {
            "gie_id": gie_id,
            "config": config_relative,
            "model": model_relative,
            "config_exists": config_path.is_file(),
            "model_exists": model_path.is_file(),
            "configured_gie_id": actual_id,
            "operate_on_gie_id": actual_parent,
        }
        checks.append(row)
        if not config_path.is_file():
            errors.append(f"missing config: {config_relative}")
        if not model_path.is_file():
            errors.append(f"missing model: {model_relative}")
        if actual_id != str(gie_id):
            errors.append(f"{config_relative}: gie-unique-id={actual_id!r}, expected {gie_id}")
        if actual_parent != expected_parent:
            errors.append(
                f"{config_relative}: operate-on-gie-id={actual_parent!r}, expected {expected_parent!r}"
            )

    payload = {"passed": not errors, "checks": checks, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
