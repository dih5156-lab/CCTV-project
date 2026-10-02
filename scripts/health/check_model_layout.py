"""기능별 모델 디렉터리 구조와 매니페스트 경로를 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models"
REQUIRED_DIRS = ("head", "fall", "person", "appearance", "legacy")
CROP_POSE_ARTIFACTS = {
    "config/deepstream/config_infer_person_primary.txt": "models/person/yolov8n.engine",
    "config/deepstream/config_infer_pose_sgie.txt": "models/fall/yolov8n-pose.engine",
    "config/deepstream/config_infer_helmet_sgie.txt": "models/head/helmet_model.engine",
    "config/deepstream/config_infer_pa100k_sgie.txt": "models/appearance/pa100k_resnet50_attr.engine",
}


def main() -> int:
    manifest_path = MODEL_ROOT / "model_manifest.json"
    errors: list[str] = []
    for directory in REQUIRED_DIRS:
        path = MODEL_ROOT / directory
        if not path.is_dir():
            errors.append(f"missing model directory: {path}")

    crop_pose_checks = []
    for config_relative, model_relative in CROP_POSE_ARTIFACTS.items():
        config_path = ROOT / config_relative
        model_path = ROOT / model_relative
        config_ok = config_path.is_file()
        model_ok = model_path.is_file()
        crop_pose_checks.append(
            {"config": config_relative, "model": model_relative, "config_exists": config_ok, "model_exists": model_ok}
        )
        if not config_ok:
            errors.append(f"missing crop-Pose config: {config_relative}")
        if not model_ok:
            errors.append(f"missing crop-Pose model: {model_relative}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"manifest unreadable: {exc}")
        manifest = {}

    checked = 0
    for model in manifest.get("models", []):
        for artifact_type, artifact in (model.get("artifacts") or {}).items():
            if not isinstance(artifact, str) or not artifact.startswith("models/"):
                continue
            checked += 1
            artifact_path = ROOT / artifact
            if not artifact_path.exists():
                errors.append(f"missing {model.get('name')} {artifact_type}: {artifact}")
            elif len(Path(artifact).parts) < 3:
                errors.append(f"uncategorized artifact path: {artifact}")

    payload = {
        "passed": not errors,
        "required_dirs": list(REQUIRED_DIRS),
        "manifest_artifacts_checked": checked,
        "crop_pose_checks": crop_pose_checks,
        "errors": errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
