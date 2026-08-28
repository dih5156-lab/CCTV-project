"""Verify the function-based model directory layout and manifest paths."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models"
REQUIRED_DIRS = ("head", "fall", "person", "appearance", "legacy")


def main() -> int:
    manifest_path = MODEL_ROOT / "model_manifest.json"
    errors: list[str] = []
    for directory in REQUIRED_DIRS:
        path = MODEL_ROOT / directory
        if not path.is_dir():
            errors.append(f"missing model directory: {path}")
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

    payload = {"passed": not errors, "required_dirs": list(REQUIRED_DIRS), "manifest_artifacts_checked": checked, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
