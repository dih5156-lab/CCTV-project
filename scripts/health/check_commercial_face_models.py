"""Validate commercial face model artifacts, licenses, and TensorRT engines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models.fetch_commercial_face_models import load_manifest, verify_file  # noqa: E402, I001

DEFAULT_MANIFEST = Path("config/models/commercial_face_models.json")
DEFAULT_MODEL_DIR = Path("models/commercial_face")
_FORBIDDEN_MARKERS = ("insightface", "buffalo_l", "w600k", "det_10g")


def _result(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def check_commercial_face_models(
    manifest_path: Path,
    model_dir: Path,
    *,
    require_engines: bool = True,
) -> dict[str, Any]:
    try:
        models = load_manifest(manifest_path)
    except Exception as exc:
        return {"passed": False, "checks": [_result("manifest", False, str(exc))]}

    serialized_manifest = manifest_path.read_text(encoding="utf-8").lower()
    forbidden = [marker for marker in _FORBIDDEN_MARKERS if marker in serialized_manifest]
    checks = [
        _result(
            "commercial model policy",
            not forbidden,
            "" if not forbidden else f"InsightFace research artifact marker found: {forbidden}",
        )
    ]
    for model_name, model in models.items():
        for artifact_name in ("model", "license"):
            artifact = model["artifacts"][artifact_name]
            path = model_dir / artifact["filename"]
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"missing: {path}")
                verify_file(path, artifact["sha256"])
            except Exception as exc:
                checks.append(_result(f"{model_name} {artifact_name}", False, str(exc)))
            else:
                checks.append(_result(f"{model_name} {artifact_name}", True))

        if require_engines:
            engine_filename = model.get("engine_filename")
            engine_path = model_dir / str(engine_filename or "")
            engine_ready = bool(
                engine_filename
                and engine_path.is_file()
                and engine_path.stat().st_size > 0
            )
            detail = "" if engine_ready else f"missing or empty engine: {engine_path}"
            checks.append(_result(f"{model_name} engine", engine_ready, detail))

    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--allow-missing-engines", action="store_true")
    args = parser.parse_args()
    result = check_commercial_face_models(
        args.manifest,
        args.model_dir,
        require_engines=not args.allow_missing_engines,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
