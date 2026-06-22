#!/usr/bin/env python3
"""Check falldata auxiliary fall verifier runtime readiness."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "falldata/2. AI학습모델파일/영상/낙상분류/FNF_RF_SMOTE_CAM_1.pkl"
)
DEFAULT_MODEL_PYTHON = PROJECT_ROOT / ".venv-falldata/bin/python"
DEFAULT_MEDIAPIPE_PYTHON = PROJECT_ROOT / ".venv-mediapipe/bin/python"
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts/datasets/extract_falldata_mediapipe_features.py"
SMOKE_SCRIPT = PROJECT_ROOT / "scripts/datasets/smoke_falldata_video_model.py"


def _run(command: list[str], timeout: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {
            "passed": False,
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "passed": proc.returncode == 0,
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _exists_check(label: str, path: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _parse_probability(output: str) -> list[float] | None:
    match = re.search(r"predict_proba:\s*\[\[([^\]]+)\]\]", output)
    if not match:
        return None
    return [float(part.strip()) for part in match.group(1).split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-python", type=Path, default=DEFAULT_MODEL_PYTHON)
    parser.add_argument("--mediapipe-python", type=Path, default=DEFAULT_MEDIAPIPE_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional sample video for end-to-end MediaPipe extraction smoke.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Frame limit for optional video smoke.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = [
        _exists_check("model_python", args.model_python),
        _exists_check("mediapipe_python", args.mediapipe_python),
        _exists_check("model", args.model),
        _exists_check("extract_script", EXTRACT_SCRIPT),
        _exists_check("smoke_script", SMOKE_SCRIPT),
    ]

    payload: dict[str, Any] = {
        "passed": False,
        "checks": checks,
        "synthetic_smoke": None,
        "video_smoke": None,
    }

    missing = [check for check in checks if not check["exists"]]
    if missing:
        payload["error"] = "missing required paths"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    synthetic = _run(
        [
            str(args.model_python),
            str(SMOKE_SCRIPT),
            "--model",
            str(args.model),
        ],
        timeout=args.timeout,
    )
    synthetic["probability"] = _parse_probability(str(synthetic.get("stdout", "")))
    payload["synthetic_smoke"] = synthetic
    if not synthetic["passed"]:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    if args.video is not None:
        with tempfile.TemporaryDirectory(prefix="falldata_health_") as tmp:
            feature_dir = Path(tmp) / "features"
            extract = _run(
                [
                    str(args.mediapipe_python),
                    str(EXTRACT_SCRIPT),
                    "--video",
                    str(args.video),
                    "--output-dir",
                    str(feature_dir),
                    "--max-frames",
                    str(args.max_frames),
                ],
                timeout=args.timeout,
            )
            infer: dict[str, Any] | None = None
            if extract["passed"]:
                infer = _run(
                    [
                        str(args.model_python),
                        str(SMOKE_SCRIPT),
                        "--model",
                        str(args.model),
                        "--sequence-dir",
                        str(feature_dir),
                    ],
                    timeout=args.timeout,
                )
                infer["probability"] = _parse_probability(str(infer.get("stdout", "")))
            payload["video_smoke"] = {
                "extract": extract,
                "inference": infer,
            }
            if not extract["passed"] or not (infer and infer["passed"]):
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 1

    payload["passed"] = True
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
