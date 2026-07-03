#!/usr/bin/env python3
"""Check falldata auxiliary fall verifier runtime readiness."""

from __future__ import annotations

import argparse
import json
import os
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
MODEL_REQUIREMENTS = PROJECT_ROOT / "requirements/falldata-model.txt"
MEDIAPIPE_REQUIREMENTS = PROJECT_ROOT / "requirements/falldata-mediapipe.txt"
MODEL_VERSION_RULES = {
    "numpy": {"min": "1.26.1", "max_exclusive": "2.0.0"},
    "scipy": {"min": "1.11.3", "max_exclusive": "1.12.0"},
    "scikit-learn": {"min": "1.3.2", "max_exclusive": "1.4.0"},
    "joblib": {"min": "1.3.2", "max_exclusive": "1.4.0"},
}
MEDIAPIPE_VERSION_RULES = {
    "mediapipe": {"min": "0.10.14", "max_exclusive": "0.11.0"},
    "opencv-python-headless": {"min": "4.8.0", "max_exclusive": "5.0.0"},
    "numpy": {"min": "2.0.0", "max_exclusive": "3.0.0"},
}


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in re.split(r"[^0-9]+", value):
        if part:
            parts.append(int(part))
    return tuple(parts)


def _version_in_range(version: str, rule: dict[str, str]) -> bool:
    parsed = _version_tuple(version)
    min_version = _version_tuple(rule["min"])
    max_version = _version_tuple(rule["max_exclusive"])
    return parsed >= min_version and parsed < max_version


def _version_check(
    *,
    label: str,
    python_path: Path,
    rules: dict[str, dict[str, str]],
    timeout: float,
) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m, json; "
        f"packages={list(rules)!r}; "
        "print(json.dumps({name: m.version(name) for name in packages}))"
    )
    result = _run([str(python_path), "-c", code], timeout=timeout)
    payload: dict[str, Any] = {
        "label": label,
        "python": str(python_path),
        "passed": False,
        "rules": rules,
        "versions": {},
        "command": result.get("command"),
        "returncode": result.get("returncode"),
        "stderr": result.get("stderr", ""),
    }
    if not result.get("passed"):
        payload["error"] = result.get("error") or result.get("stderr") or "version command failed"
        return payload
    try:
        versions = json.loads(str(result.get("stdout", "{}")))
    except json.JSONDecodeError as exc:
        payload["error"] = f"invalid version JSON: {exc}"
        payload["stdout"] = result.get("stdout", "")
        return payload
    checks = []
    for package, rule in rules.items():
        version = str(versions.get(package, ""))
        checks.append(
            {
                "package": package,
                "version": version,
                "min": rule["min"],
                "max_exclusive": rule["max_exclusive"],
                "passed": bool(version) and _version_in_range(version, rule),
            }
        )
    payload["versions"] = versions
    payload["checks"] = checks
    payload["passed"] = all(check["passed"] for check in checks)
    return payload


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


def _policy_check(
    *,
    mode: str,
    fail_open_on_unavailable: bool,
    confirm_borderline: bool,
    compare_veto_enabled: bool,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"shadow", "confirm"}:
        errors.append("FALLDATA_AUX_MODE must be shadow or confirm")
    if normalized_mode == "confirm":
        warnings.append("confirm mode can suppress fall events when aux returns ok/confirmed=false")
        if not fail_open_on_unavailable:
            errors.append("confirm mode requires fail-open on unavailable aux results for deployment")
    if confirm_borderline:
        warnings.append("DeepStream borderline confirm can delay or cancel low-score fall events")
    if compare_veto_enabled:
        warnings.append("compare veto can cancel aux-confirmed fall events; use only after field review")
    return {
        "passed": not errors,
        "mode": normalized_mode,
        "fail_open_on_unavailable": fail_open_on_unavailable,
        "confirm_borderline": confirm_borderline,
        "compare_veto_enabled": compare_veto_enabled,
        "warnings": warnings,
        "errors": errors,
    }


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
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        help="Skip package version checks and only run path/smoke checks.",
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("FALLDATA_AUX_MODE", "shadow"),
        help="Runtime policy mode to validate: shadow or confirm.",
    )
    parser.add_argument(
        "--fail-open-on-unavailable",
        action=argparse.BooleanOptionalAction,
        default=_parse_bool(os.environ.get("FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE"), True),
        help="Whether confirm mode publishes original fall events when aux is unavailable.",
    )
    parser.add_argument(
        "--confirm-borderline",
        action=argparse.BooleanOptionalAction,
        default=_parse_bool(os.environ.get("FALLDATA_AUX_CONFIRM_BORDERLINE"), False),
        help="Whether DeepStream borderline fall events wait for aux confirmation.",
    )
    parser.add_argument(
        "--compare-veto-enabled",
        action=argparse.BooleanOptionalAction,
        default=_parse_bool(os.environ.get("FALLDATA_AUX_COMPARE_VETO_ENABLED"), False),
        help="Whether the compare model can veto aux-confirmed fall events.",
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
        _exists_check("model_requirements", MODEL_REQUIREMENTS),
        _exists_check("mediapipe_requirements", MEDIAPIPE_REQUIREMENTS),
    ]

    payload: dict[str, Any] = {
        "passed": False,
        "checks": checks,
        "policy_check": _policy_check(
            mode=args.mode,
            fail_open_on_unavailable=args.fail_open_on_unavailable,
            confirm_borderline=args.confirm_borderline,
            compare_veto_enabled=args.compare_veto_enabled,
        ),
        "version_checks": [],
        "synthetic_smoke": None,
        "video_smoke": None,
    }

    if not payload["policy_check"]["passed"]:
        payload["error"] = "unsafe falldata aux policy"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    missing = [check for check in checks if not check["exists"]]
    if missing:
        payload["error"] = "missing required paths"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    if not args.skip_version_check:
        version_checks = [
            _version_check(
                label="model_python",
                python_path=args.model_python,
                rules=MODEL_VERSION_RULES,
                timeout=args.timeout,
            ),
            _version_check(
                label="mediapipe_python",
                python_path=args.mediapipe_python,
                rules=MEDIAPIPE_VERSION_RULES,
                timeout=args.timeout,
            ),
        ]
        payload["version_checks"] = version_checks
        failed_version_checks = [check for check in version_checks if not check["passed"]]
        if failed_version_checks:
            payload["error"] = "version check failed"
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
