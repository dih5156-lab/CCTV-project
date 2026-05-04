"""Run offline readiness checks that do not require field devices to be powered."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    command: list[str]
    timeout: int = 60
    env: dict[str, str] | None = None


QUICK_TESTS = (
    "tests/test_public_api.py",
    "tests/test_api_auth.py",
    "tests/test_control_api.py",
    "tests/test_internal_health_endpoints.py",
    "tests/test_appearance_analyzer.py",
    "tests/test_appearance_pipeline.py",
    "tests/test_deepstream_event_factory.py",
    "tests/test_deepstream_face_context.py",
    "tests/test_event_context.py",
    "tests/test_synthetic_object_ids.py",
    "tests/test_yolo_postprocess.py",
    "tests/test_adapter_service.py",
    "tests/test_device_service.py",
)

PARSER_TESTS = (
    "parser-python/tests/test_edgex_outbox.py",
    "parser-python/tests/test_tlv_parser.py",
)


def build_checks(*, full: bool = False) -> list[Check]:
    """Build an offline verification plan for the current checkout."""
    checks = [
        Check("deployment readiness", [sys.executable, "scripts/check_deployment_readiness.py"], timeout=90),
        Check(
            "alarm config without network",
            [
                sys.executable,
                "scripts/check_alarm_devices.py",
                "--skip-network",
                "--allow-unconfigured",
            ],
        ),
    ]

    if full:
        checks.append(Check("full pytest", [sys.executable, "-m", "pytest"], timeout=300))
    else:
        checks.append(
            Check(
                "offline focused pytest",
                [sys.executable, "-m", "pytest", *QUICK_TESTS],
                timeout=180,
            )
        )

    parser_env = {"PYTHONPATH": "parser-python"}
    checks.append(
        Check(
            "parser pytest",
            [sys.executable, "-m", "pytest", "-c", "/dev/null", *PARSER_TESTS],
            timeout=120,
            env=parser_env,
        )
    )
    return checks


def _run_check(check: Check) -> tuple[bool, str]:
    env = os.environ.copy()
    if check.env:
        env.update(check.env)
    try:
        result = subprocess.run(
            check.command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=check.timeout,
            env=env,
        )
    except Exception as exc:
        return False, str(exc)

    if result.returncode == 0:
        return True, ""
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return False, output[-4000:]


def run_checks(*, full: bool = False) -> dict:
    results = []
    failed = False
    for check in build_checks(full=full):
        passed, detail = _run_check(check)
        failed = failed or not passed
        results.append(
            {
                "name": check.name,
                "passed": passed,
                "command": " ".join(check.command),
                "detail": detail,
            }
        )
    return {
        "passed": not failed,
        "mode": "full" if full else "quick",
        "device_network_required": False,
        "checks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline readiness checks without requiring alarm devices or cameras."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full pytest suite instead of the focused offline subset.",
    )
    args = parser.parse_args()

    result = run_checks(full=args.full)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
