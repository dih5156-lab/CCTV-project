"""Check runtime assumptions that docker compose config alone cannot catch."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# These images are known to be a risk on arm64 hosts when pulled without an
# explicit arm64-compatible tag or platform override in the default compose file.
ARM64_RISK_IMAGES = (
    "edgexfoundry/core-common-config-bootstrapper:",
    "edgexfoundry/core-data:",
    "edgexfoundry/core-metadata:",
    "edgexfoundry/device-rest:",
    "edgexfoundry/edgex-ui:",
)

ARM64_OVERRIDE_SERVICES = (
    "core-common-config-bootstrapper",
    "core-data",
    "core-metadata",
    "device-rest",
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _normalize_machine(machine: str | None = None) -> str:
    value = (machine or platform.machine()).strip().lower()
    if value in {"aarch64", "arm64"}:
        return "arm64"
    if value in {"x86_64", "amd64"}:
        return "amd64"
    return value or "unknown"


def check_default_compose_architecture(
    *,
    machine: str | None = None,
    compose_text: str | None = None,
    arm64_override_text: str | None = None,
) -> dict[str, Any]:
    """Detect default compose services likely to fail with exec format errors."""
    arch = _normalize_machine(machine)
    text = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    override = (
        arm64_override_text
        if arm64_override_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.arm64.yml")
    )
    risky_images = [image for image in ARM64_RISK_IMAGES if image in text]

    if arch != "arm64" or not risky_images:
        return {
            "name": "default compose architecture",
            "passed": True,
            "detail": "",
        }

    has_platform_override = "platform:" in text
    if has_platform_override:
        return {
            "name": "default compose architecture",
            "passed": True,
            "detail": "arm64 host detected; compose contains platform override",
        }

    override_has_platform = "platform: linux/arm64" in override
    override_has_services = all(f"{service}:" in override for service in ARM64_OVERRIDE_SERVICES)
    override_disables_ui = "ui:" in override and "profiles:" in override
    if override_has_platform and override_has_services and override_disables_ui:
        return {
            "name": "default compose architecture",
            "passed": True,
            "detail": (
                "arm64 host detected; use docker-compose.arm64.yml with docker-compose.yml "
                "when starting the full EdgeX stack. EdgeX UI is excluded on arm64."
            ),
        }

    return {
        "name": "default compose architecture",
        "passed": False,
        "detail": (
            "arm64 host detected but docker-compose.yml includes EdgeX images that may be amd64-only: "
            f"{', '.join(risky_images)}. Use docker-compose.arm64.yml with docker-compose.yml, "
            "use docker-compose.jetson.yml for Jetson-specific deployment, or run only the "
            "API/action services from docker-compose.yml."
        ),
    }


def check_parser_db_defaults(
    env_text: str | None = None,
    compose_text: str | None = None,
) -> dict[str, Any]:
    """Warn when parser DB config is likely to resolve to localhost inside a container."""
    env_path = PROJECT_ROOT / "parser-python" / ".env"
    text = env_text if env_text is not None else _read_text(env_path)
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    if not text:
        return {
            "name": "aiot parser database config",
            "passed": True,
            "detail": "",
        }

    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    db_host = values.get("DB_HOST", "localhost")
    if db_host in {"localhost", "127.0.0.1", "::1"}:
        if "aiot-parser-db:" in compose and "DB_HOST: aiot-parser-db" in compose:
            return {
                "name": "aiot parser database config",
                "passed": True,
                "detail": "parser .env uses localhost, but docker-compose.yml overrides DB_HOST to aiot-parser-db",
            }
        return {
            "name": "aiot parser database config",
            "passed": False,
            "detail": (
                "parser-python/.env uses DB_HOST=localhost. Inside the aiot-parser container, "
                "localhost is the container itself, so PostgreSQL must run in the same container "
                "or DB_HOST must point to a compose service/external host."
            ),
        }

    return {
        "name": "aiot parser database config",
        "passed": True,
        "detail": "",
    }


def run_checks() -> list[dict[str, Any]]:
    return [
        check_default_compose_architecture(),
        check_parser_db_defaults(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check runtime assumptions for CCTV compose deployment.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    checks = run_checks()
    passed = all(check["passed"] for check in checks)
    payload = {"passed": passed, "checks": checks}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for check in checks:
            status = "PASS" if check["passed"] else "FAIL"
            detail = f" - {check['detail']}" if check["detail"] else ""
            print(f"[{status}] {check['name']}{detail}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
