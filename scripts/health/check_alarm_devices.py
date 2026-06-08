"""Check alarm device configuration and TCP reachability."""

from __future__ import annotations

import argparse
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class DeviceCheck:
    name: str
    host_env: str
    port_env: str
    default_port: int
    required_env: tuple[str, ...] = ()


DEVICE_CHECKS = (
    DeviceCheck(
        name="speaker",
        host_env="SPEAKER_HOST",
        port_env="SPEAKER_PORT",
        default_port=80,
        required_env=("SPEAKER_USER", "SPEAKER_PASSWORD"),
    ),
    DeviceCheck(
        name="siren",
        host_env="SIREN_HOST",
        port_env="SIREN_PORT",
        default_port=80,
        required_env=("SIREN_USER", "SIREN_PASSWORD"),
    ),
    DeviceCheck(
        name="signboard",
        host_env="SIGNBOARD_HOST",
        port_env="SIGNBOARD_PORT",
        default_port=5000,
    ),
)


def _env_value(env: Mapping[str, str], key: str) -> str:
    return str(env.get(key, "")).strip()


def load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _parse_port(raw: str, default_port: int) -> tuple[int, str | None]:
    if not raw:
        return default_port, None
    try:
        port = int(raw)
    except ValueError:
        return default_port, f"invalid port: {raw!r}"
    if port <= 0 or port > 65535:
        return default_port, f"port out of range: {port}"
    return port, None


def _can_connect(host: str, port: int, timeout: float) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, ""
    except OSError as exc:
        return False, str(exc)


def run_device_check(
    check: DeviceCheck,
    *,
    env: Mapping[str, str],
    timeout: float,
    skip_network: bool = False,
) -> dict:
    host = _env_value(env, check.host_env)
    port, port_error = _parse_port(_env_value(env, check.port_env), check.default_port)
    missing = [key for key in (check.host_env, *check.required_env) if not _env_value(env, key)]

    if missing or port_error:
        return {
            "name": check.name,
            "configured": False,
            "reachable": False,
            "host": host,
            "port": port,
            "missing_env": missing,
            "detail": port_error or "required environment variables are missing",
        }

    if skip_network:
        return {
            "name": check.name,
            "configured": True,
            "reachable": None,
            "host": host,
            "port": port,
            "missing_env": [],
            "detail": "network check skipped",
        }

    reachable, detail = _can_connect(host, port, timeout)
    return {
        "name": check.name,
        "configured": True,
        "reachable": reachable,
        "host": host,
        "port": port,
        "missing_env": [],
        "detail": detail,
    }


def run_checks(
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = 2.0,
    skip_network: bool = False,
    allow_unconfigured: bool = False,
) -> dict:
    source_env = env if env is not None else {**load_env_file(".env"), **os.environ}
    checks = [
        run_device_check(
            check,
            env=source_env,
            timeout=timeout,
            skip_network=skip_network,
        )
        for check in DEVICE_CHECKS
    ]
    configured = allow_unconfigured or all(bool(check["configured"]) for check in checks)
    reachable = all(
        check["reachable"] is not False
        or (allow_unconfigured and not check["configured"])
        for check in checks
    )
    return {
        "passed": configured and reachable,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check alarm device env settings and TCP reachability."
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="TCP timeout seconds.")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file to read before process environment overrides.",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Only validate environment variables and port values.",
    )
    parser.add_argument(
        "--allow-unconfigured",
        action="store_true",
        help="Return success even when optional alarm devices are not configured.",
    )
    args = parser.parse_args()

    merged_env = {**load_env_file(args.env_file), **os.environ}
    result = run_checks(
        env=merged_env,
        timeout=args.timeout,
        skip_network=args.skip_network,
        allow_unconfigured=args.allow_unconfigured,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
