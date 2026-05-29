#!/usr/bin/env python3
"""Validate and optionally repair runtime secret drift for the Jetson stack.

This check intentionally validates live state, not only compose syntax:
- .env is the single source for MQTT and AIoT DB passwords.
- Running service containers received the same MQTT secret values.
- Mosquitto accepts the .env credentials.
- PostgreSQL accepts the .env AIOT_DB_PASSWORD over TCP.

Use --fix only for local/edge operation recovery. It updates mosquitto/passwd
and the existing aiot-parser-db postgres password to match .env.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENV = ("MQTT_USER", "MQTT_PASSWORD", "AIOT_DB_PASSWORD")
MQTT_ENV_CONTAINERS = {
    "cctv-ai-engine": ("MQTT_USER", "MQTT_PASSWORD"),
    "cctv-action-layer": ("MQTT_USER", "MQTT_PASSWORD"),
    "aiot-parser": ("MQTT_USER", "MQTT_PASSWORD"),
    "cctv-sensor-rule-bridge": ("MQTT_USER", "MQTT_PASSWORD"),
    "edgex-app-rules-engine": ("MQTT_USER", "MQTT_PASSWORD"),
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _run(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=merged_env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _docker_available() -> CheckResult:
    result = _run(["docker", "info"], timeout=10)
    if result.returncode == 0:
        return CheckResult("docker access", True)
    return CheckResult("docker access", False, (result.stderr or result.stdout).strip())


def _inspect_container_env(container: str) -> dict[str, str] | None:
    result = _run(["docker", "inspect", "--format", "{{json .Config.Env}}", container], timeout=10)
    if result.returncode != 0:
        return None
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    values: dict[str, str] = {}
    for entry in entries or []:
        if "=" in entry:
            key, value = entry.split("=", 1)
            values[key] = value
    return values


def check_env_source(env_values: dict[str, str]) -> CheckResult:
    missing = [key for key in REQUIRED_ENV if not env_values.get(key)]
    if missing:
        return CheckResult("runtime .env secrets", False, "missing non-empty values: " + ", ".join(missing))
    return CheckResult("runtime .env secrets", True)


def check_container_mqtt_env(env_values: dict[str, str]) -> list[CheckResult]:
    results: list[CheckResult] = []
    expected = {
        "MQTT_USER": env_values.get("MQTT_USER", ""),
        "MQTT_PASSWORD": env_values.get("MQTT_PASSWORD", ""),
        "WRITABLE_INSECURESECRETS_MQTT_SECRETDATA_USERNAME": env_values.get("MQTT_USER", ""),
        "WRITABLE_INSECURESECRETS_MQTT_SECRETDATA_PASSWORD": env_values.get("MQTT_PASSWORD", ""),
    }
    for container, keys in MQTT_ENV_CONTAINERS.items():
        container_env = _inspect_container_env(container)
        if container_env is None:
            results.append(CheckResult(f"{container} mqtt env", False, "container not found or inspect failed"))
            continue
        mismatched = [key for key in keys if container_env.get(key) != expected[key]]
        if mismatched:
            results.append(CheckResult(f"{container} mqtt env", False, "mismatch: " + ", ".join(mismatched)))
        else:
            results.append(CheckResult(f"{container} mqtt env", True))
    return results


def check_mosquitto_auth(env_values: dict[str, str]) -> CheckResult:
    result = _run(
        [
            "docker",
            "exec",
            "edgex-mqtt-broker",
            "mosquitto_pub",
            "-h",
            "localhost",
            "-p",
            "1883",
            "-u",
            env_values["MQTT_USER"],
            "-P",
            env_values["MQTT_PASSWORD"],
            "-t",
            "health/secret-check",
            "-m",
            "ping",
        ],
        timeout=10,
    )
    if result.returncode == 0:
        return CheckResult("mosquitto accepts .env mqtt credentials", True)
    return CheckResult("mosquitto accepts .env mqtt credentials", False, (result.stderr or result.stdout).strip())


def check_postgres_password(env_values: dict[str, str]) -> CheckResult:
    sql = "SELECT 1;"
    result = _run(
        [
            "docker",
            "exec",
            "-e",
            f"PGPASSWORD={env_values['AIOT_DB_PASSWORD']}",
            "aiot-parser-db",
            "psql",
            "-h",
            "127.0.0.1",
            "-U",
            "postgres",
            "-d",
            "aiot_sensor",
            "-tAc",
            sql,
        ],
        timeout=10,
    )
    if result.returncode == 0 and result.stdout.strip() == "1":
        return CheckResult("aiot postgres accepts .env password", True)
    return CheckResult("aiot postgres accepts .env password", False, (result.stderr or result.stdout).strip())



def check_recent_mqtt_authorization_errors() -> CheckResult:
    result = _run(["docker", "logs", "--since", "30s", "edgex-mqtt-broker"], timeout=10)
    if result.returncode != 0:
        return CheckResult("recent mqtt authorization errors", False, (result.stderr or result.stdout).strip())
    lines = [
        line
        for line in (result.stderr + result.stdout).splitlines()
        if "not authorised" in line or "not authorized" in line
    ]
    if lines:
        samples = "; ".join(lines[-3:])
        return CheckResult("recent mqtt authorization errors", False, samples)
    return CheckResult("recent mqtt authorization errors", True)

def fix_mosquitto_password(env_values: dict[str, str]) -> CheckResult:
    passwd = PROJECT_ROOT / "mosquitto" / "passwd"
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{PROJECT_ROOT / 'mosquitto'}:/mosquitto/config",
            "eclipse-mosquitto:2.0",
            "mosquitto_passwd",
            "-b",
            "/mosquitto/config/passwd",
            env_values["MQTT_USER"],
            env_values["MQTT_PASSWORD"],
        ],
        timeout=30,
    )
    if result.returncode != 0:
        return CheckResult("fix mosquitto/passwd", False, (result.stderr or result.stdout).strip())
    try:
        passwd.chmod(0o700)
    except OSError as exc:
        return CheckResult("fix mosquitto/passwd", False, f"password updated but chmod failed: {exc}")
    return CheckResult("fix mosquitto/passwd", True)


def fix_postgres_password(env_values: dict[str, str]) -> CheckResult:
    password_sql = env_values["AIOT_DB_PASSWORD"].replace("'", "''")
    result = _run(
        [
            "docker",
            "exec",
            "aiot-parser-db",
            "psql",
            "-U",
            "postgres",
            "-d",
            "aiot_sensor",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            f"ALTER USER postgres WITH PASSWORD '{password_sql}';",
        ],
        timeout=10,
    )
    if result.returncode == 0:
        return CheckResult("fix aiot postgres password", True)
    return CheckResult("fix aiot postgres password", False, (result.stderr or result.stdout).strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Check live runtime secret consistency for CCTV Jetson deployment.")
    parser.add_argument("--env-file", default=".env", help="Env file used as the single runtime secret source.")
    parser.add_argument("--fix", action="store_true", help="Repair mosquitto/passwd and AIoT Postgres password from .env.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    env_values = _parse_env_file(PROJECT_ROOT / args.env_file)
    results = [check_env_source(env_values)]
    if results[-1].passed:
        docker_check = _docker_available()
        results.append(docker_check)
        if docker_check.passed:
            if args.fix:
                results.append(fix_mosquitto_password(env_values))
                results.append(fix_postgres_password(env_values))
            results.extend(check_container_mqtt_env(env_values))
            results.append(check_mosquitto_auth(env_values))
            results.append(check_postgres_password(env_values))
            results.append(check_recent_mqtt_authorization_errors())

    payload: dict[str, Any] = {
        "passed": all(result.passed for result in results),
        "checks": [result.__dict__ for result in results],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            detail = f" - {result.detail}" if result.detail else ""
            print(f"[{status}] {result.name}{detail}")

    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
