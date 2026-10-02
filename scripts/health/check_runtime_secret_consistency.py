#!/usr/bin/env python3
"""Jetson 스택의 실행 보안 값 불일치를 검증하고 필요하면 복구한다.

이 검사는 Compose 문법뿐 아니라 실제 실행 상태를 확인한다.
- MQTT와 AIoT DB 비밀번호의 단일 기준은 .env 파일이다.
- 실행 중인 서비스 컨테이너가 동일한 MQTT 보안 값을 전달받았는지 확인한다.
- Mosquitto가 .env 인증 정보를 받아들이는지 확인한다.
- PostgreSQL이 TCP를 통해 .env의 AIOT_DB_PASSWORD를 받아들이는지 확인한다.

--fix 옵션은 로컬·엣지 운영 복구가 필요한 경우에만 사용한다. .env 기준으로
mosquitto/passwd와 기존 aiot-parser-db PostgreSQL 비밀번호를 갱신한다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ENV = ("MQTT_USER", "MQTT_PASSWORD", "AIOT_DB_PASSWORD")
MQTT_ENV_CONTAINERS = {
    "cctv-ai-engine": ("MQTT_USER", "MQTT_PASSWORD"),
    "cctv-action-layer": ("MQTT_USER", "MQTT_PASSWORD"),
    "aiot-parser": ("MQTT_USER", "MQTT_PASSWORD"),
    "cctv-sensor-rule-bridge": ("MQTT_USER", "MQTT_PASSWORD"),
    "edgex-app-rules-engine": ("MQTT_USER", "MQTT_PASSWORD"),
}


def _resolve_env_file(env_file_arg: str) -> Path:
    requested = Path(env_file_arg)
    if requested.is_absolute():
        return requested

    candidate = PROJECT_ROOT / requested
    if candidate.exists() or env_file_arg != ".env":
        return candidate

    jetson_candidate = PROJECT_ROOT / ".env.jetson"
    if jetson_candidate.exists():
        return jetson_candidate

    return candidate


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
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file used as the single runtime secret source. Defaults to .env, or .env.jetson when only that file exists.",
    )
    parser.add_argument("--fix", action="store_true", help="Repair mosquitto/passwd and AIoT Postgres password from .env.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    env_file_path = _resolve_env_file(args.env_file)
    env_values = _parse_env_file(env_file_path)
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
