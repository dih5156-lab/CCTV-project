"""Check host routing toward configured field alarm devices."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
from pathlib import Path
from typing import Mapping

from check_alarm_devices import DEVICE_CHECKS, load_env_file


def _env_value(env: Mapping[str, str], key: str) -> str:
    return str(env.get(key, "")).strip()


def _run_ip_route_get(host: str, timeout: float) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["ip", "route", "get", host],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return False, str(exc)
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return result.returncode == 0, output


def _parse_route_output(output: str) -> dict[str, str]:
    tokens = output.split()
    parsed = {"interface": "", "source": "", "gateway": ""}
    for index, token in enumerate(tokens):
        if token == "dev" and index + 1 < len(tokens):
            parsed["interface"] = tokens[index + 1]
        elif token == "src" and index + 1 < len(tokens):
            parsed["source"] = tokens[index + 1]
        elif token == "via" and index + 1 < len(tokens):
            parsed["gateway"] = tokens[index + 1]
    return parsed


def _source_matches_subnet(source: str, expected_subnet: str) -> tuple[bool, str]:
    if not expected_subnet:
        return True, ""
    if not source:
        return False, "route source address is missing"
    try:
        return ipaddress.ip_address(source) in ipaddress.ip_network(expected_subnet, strict=False), ""
    except ValueError as exc:
        return False, str(exc)


def run_route_check(
    name: str,
    host: str,
    *,
    timeout: float = 2.0,
    expected_interface: str = "",
    expected_subnet: str = "",
    allow_permission_denied: bool = False,
) -> dict:
    route_ok, detail = _run_ip_route_get(host, timeout)
    permission_denied = "Operation not permitted" in detail or "Cannot open netlink socket" in detail
    if not route_ok and allow_permission_denied and permission_denied:
        return {
            "name": name,
            "configured": True,
            "host": host,
            "route_ok": None,
            "passed": True,
            "skipped": True,
            "interface": "",
            "source": "",
            "gateway": "",
            "detail": detail,
        }

    parsed = _parse_route_output(detail) if route_ok else {"interface": "", "source": "", "gateway": ""}

    interface_ok = not expected_interface or parsed["interface"] == expected_interface
    subnet_ok, subnet_detail = _source_matches_subnet(parsed["source"], expected_subnet)
    passed = route_ok and interface_ok and subnet_ok

    reasons: list[str] = []
    if not route_ok:
        reasons.append(detail)
    if not interface_ok:
        reasons.append(f"expected interface {expected_interface}, got {parsed['interface'] or 'unknown'}")
    if not subnet_ok:
        reasons.append(subnet_detail)

    return {
        "name": name,
        "configured": True,
        "host": host,
        "route_ok": route_ok,
        "passed": passed,
        "skipped": False,
        "interface": parsed["interface"],
        "source": parsed["source"],
        "gateway": parsed["gateway"],
        "detail": "; ".join(reason for reason in reasons if reason),
    }


def run_checks(
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = 2.0,
    allow_unconfigured: bool = False,
    expected_interface: str = "",
    expected_subnet: str = "",
    allow_permission_denied: bool = False,
) -> dict:
    source_env = env if env is not None else {**load_env_file(".env"), **os.environ}
    checks = []
    failed = False

    for device in DEVICE_CHECKS:
        host = _env_value(source_env, device.host_env)
        if not host:
            check = {
                "name": device.name,
                "configured": False,
                "host": "",
                "route_ok": None,
                "passed": bool(allow_unconfigured),
                "skipped": bool(allow_unconfigured),
                "interface": "",
                "source": "",
                "gateway": "",
                "detail": f"{device.host_env} is not configured",
            }
        else:
            check = run_route_check(
                device.name,
                host,
                timeout=timeout,
                expected_interface=expected_interface,
                expected_subnet=expected_subnet,
                allow_permission_denied=allow_permission_denied,
            )
        failed = failed or not bool(check["passed"])
        checks.append(check)

    return {
        "passed": not failed,
        "device_network_required": False,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check local host routing toward configured field alarm device IPs."
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Command timeout seconds.")
    parser.add_argument("--env-file", default=".env", help="Env file to read before process environment overrides.")
    parser.add_argument(
        "--allow-unconfigured",
        action="store_true",
        help="Return success when optional device hosts are not configured.",
    )
    parser.add_argument(
        "--allow-permission-denied",
        action="store_true",
        help="Return success when ip route cannot access netlink in a sandboxed environment.",
    )
    parser.add_argument(
        "--expected-interface",
        default="",
        help="Require routes to use this interface, e.g. eno1.",
    )
    parser.add_argument(
        "--expected-subnet",
        default="",
        help="Require route source IP to belong to this subnet, e.g. 192.168.88.0/24.",
    )
    args = parser.parse_args()

    merged_env = {**load_env_file(Path(args.env_file)), **os.environ}
    result = run_checks(
        env=merged_env,
        timeout=args.timeout,
        allow_unconfigured=args.allow_unconfigured,
        expected_interface=args.expected_interface,
        expected_subnet=args.expected_subnet,
        allow_permission_denied=args.allow_permission_denied,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
