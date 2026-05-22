"""Fail when sensitive runtime defaults are committed into shared config files."""

from __future__ import annotations

import re
from pathlib import Path


CHECK_FILES = (
    Path(".env.example"),
    Path(".env.jetson"),
    Path(".env.jetson.example"),
    Path("docker-compose.yml"),
    Path("docker-compose.jetson.yml"),
)

FORBIDDEN_LITERAL_PATTERNS = (
    re.compile(r"sawwaveap", re.IGNORECASE),
)

SECRET_ASSIGNMENT_PATTERNS = (
    re.compile(
        r"^(SPEAKER_PASSWORD|SIREN_PASSWORD|INTERNAL_SERVICE_TOKEN|PUBLIC_API_KEY|"
        r"AIOT_DB_PASSWORD|GRAFANA_ADMIN_PASSWORD|MQTT_PASSWORD)=(.+)$"
    ),
    re.compile(
        r"^\s*(SPEAKER_PASSWORD|SIREN_PASSWORD|INTERNAL_SERVICE_TOKEN|PUBLIC_API_KEY|"
        r"AIOT_DB_PASSWORD|GRAFANA_ADMIN_PASSWORD|MQTT_PASSWORD):\s*(.+)$"
    ),
)

UNSAFE_ASSIGNMENT_PATTERNS = (
    re.compile(r"^(CORS_ORIGINS)=(\*)$"),
    re.compile(r"^\s*(CORS_ORIGINS):\s*(\*)$"),
    re.compile(r"^\s*(CORS_ORIGINS):\s*\$\{CORS_ORIGINS:-\*\}$"),
)


def _is_allowed_value(value: str) -> bool:
    normalized = value.strip().strip("\"'")
    if normalized in {"", "null", "None"}:
        return True
    if normalized.startswith("${") and normalized.endswith("}"):
        default_part = normalized.split(":-", 1)
        return len(default_part) == 1 or default_part[1].rstrip("}") == ""
    return False


def find_sensitive_defaults() -> list[str]:
    findings: list[str] = []
    for path in CHECK_FILES:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern in FORBIDDEN_LITERAL_PATTERNS:
                if pattern.search(line):
                    findings.append(f"{path}:{line_no}: forbidden literal pattern")

            for pattern in SECRET_ASSIGNMENT_PATTERNS:
                match = pattern.match(line)
                if match and not _is_allowed_value(match.group(2)):
                    findings.append(f"{path}:{line_no}: non-empty default for {match.group(1)}")

            for pattern in UNSAFE_ASSIGNMENT_PATTERNS:
                match = pattern.match(line)
                if match:
                    findings.append(f"{path}:{line_no}: unsafe default for {match.group(1)}")
    return findings


def main() -> int:
    findings = find_sensitive_defaults()
    if findings:
        print("Sensitive defaults found:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("No sensitive defaults found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
