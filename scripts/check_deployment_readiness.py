"""Run lightweight deployment readiness checks that do not require Docker builds."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    command: list[str]


CHECKS = (
    Check("sensitive defaults", [sys.executable, "scripts/check_sensitive_defaults.py"]),
    Check("dockerfile copy sources", [sys.executable, "scripts/check_dockerfile_sources.py"]),
    Check("monitoring config", [sys.executable, "scripts/check_monitoring_config.py"]),
    Check("compose runtime assumptions", [sys.executable, "scripts/check_compose_runtime_assumptions.py"]),
    Check("runtime secret consistency script", [sys.executable, "scripts/check_runtime_secret_consistency.py", "--help"]),
    Check("model manifest JSON", [sys.executable, "-m", "json.tool", "models/model_manifest.json"]),
    Check("model manifest artifacts", [sys.executable, "scripts/check_model_report.py", "--check-artifacts"]),
    Check("grafana dashboard JSON", [sys.executable, "-m", "json.tool", "monitoring/grafana/provisioning/dashboards/cctv_overview.json"]),
    Check("alarm device config", [sys.executable, "scripts/check_alarm_devices.py", "--skip-network", "--allow-unconfigured"]),
    Check("python scripts compile", [sys.executable, "-m", "py_compile", "scripts/evaluate_detection.py", "scripts/check_model_report.py", "scripts/check_sensitive_defaults.py", "scripts/check_monitoring_config.py", "scripts/check_compose_runtime_assumptions.py", "scripts/check_runtime_secret_consistency.py", "scripts/check_alarm_devices.py", "scripts/check_field_network.py", "scripts/check_offline_readiness.py", "scripts/smoke_test_deployment.py", "scripts/smoke_test_data_flow.py", "runners/run_sensor_rule_bridge.py"]),
    Check("docker compose config", ["docker", "compose", "config"]),
    Check("docker compose jetson config", ["docker", "compose", "-f", "docker-compose.jetson.yml", "config"]),
    Check("docker-build.sh syntax", ["bash", "-n", "docker-build.sh"]),
    Check("runner action bridge entrypoint", [sys.executable, "runners/run_action_bridge.py", "--help"]),
    Check("runner alert api entrypoint", [sys.executable, "runners/run_alert_api.py", "--help"]),
    Check("runner edgex adapter entrypoint", [sys.executable, "runners/run_edgex_adapter.py", "--help"]),
    Check("runner kuiper rules entrypoint", [sys.executable, "runners/run_kuiper_rules.py", "--help"]),
    Check("runner public api entrypoint", [sys.executable, "runners/run_public_api.py", "--help"]),
    Check("runner sensor rule bridge entrypoint", [sys.executable, "runners/run_sensor_rule_bridge.py", "--help"]),
)


def _run_check(check: Check) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            check.command,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return False, str(exc)

    if result.returncode == 0:
        return True, ""
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return False, output


def main() -> int:
    results = []
    failed = False
    for check in CHECKS:
        passed, detail = _run_check(check)
        failed = failed or not passed
        results.append(
            {
                "name": check.name,
                "passed": passed,
                "detail": detail,
            }
        )

    print(json.dumps({"passed": not failed, "checks": results}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
