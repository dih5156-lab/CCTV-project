"""Run lightweight deployment readiness checks that do not require Docker builds."""

from __future__ import annotations

import json
import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    command: list[str]


CHECKS = (
    Check("sensitive defaults", [sys.executable, "scripts/health/check_sensitive_defaults.py"]),
    Check("dockerfile copy sources", [sys.executable, "scripts/health/check_dockerfile_sources.py"]),
    Check("monitoring config", [sys.executable, "scripts/health/check_monitoring_config.py"]),
    Check("compose runtime assumptions", [sys.executable, "scripts/health/check_compose_runtime_assumptions.py"]),
    Check("runtime secret consistency script", [sys.executable, "scripts/health/check_runtime_secret_consistency.py", "--help"]),
    Check("model manifest JSON", [sys.executable, "-m", "json.tool", "models/model_manifest.json"]),
    Check("model manifest artifacts", [sys.executable, "scripts/health/check_model_report.py", "--check-artifacts"]),
    Check("model directory layout", [sys.executable, "scripts/health/check_model_layout.py"]),
    Check("grafana dashboard JSON", [sys.executable, "-m", "json.tool", "monitoring/grafana/provisioning/dashboards/cctv_overview.json"]),
    Check("alarm device config", [sys.executable, "scripts/health/check_alarm_devices.py", "--skip-network", "--allow-unconfigured"]),
    Check("python scripts compile", [sys.executable, "-m", "py_compile", "scripts/ops/evaluate_detection.py", "scripts/health/check_model_report.py", "scripts/health/check_sensitive_defaults.py", "scripts/health/check_monitoring_config.py", "scripts/health/check_compose_runtime_assumptions.py", "scripts/health/check_runtime_secret_consistency.py", "scripts/health/check_alarm_devices.py", "scripts/health/check_field_network.py", "scripts/health/check_offline_readiness.py", "scripts/health/check_public_api_fd_stability.py", "scripts/cleanup/cleanup_appearance_crop_refs.py", "scripts/smoke/smoke_test_deployment.py", "scripts/smoke/smoke_test_data_flow.py", "runners/run_sensor_rule_bridge.py"]),
    Check("parser python compile", [sys.executable, "-m", "py_compile", "parser-python/time_utils.py", "parser-python/database/models.py", "parser-python/batch/devices_batch.py", "parser-python/service/sensor_service.py", "parser-python/live_receiver.py"]),
    Check("docker compose config", ["docker", "compose", "config"]),
    Check("docker compose jetson config", ["docker", "compose", "-f", "docker-compose.jetson.yml", "config"]),
    Check("docker-build.sh syntax", ["bash", "-n", "docker-build.sh"]),
    Check("runtime cleanup scripts syntax", ["bash", "-n", "scripts/cleanup/cleanup_runtime_data.sh", "scripts/ops/rotate_alert_log.sh", "scripts/ops/install_runtime_cleanup_timer.sh"]),
    Check("runner action bridge entrypoint", [sys.executable, "runners/run_action_bridge.py", "--help"]),
    Check("runner alert api entrypoint", [sys.executable, "runners/run_alert_api.py", "--help"]),
    Check("runner edgex adapter entrypoint", [sys.executable, "runners/run_edgex_adapter.py", "--help"]),
    Check("runner kuiper rules entrypoint", [sys.executable, "runners/run_kuiper_rules.py", "--help"]),
    Check("runner public api entrypoint", [sys.executable, "runners/run_public_api.py", "--help"]),
    Check("runner sensor rule bridge entrypoint", [sys.executable, "runners/run_sensor_rule_bridge.py", "--help"]),
)

CI_CHECKS = (
    Check("sensitive defaults", [sys.executable, "scripts/health/check_sensitive_defaults.py"]),
    Check("dockerfile copy sources", [sys.executable, "scripts/health/check_dockerfile_sources.py"]),
    Check("monitoring config", [sys.executable, "scripts/health/check_monitoring_config.py"]),
    Check("runtime secret consistency script", [sys.executable, "scripts/health/check_runtime_secret_consistency.py", "--help"]),
    Check("model manifest JSON", [sys.executable, "-m", "json.tool", "models/model_manifest.json"]),
    Check("model directory layout", [sys.executable, "scripts/health/check_model_layout.py"]),
    Check("grafana dashboard JSON", [sys.executable, "-m", "json.tool", "monitoring/grafana/provisioning/dashboards/cctv_overview.json"]),
    Check("alarm device config", [sys.executable, "scripts/health/check_alarm_devices.py", "--skip-network", "--allow-unconfigured"]),
    Check("python scripts compile", [sys.executable, "-m", "py_compile", "scripts/ops/evaluate_detection.py", "scripts/health/check_model_report.py", "scripts/health/check_sensitive_defaults.py", "scripts/health/check_monitoring_config.py", "scripts/health/check_compose_runtime_assumptions.py", "scripts/health/check_runtime_secret_consistency.py", "scripts/health/check_alarm_devices.py", "scripts/health/check_field_network.py", "scripts/health/check_offline_readiness.py", "scripts/health/check_public_api_fd_stability.py", "scripts/cleanup/cleanup_appearance_crop_refs.py", "scripts/smoke/smoke_test_deployment.py", "scripts/smoke/smoke_test_data_flow.py", "runners/run_sensor_rule_bridge.py"]),
    Check("parser python compile", [sys.executable, "-m", "py_compile", "parser-python/time_utils.py", "parser-python/database/models.py", "parser-python/batch/devices_batch.py", "parser-python/service/sensor_service.py", "parser-python/live_receiver.py"]),
    Check("docker compose config", ["docker", "compose", "config"]),
    Check("docker compose jetson config", ["docker", "compose", "--env-file", ".env.jetson.example", "-f", "docker-compose.jetson.yml", "config"]),
    Check("docker-build.sh syntax", ["bash", "-n", "docker-build.sh"]),
    Check("runtime cleanup scripts syntax", ["bash", "-n", "scripts/cleanup/cleanup_runtime_data.sh", "scripts/ops/rotate_alert_log.sh", "scripts/ops/install_runtime_cleanup_timer.sh"]),
)


def _run_check(check: Check) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            check.command,
            cwd=Path(__file__).resolve().parents[2],
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


def _parse_args() -> object:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run only checks that are valid on a clean GitHub Actions runner.",
    )
    parser.add_argument(
        "--require-model-quality",
        action="store_true",
        help="Also enforce the fixed fall/non-fall model quality gate.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checks = CI_CHECKS if args.ci else CHECKS
    if args.require_model_quality:
        checks = checks + (
            Check(
                "fall model quality gate",
                [
                    sys.executable,
                    "scripts/health/check_fall_quality_gate.py",
                    "--fall",
                    "data/fall_eval/pilot_rtsp_input_split_fall3.jsonl",
                    "--nonfall",
                    "data/fall_eval/pilot_rtsp_input_split_notfall3.jsonl",
                ],
            ),
        )
    results = []
    failed = False
    for check in checks:
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
