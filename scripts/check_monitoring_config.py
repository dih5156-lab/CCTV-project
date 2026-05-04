"""Validate the Prometheus/Grafana wiring used by the CCTV stack."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMETHEUS_CONFIG = ROOT / "monitoring/prometheus.yml"
GRAFANA_DATASOURCE = ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml"
GRAFANA_DASHBOARD = ROOT / "monitoring/grafana/provisioning/dashboards/cctv_overview.json"

REQUIRED_PROMETHEUS_JOBS = {
    "cctv-action-layer": {
        "target": "cctv-action-layer:8080",
        "metrics_path": "/metrics",
    },
    "cctv-public-api": {
        "target": "cctv-public-api:9000",
        "metrics_path": "/api/v1/metrics",
    },
}

REQUIRED_DASHBOARD_EXPRESSIONS = (
    'up{job="cctv-action-layer"}',
    'up{job="cctv-public-api"}',
    "cctv_action_bridge_up",
    "sum(cctv_mqtt_events_received_total)",
    "sum(cctv_events_handled_total)",
    "cctv_pending_events",
    "rate(cctv_public_api_http_requests_total[1m])",
)


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"').strip("'")


def parse_prometheus_jobs(text: str) -> dict[str, dict[str, object]]:
    jobs: dict[str, dict[str, object]] = {}
    current_job: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        job_match = re.match(r"-\s+job_name:\s*(.+)$", line)
        if job_match:
            current_job = _strip_quotes(job_match.group(1))
            jobs[current_job] = {"targets": [], "metrics_path": ""}
            continue

        if current_job is None:
            continue

        if line.startswith("metrics_path:"):
            jobs[current_job]["metrics_path"] = _strip_quotes(line.split(":", 1)[1])
            continue

        if "targets:" in line:
            jobs[current_job]["targets"] = re.findall(r'"([^"]+)"', line)

    return jobs


def collect_dashboard_expressions(dashboard: dict[str, object]) -> set[str]:
    expressions: set[str] = set()
    for panel in dashboard.get("panels", []):
        if not isinstance(panel, dict):
            continue
        for target in panel.get("targets", []):
            if isinstance(target, dict) and isinstance(target.get("expr"), str):
                expressions.add(target["expr"])
    return expressions


def find_monitoring_config_issues() -> list[str]:
    issues: list[str] = []
    jobs = parse_prometheus_jobs(PROMETHEUS_CONFIG.read_text(encoding="utf-8"))

    for job_name, expected in REQUIRED_PROMETHEUS_JOBS.items():
        job = jobs.get(job_name)
        if not job:
            issues.append(f"missing prometheus job: {job_name}")
            continue
        if expected["target"] not in job.get("targets", []):
            issues.append(f"prometheus job {job_name} missing target {expected['target']}")
        if job.get("metrics_path") != expected["metrics_path"]:
            issues.append(f"prometheus job {job_name} has wrong metrics_path")

    datasource_text = GRAFANA_DATASOURCE.read_text(encoding="utf-8")
    if "url: http://prometheus:9090" not in datasource_text:
        issues.append("grafana datasource must point to http://prometheus:9090")

    dashboard = json.loads(GRAFANA_DASHBOARD.read_text(encoding="utf-8"))
    expressions = collect_dashboard_expressions(dashboard)
    for expr in REQUIRED_DASHBOARD_EXPRESSIONS:
        if expr not in expressions:
            issues.append(f"grafana dashboard missing expression: {expr}")

    return issues


def main() -> int:
    issues = find_monitoring_config_issues()
    if issues:
        print(json.dumps({"passed": False, "issues": issues}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"passed": True, "issues": []}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
