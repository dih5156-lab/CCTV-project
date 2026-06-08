"""Run post-deploy smoke checks for the local CCTV Docker stack."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpCheck:
    name: str
    url: str
    expected_statuses: tuple[int, ...] = (200,)
    required_text: str | None = None
    forbidden_texts: tuple[str, ...] = ()


def _read_url(url: str, timeout: float) -> tuple[bool, int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return True, int(response.status), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, int(exc.code), body
    except Exception as exc:
        return False, None, str(exc)


def run_http_check(check: HttpCheck, timeout: float) -> dict[str, Any]:
    ok, status, body = _read_url(check.url, timeout)
    passed = ok and status in check.expected_statuses
    detail = ""
    if check.required_text is not None and check.required_text not in body:
        passed = False
        detail = f"missing required text: {check.required_text}"

    found_forbidden = [text for text in check.forbidden_texts if text in body]
    if found_forbidden:
        passed = False
        detail = f"forbidden text found: {', '.join(found_forbidden)}"

    if not passed and not detail:
        detail = body[:500]

    return {
        "name": check.name,
        "url": check.url,
        "passed": passed,
        "status": status,
        "detail": "" if passed else detail,
    }


def check_prometheus_targets(prometheus_url: str, timeout: float) -> dict[str, Any]:
    url = f"{prometheus_url.rstrip('/')}/api/v1/targets"
    ok, status, body = _read_url(url, timeout)
    if not ok or status != 200:
        return {
            "name": "prometheus scrape targets",
            "url": url,
            "passed": False,
            "status": status,
            "detail": body[:500],
        }

    try:
        payload = json.loads(body)
        active_targets = payload.get("data", {}).get("activeTargets", [])
    except json.JSONDecodeError as exc:
        return {
            "name": "prometheus scrape targets",
            "url": url,
            "passed": False,
            "status": status,
            "detail": f"invalid JSON: {exc}",
        }

    required_jobs = {"cctv-action-layer", "cctv-public-api"}
    unhealthy: list[str] = []
    found: set[str] = set()
    for target in active_targets:
        labels = target.get("labels", {})
        job = labels.get("job")
        if job not in required_jobs:
            continue
        found.add(job)
        if target.get("health") != "up":
            unhealthy.append(f"{job}: {target.get('lastError') or target.get('health')}")

    missing = sorted(required_jobs - found)
    passed = not missing and not unhealthy
    detail_parts = []
    if missing:
        detail_parts.append(f"missing jobs: {', '.join(missing)}")
    if unhealthy:
        detail_parts.append(f"unhealthy jobs: {', '.join(unhealthy)}")

    return {
        "name": "prometheus scrape targets",
        "url": url,
        "passed": passed,
        "status": status,
        "detail": "; ".join(detail_parts),
    }


def build_checks(host: str, include_monitoring: bool = False) -> list[HttpCheck]:
    checks = [
        HttpCheck("alert api health", f"http://{host}:8000/health", required_text="cctv-alert-api"),
        HttpCheck("action layer health", f"http://{host}:8080/health", required_text="cctv-action-layer"),
        HttpCheck("public api health", f"http://{host}:9000/api/v1/health"),
        HttpCheck("public api readiness", f"http://{host}:9000/api/v1/readiness", required_text="ready"),
        HttpCheck(
            "public api docs",
            f"http://{host}:9000/docs",
            required_text="CCTV Platform API",
            forbidden_texts=("cdn.jsdelivr.net", "unpkg.com"),
        ),
        HttpCheck("public api openapi schema", f"http://{host}:9000/openapi.json", required_text="CCTV Platform API"),
    ]
    if include_monitoring:
        checks.extend(
            [
                HttpCheck("prometheus readiness", f"http://{host}:9090/-/ready", required_text="Prometheus Server is Ready"),
                HttpCheck("grafana health", f"http://{host}:3001/api/health"),
            ]
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running CCTV Docker deployment.")
    parser.add_argument("--host", default="localhost", help="Published Docker host address.")
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--include-monitoring",
        action="store_true",
        help="Also check optional Prometheus and Grafana services.",
    )
    parser.add_argument(
        "--skip-prometheus-targets",
        action="store_true",
        help="Compatibility option. Prometheus target checks now run only with --include-monitoring.",
    )
    args = parser.parse_args()

    results = [
        run_http_check(check, args.timeout)
        for check in build_checks(args.host, include_monitoring=args.include_monitoring)
    ]
    if args.include_monitoring and not args.skip_prometheus_targets:
        results.append(check_prometheus_targets(f"http://{args.host}:9090", args.timeout))

    passed = all(result["passed"] for result in results)
    print(json.dumps({"passed": passed, "checks": results}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
