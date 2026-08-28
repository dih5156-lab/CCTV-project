#!/usr/bin/env python3
"""Run a one-shot operational health check for fall Shadow inference."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CONTAINER = "cctv-ai-engine"
DEFAULT_REVIEW_LOG = Path("data/fall_dataset/annotations/review.jsonl")
DEFAULT_API_URL = "http://localhost:8000/health"
RUNTIME_FAILURE_STATUSES = {"error", "missing_dependency", "no_frames"}
STATS_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?) .*DeepStream stats:.*?frames=(?P<frames>\d+)"
    r".*?frame_dropped=(?P<dropped>\d+).*?failed=(?P<failed>\d+)",
    re.MULTILINE,
)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def analyze_shadow_records(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    max_age_seconds: float,
    expected_threshold: float,
    window_seconds: float | None = None,
) -> dict[str, Any]:
    shadow_rows = []
    for row in rows:
        if row.get("event_type") != "fall_shadow_window":
            continue
        created_at = _parse_datetime(row.get("created_at"))
        if window_seconds is not None and created_at is not None:
            age_seconds = (now.astimezone(timezone.utc) - created_at).total_seconds()
            if age_seconds > window_seconds:
                continue
        shadow_rows.append(row)
    statuses: Counter[str] = Counter()
    modes: set[str] = set()
    thresholds: set[float] = set()
    latest_at: datetime | None = None
    publish_pending_count = 0
    parse_timestamp_errors = 0

    for row in shadow_rows:
        aux = row.get("falldata_aux")
        if not isinstance(aux, dict):
            aux = {}
        statuses[str(aux.get("status") or "missing")] += 1
        modes.add(str(aux.get("mode") or "missing"))
        threshold = aux.get("threshold")
        if isinstance(threshold, (int, float)):
            thresholds.add(float(threshold))
        if row.get("falldata_aux_publish_pending") is True:
            publish_pending_count += 1
        created_at = _parse_datetime(row.get("created_at"))
        if created_at is None:
            parse_timestamp_errors += 1
        elif latest_at is None or created_at > latest_at:
            latest_at = created_at

    latest_age_seconds = (
        max(0.0, (now.astimezone(timezone.utc) - latest_at).total_seconds())
        if latest_at
        else None
    )
    runtime_failure_count = sum(statuses[status] for status in RUNTIME_FAILURE_STATUSES)
    failures: list[str] = []
    if not shadow_rows:
        failures.append("no_shadow_records")
    if latest_age_seconds is None or latest_age_seconds > max_age_seconds:
        failures.append("stale_shadow_records")
    if modes and modes != {"shadow"}:
        failures.append("non_shadow_mode")
    if thresholds and any(
        abs(threshold - expected_threshold) > 1e-9 for threshold in thresholds
    ):
        failures.append("threshold_mismatch")
    if runtime_failure_count:
        failures.append("runtime_failures")
    if publish_pending_count:
        failures.append("publish_pending_in_shadow")
    if parse_timestamp_errors:
        failures.append("invalid_shadow_timestamps")

    return {
        "passed": not failures,
        "records": len(shadow_rows),
        "status_counts": dict(sorted(statuses.items())),
        "modes": sorted(modes),
        "thresholds": sorted(thresholds),
        "latest_at": latest_at.isoformat() if latest_at else None,
        "latest_age_seconds": latest_age_seconds,
        "runtime_failure_count": runtime_failure_count,
        "publish_pending_count": publish_pending_count,
        "invalid_timestamp_count": parse_timestamp_errors,
        "failures": failures,
    }


def analyze_deepstream_stats(
    logs: str, *, now: datetime, max_age_seconds: float
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for match in STATS_PATTERN.finditer(logs):
        timestamp = _parse_datetime(match.group("timestamp"))
        if timestamp is None:
            continue
        samples.append(
            {
                "timestamp": timestamp,
                "frames": int(match.group("frames")),
                "frame_dropped": int(match.group("dropped")),
                "failed": int(match.group("failed")),
            }
        )

    failures: list[str] = []
    latest = samples[-1] if samples else None
    latest_age_seconds = (
        max(0.0, (now.astimezone(timezone.utc) - latest["timestamp"]).total_seconds())
        if latest
        else None
    )
    frame_progress = samples[-1]["frames"] - samples[0]["frames"] if len(samples) >= 2 else 0
    if not samples:
        failures.append("no_deepstream_stats")
    if latest_age_seconds is None or latest_age_seconds > max_age_seconds:
        failures.append("stale_deepstream_stats")
    if len(samples) < 2 or frame_progress <= 0:
        failures.append("no_frame_progress")
    if latest and latest["failed"] > 0:
        failures.append("deepstream_failures")

    return {
        "passed": not failures,
        "samples": len(samples),
        "latest_frames": latest["frames"] if latest else None,
        "frame_progress": frame_progress,
        "latest_frame_dropped": latest["frame_dropped"] if latest else None,
        "latest_failed": latest["failed"] if latest else None,
        "latest_age_seconds": latest_age_seconds,
        "failures": failures,
    }


def _read_recent_jsonl(path: Path, max_lines: int) -> tuple[list[dict[str, Any]], int]:
    recent_lines: deque[str] = deque(maxlen=max_lines)
    parse_errors = 0
    with path.open("r", encoding="utf-8") as file_pointer:
        recent_lines.extend(file_pointer)
    rows = []
    for line in recent_lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            parse_errors += 1
    return rows, parse_errors


def _run(command: list[str], *, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )


def _docker_prefix() -> list[str]:
    direct = _run(["docker", "ps"], timeout=5.0)
    if direct.returncode == 0:
        return ["docker"]
    sudo = _run(["sudo", "-n", "docker", "ps"], timeout=5.0)
    if sudo.returncode == 0:
        return ["sudo", "-n", "docker"]
    raise RuntimeError("Docker 접근 실패: docker 권한 또는 비대화형 sudo를 확인하세요")


def _container_check(docker: list[str], container: str) -> dict[str, Any]:
    command = [
        *docker,
        "inspect",
        container,
        "--format",
        "{{json .}}",
    ]
    result = _run(command)
    if result.returncode != 0:
        return {"passed": False, "error": result.stderr.strip() or result.stdout.strip()}
    inspect_data = json.loads(result.stdout)
    state = inspect_data.get("State", {})
    health = (state.get("Health") or {}).get("Status")
    failures = []
    if state.get("Status") != "running":
        failures.append("container_not_running")
    if health not in {None, "healthy"}:
        failures.append("container_unhealthy")
    restart_count = int(inspect_data.get("RestartCount") or 0)
    if restart_count > 0:
        failures.append("container_restarted")
    if state.get("OOMKilled") is True:
        failures.append("container_oom_killed")
    return {
        "passed": not failures,
        "status": state.get("Status"),
        "health": health,
        "restart_count": restart_count,
        "oom_killed": bool(state.get("OOMKilled")),
        "started_at": state.get("StartedAt"),
        "failures": failures,
    }


def _resource_check(docker: list[str], container: str) -> dict[str, Any]:
    result = _run(
        [
            *docker,
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            container,
        ]
    )
    if result.returncode != 0:
        return {"passed": False, "error": result.stderr.strip() or result.stdout.strip()}
    payload = json.loads(result.stdout)
    return {
        "passed": True,
        "cpu_percent": payload.get("CPUPerc"),
        "memory_usage": payload.get("MemUsage"),
        "memory_percent": payload.get("MemPerc"),
        "pids": payload.get("PIDs"),
    }


def _runtime_env_check(docker: list[str], container: str) -> dict[str, Any]:
    result = _run([*docker, "exec", container, "env"])
    if result.returncode != 0:
        return {"passed": False, "error": result.stderr.strip() or result.stdout.strip()}
    env = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    model_path = env.get("FALLDATA_AUX_COMPARE_MODEL_PATH", "")
    threshold_raw = env.get("FALLDATA_AUX_COMPARE_THRESHOLD", "")
    mode = env.get("FALLDATA_AUX_MODE", "")
    veto = env.get("FALLDATA_AUX_COMPARE_VETO_ENABLED", "").lower()
    failures = []
    if mode != "shadow":
        failures.append("mode_not_shadow")
    if veto not in {"false", "0", "no", "off"}:
        failures.append("compare_veto_enabled")
    if not model_path:
        failures.append("compare_model_missing")
    try:
        threshold = float(threshold_raw)
    except ValueError:
        threshold = None
        failures.append("invalid_compare_threshold")
    return {
        "passed": not failures,
        "mode": mode,
        "compare_veto_enabled": veto,
        "compare_model_path": model_path,
        "compare_threshold": threshold,
        "failures": failures,
    }


def _api_check(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except Exception as exc:
        return {"passed": False, "url": url, "error": str(exc)}
    return {"passed": 200 <= status < 300, "url": url, "status": status, "body": body}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--review-log", type=Path, default=DEFAULT_REVIEW_LOG)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--max-shadow-age-seconds", type=float, default=120.0)
    parser.add_argument("--shadow-window-seconds", type=float, default=600.0)
    parser.add_argument("--max-stats-age-seconds", type=float, default=30.0)
    parser.add_argument("--log-since", default="2m")
    parser.add_argument("--max-review-lines", type=int, default=500)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_review_lines < 1:
        raise SystemExit("--max-review-lines must be positive")
    now = datetime.now(timezone.utc)
    try:
        docker = _docker_prefix()
    except RuntimeError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    container = _container_check(docker, args.container)
    resources = _resource_check(docker, args.container)
    runtime_env = _runtime_env_check(docker, args.container)
    logs_result = _run([*docker, "logs", "--since", args.log_since, args.container])
    deepstream = (
        analyze_deepstream_stats(
            logs_result.stdout + logs_result.stderr,
            now=now,
            max_age_seconds=args.max_stats_age_seconds,
        )
        if logs_result.returncode == 0
        else {"passed": False, "error": logs_result.stderr.strip()}
    )

    if args.review_log.exists():
        rows, parse_errors = _read_recent_jsonl(args.review_log, args.max_review_lines)
        threshold = runtime_env.get("compare_threshold")
        shadow = (
            analyze_shadow_records(
                rows,
                now=now,
                max_age_seconds=args.max_shadow_age_seconds,
                expected_threshold=float(threshold),
                window_seconds=args.shadow_window_seconds,
            )
            if isinstance(threshold, (int, float))
            else {"passed": False, "error": "runtime threshold unavailable"}
        )
        shadow["json_parse_errors"] = parse_errors
        if parse_errors:
            shadow["passed"] = False
    else:
        shadow = {"passed": False, "error": f"review log not found: {args.review_log}"}

    api = _api_check(args.api_url, args.http_timeout)
    checks = {
        "container": container,
        "resources": resources,
        "runtime_env": runtime_env,
        "deepstream": deepstream,
        "shadow": shadow,
        "api": api,
    }
    passed = all(check.get("passed") is True for check in checks.values())
    payload = {
        "passed": passed,
        "checked_at": now.isoformat(),
        "container_name": args.container,
        "review_log": str(args.review_log),
        "checks": checks,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
