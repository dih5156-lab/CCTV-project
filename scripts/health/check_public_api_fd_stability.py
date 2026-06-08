#!/usr/bin/env python3
"""Check that repeated Public API readiness calls do not leak file descriptors."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


def _read_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(str(exc)) from exc


def _sample_fd_usage(url: str, timeout: float) -> dict[str, int | None]:
    payload = _read_json(url, timeout)
    data = payload.get("data", {})
    if payload.get("success") is not True or data.get("status") != "ready":
        raise RuntimeError(f"readiness is not ready: {json.dumps(payload, ensure_ascii=False)[:500]}")

    fd_usage = data.get("resources", {}).get("file_descriptors", {})
    open_fds = fd_usage.get("open")
    if not isinstance(open_fds, int):
        raise RuntimeError(f"readiness does not expose an integer FD count: {fd_usage}")
    soft_limit = fd_usage.get("soft_limit")
    if soft_limit is not None and not isinstance(soft_limit, int):
        raise RuntimeError(f"readiness does not expose an integer FD soft limit: {fd_usage}")
    return {
        "open": open_fds,
        "soft_limit": soft_limit,
    }


def check_fd_stability(
    url: str,
    *,
    samples: int,
    interval: float,
    timeout: float,
    max_growth: int,
    max_open: int | None,
) -> dict[str, Any]:
    fd_counts: list[int] = []
    soft_limits: list[int] = []
    error = ""
    for sample in range(samples):
        try:
            fd_usage = _sample_fd_usage(url, timeout)
        except RuntimeError as exc:
            error = str(exc)
            break
        fd_counts.append(fd_usage["open"])
        if fd_usage["soft_limit"] is not None:
            soft_limits.append(fd_usage["soft_limit"])
        if sample + 1 < samples:
            time.sleep(interval)

    growth = fd_counts[-1] - fd_counts[0] if len(fd_counts) >= 2 else None
    peak = max(fd_counts) if fd_counts else None
    inferred_max_open = min(soft_limits) if soft_limits else None
    effective_max_open = max_open if max_open is not None else inferred_max_open
    passed = (
        not error
        and len(fd_counts) == samples
        and growth is not None
        and growth <= max_growth
        and peak is not None
        and (effective_max_open is None or peak <= effective_max_open)
    )
    if not error and growth is not None and growth > max_growth:
        error = f"FD growth exceeded limit: {growth} > {max_growth}"
    if not error and effective_max_open is not None and peak is not None and peak > effective_max_open:
        error = f"FD count exceeded limit: {peak} > {effective_max_open}"
    return {
        "passed": passed,
        "url": url,
        "samples_requested": samples,
        "samples_collected": len(fd_counts),
        "fd_counts": fd_counts,
        "growth": growth,
        "max_growth": max_growth,
        "peak": peak,
        "max_open": effective_max_open,
        "soft_limit_samples": soft_limits,
        "detail": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Public API FD stability with repeated readiness calls.")
    parser.add_argument("--url", default="http://localhost:9000/api/v1/readiness")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--max-growth", type=int, default=32)
    parser.add_argument("--max-open", type=int, default=None)
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.interval < 0:
        parser.error("--interval must be zero or greater")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.max_growth < 0:
        parser.error("--max-growth must be zero or greater")
    if args.max_open is not None and args.max_open <= 0:
        parser.error("--max-open must be greater than zero")

    result = check_fd_stability(
        args.url,
        samples=args.samples,
        interval=args.interval,
        timeout=args.timeout,
        max_growth=args.max_growth,
        max_open=args.max_open,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
