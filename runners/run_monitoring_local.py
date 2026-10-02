"""Explicit local POC entry point; only a Unix MQTT socket is supported."""

import argparse
import json
import math
import signal
import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.monitoring_service import build_local_service


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("positive finite number required")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--receiver-db", required=True, type=Path)
    parser.add_argument("--consumer-id", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--mqtt-socket", required=True, type=Path)
    parser.add_argument("--namespace", action="append", required=True)
    parser.add_argument("--interval", type=positive, default=1)
    parser.add_argument("--timeout", type=positive, default=5)
    parser.add_argument("--lease-seconds", type=positive, default=30)
    parser.add_argument("--ack-wait", type=positive, default=10)
    parser.add_argument("--run-seconds", type=positive)
    parser.add_argument("--clock-sync-confirmed", action="store_true")
    args = parser.parse_args()
    if not args.mqtt_socket.is_socket():
        parser.error("--mqtt-socket must point to an existing local Unix socket")
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    service, source, receiver, binding, timings, commits = build_local_service(
        args.source_db,
        args.receiver_db,
        args.consumer_id,
        args.destination,
        args.mqtt_socket,
        namespaces=args.namespace,
        interval=args.interval,
        timeout=args.timeout,
        lease_seconds=args.lease_seconds,
        ack_wait=args.ack_wait,
        clock_sync="confirmed" if args.clock_sync_confirmed else "unknown",
    )
    try:
        service.start()
        print(
            json.dumps(
                {
                    "event": "started",
                    "origin_id": binding.origin_id,
                    "consumer_id": binding.consumer_id,
                }
            ),
            flush=True,
        )
        stopped.wait(args.run_seconds)
    finally:
        service.close(timeout=max(15, 2 * args.timeout + 5))
    print(
        json.dumps(
            {
                "event": "stopped",
                "service": service.status(),
                "receiver": service.receiver.status(),
                "timings": timings.report(),
                "commits": commits.report(),
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
