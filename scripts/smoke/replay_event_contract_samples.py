#!/usr/bin/env python3
"""이벤트 계약 샘플을 검증하고 선택적으로 MQTT에 재생한다."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# `python scripts/smoke/replay_event_contract_samples.py` 실행 시 프로젝트 루트를 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_event_contracts import _sample_payloads, validate_payload  # noqa: E402


CRITICAL_TYPES = {"fall_detected", "danger_zone", "intrusion", "unsafe_behavior"}


def replay_samples(
    *,
    broker: str = "localhost",
    port: int = 1883,
    publish: bool = False,
    include_critical: bool = False,
) -> dict[str, Any]:
    payloads = _sample_payloads()
    results: list[dict[str, Any]] = []
    client = None

    if publish:
        import paho.mqtt.client as mqtt

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="event-contract-replay")
        client.connect(broker, port, 5)

    try:
        for index, payload in enumerate(payloads):
            result = validate_payload(payload, index=index)
            event_type = result.get("event_type")
            item: dict[str, Any] = {
                "event_type": event_type,
                "valid": result["valid"],
                "warnings": result["warnings"],
                "published": False,
            }
            if result["valid"] and publish and client is not None:
                if event_type in CRITICAL_TYPES and not include_critical:
                    item["skipped"] = "critical event; use --include-critical to publish"
                else:
                    topic = f"cctv/ai/events/contract-test/{event_type}"
                    message = dict(payload)
                    message["timestamp"] = time.time()
                    client.publish(topic, json.dumps(message, ensure_ascii=False), qos=0).wait_for_publish()
                    item["published"] = True
                    item["topic"] = topic
            if not result["valid"]:
                item["errors"] = result["errors"]
            results.append(item)
    finally:
        if client is not None:
            client.disconnect()

    return {
        "valid": all(item["valid"] for item in results),
        "published": sum(item["published"] for item in results),
        "total": len(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="이벤트 계약 샘플 검증/재생")
    parser.add_argument("--publish", action="store_true", help="실제 MQTT 브로커에 발행")
    parser.add_argument("--include-critical", action="store_true", help="낙상·침입 등 critical 이벤트도 발행")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()
    report = replay_samples(
        broker=args.broker,
        port=args.port,
        publish=args.publish,
        include_critical=args.include_critical,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
