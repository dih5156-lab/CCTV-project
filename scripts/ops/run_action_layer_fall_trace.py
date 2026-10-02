#!/usr/bin/env python3
"""Action Layer에 합성 낙상 이벤트를 보내고 명령 이력을 추적한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


def build_fall_payload(*, camera_id: str, event_id: str, request_id: str) -> dict[str, Any]:
    """실제 영상 입력과 구분되는 합성 낙상 이벤트 payload를 만든다."""
    return {
        "camera_id": camera_id,
        "type": "fall_detected",
        "event_type": "fall_detected",
        "severity": "critical",
        "confidence": 0.99,
        "event_id": event_id,
        "request_id": request_id,
        "metadata": {"source": "synthetic_uat", "test": True},
    }


def _request_json(
    *,
    url: str,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> tuple[int, dict[str, Any]]:
    """내부 토큰을 포함한 JSON HTTP 요청을 보내고 응답을 반환한다."""
    body = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Internal-Token": token,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except (json.JSONDecodeError, OSError):
            return exc.code, {"error": "invalid_response"}
    except (urllib.error.URLError, OSError) as exc:
        return 0, {"error": "connection_failed", "detail": str(exc)}


def trace_fall_event(
    *,
    base_url: str,
    token: str,
    camera_id: str,
    event_id: str,
    request_id: str,
    send: bool,
    wait_seconds: float,
) -> dict[str, Any]:
    """합성 낙상 이벤트를 전송하고 동일 event_id의 명령 이력을 찾는다."""
    event_payload = build_fall_payload(
        camera_id=camera_id,
        event_id=event_id,
        request_id=request_id,
    )
    result: dict[str, Any] = {
        "event_id": event_id,
        "request_id": request_id,
        "camera_id": camera_id,
        "synthetic": True,
        "sent": False,
        "event_payload": event_payload,
        "commands": [],
    }
    if not send:
        result["status"] = "preview"
        return result

    code, response = _request_json(
        url=f"{base_url.rstrip('/')}/events",
        token=token,
        method="POST",
        payload=event_payload,
    )
    result.update({"sent": code == 200, "event_response_code": code, "event_response": response})
    if code != 200:
        result["status"] = "event_rejected"
        return result

    deadline = time.monotonic() + max(0.0, wait_seconds)
    while True:
        command_code, command_response = _request_json(
            url=f"{base_url.rstrip('/')}/commands?limit=50",
            token=token,
        )
        commands = command_response if isinstance(command_response, list) else []
        matching = [
            command
            for command in commands
            if isinstance(command, dict)
            and str(command.get("command_id", "")).startswith(f"{event_id}:")
        ]
        if matching or time.monotonic() >= deadline:
            result.update(
                {
                    "commands_response_code": command_code,
                    "commands": matching,
                    "status": "traced" if matching else "no_command_found",
                }
            )
            return result
        time.sleep(0.2)


def main() -> int:
    """명령행 옵션을 읽어 낙상 추적 UAT를 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("ACTION_LAYER_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--token", default=os.environ.get("INTERNAL_SERVICE_TOKEN", ""))
    parser.add_argument("--camera-id", default="cam-01")
    parser.add_argument("--event-id", default=f"fall-trace-{uuid.uuid4().hex[:12]}")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--send", action="store_true", help="실제 Action Layer에 이벤트를 전송")
    parser.add_argument(
        "--confirm-physical-control",
        action="store_true",
        help="실제 출력 장치 제어를 명시적으로 승인",
    )
    parser.add_argument("--wait-seconds", type=float, default=5.0)
    args = parser.parse_args()
    request_id = args.request_id or f"{args.event_id}:input"
    if args.send and not args.confirm_physical_control:
        parser.error("실제 전송에는 --confirm-physical-control 옵션이 필요합니다")
    if args.send and not args.token:
        parser.error("실제 전송에는 INTERNAL_SERVICE_TOKEN 또는 --token이 필요합니다")
    report = trace_fall_event(
        base_url=args.base_url,
        token=args.token,
        camera_id=args.camera_id,
        event_id=args.event_id,
        request_id=request_id,
        send=args.send,
        wait_seconds=args.wait_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"preview", "traced"} else 1


if __name__ == "__main__":
    sys.exit(main())
