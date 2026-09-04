#!/usr/bin/env python3
"""세 출력 장치 Device Service의 HTTP 현장 UAT를 실행한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_cases() -> list[dict[str, Any]]:
    """장치별 health 경로와 안전한 테스트 명령을 구성한다."""
    return [
        {
            "device": "speaker",
            "host": os.environ.get("SPEAKER_SERVICE_HOST", "127.0.0.1"),
            "port": int(os.environ.get("SPEAKER_SERVICE_PORT", "59991")),
            "path": "/api/v3/device/name/cctv-speaker-01/play",
            "payload": {"event_id": "uat-speaker", "text": "EdgeX UAT 점검"},
        },
        {
            "device": "siren",
            "host": os.environ.get("SIREN_SERVICE_HOST", "127.0.0.1"),
            "port": int(os.environ.get("SIREN_SERVICE_PORT", "59992")),
            "path": "/api/v3/device/name/cctv-siren-01/trigger",
            "payload": {
                "event_id": "uat-siren",
                "event_type": "uat",
                "camera_id": "uat-camera",
            },
        },
        {
            "device": "signboard",
            "host": os.environ.get("SIGNBOARD_SERVICE_HOST", "127.0.0.1"),
            "port": int(os.environ.get("SIGNBOARD_SERVICE_PORT", "59993")),
            "path": "/api/v3/device/name/cctv-signboard-01/display",
            "payload": {"event_id": "uat-signboard", "text": "EdgeX UAT 점검"},
        },
    ]


def request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """HTTP JSON 요청을 보내고 상태 코드와 응답 본문을 반환한다."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "X-Command-Id": "uat-command"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"{}")
    except HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except (json.JSONDecodeError, OSError):
            return exc.code, {"error_code": "invalid_response"}
    except (URLError, OSError) as exc:
        return 0, {"error_code": "connection_failed", "detail": str(exc)}


def run_uat(mode: str, confirm_physical_control: bool = False) -> list[dict[str, Any]]:
    """세 서비스의 상태와 명령 응답을 점검하고 결과를 반환한다."""
    if mode == "real" and not confirm_physical_control:
        raise ValueError("실제 장치 제어는 --confirm-physical-control 옵션이 필요합니다")

    results = []
    expected_status = "simulated" if mode == "dry-run" else "acknowledged"
    for case in build_cases():
        base_url = f"http://{case['host']}:{case['port']}"
        health_code, health_body = request_json(f"{base_url}/health")
        command_code, command_body = request_json(
            f"{base_url}{case['path']}", "PUT", case["payload"]
        )
        results.append(
            {
                "device": case["device"],
                "health_code": health_code,
                "health_status": health_body.get("status"),
                "command_code": command_code,
                "command_status": command_body.get("status"),
                "expected_status": expected_status,
                "passed": (
                    health_code == 200
                    and health_body.get("status") == "up"
                    and command_code == 200
                    and command_body.get("status") == expected_status
                ),
            }
        )
    return results


def main() -> int:
    """명령행 UAT를 실행하고 실패한 장치가 있으면 종료 코드 1을 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "real"), default="dry-run")
    parser.add_argument("--confirm-physical-control", action="store_true")
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    args = parser.parse_args()
    try:
        results = run_uat(args.mode, args.confirm_physical_control)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(
                f"{result['device']}: "
                f"health={result['health_code']} command={result['command_code']} "
                f"status={result['command_status']} "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
