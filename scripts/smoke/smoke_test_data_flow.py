"""배포 후 CCTV 스택의 데이터 흐름을 점검하는 smoke 검사를 수행한다."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestCheck:
    name: str
    method: str
    url: str
    expected_statuses: tuple[int, ...]
    payload: dict[str, Any] | None = None
    required_text: str | None = None


def _request(
    method: str,
    url: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[bool, int | None, str]:
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return True, int(response.status), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, int(exc.code), body
    except Exception as exc:
        return False, None, str(exc)


def run_request_check(
    check: RequestCheck,
    timeout: float,
    headers: dict[str, str] | None = None,
    retries: int = 1,
    retry_delay: float = 0.0,
) -> dict[str, Any]:
    last_result: dict[str, Any] | None = None

    for attempt in range(1, max(1, retries) + 1):
        ok, status, body = _request(
            check.method,
            check.url,
            timeout,
            payload=check.payload,
            headers=headers,
        )
        passed = ok and status in check.expected_statuses
        if check.required_text is not None:
            passed = passed and check.required_text in body

        last_result = {
            "name": check.name,
            "method": check.method,
            "url": check.url,
            "passed": passed,
            "status": status,
            "detail": "" if passed else body[:500],
            "attempts": attempt,
        }
        if passed:
            return last_result
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    assert last_result is not None
    return last_result


def build_checks(host: str) -> list[RequestCheck]:
    timestamp = time.time()
    alert_payload = {
        "camera_id": "smoke-cam-01",
        "type": "smoke_test_alert",
        "severity": "low",
        "confidence": 0.99,
        "timestamp": timestamp,
    }
    sensor_payload = {
        "device_id": "smoke-sensor-01",
        "table": "t34957",
        "data": {
            "temperature": 25.5,
            "angle_x": 1.2,
            "angle_y": 0.3,
            "event_code": 0,
        },
        "received_at": int(timestamp * 1000),
    }
    action_payload = {
        "camera_id": "smoke-cam-01",
        "type": "helmet",
        "severity": "low",
        "confidence": 0.99,
        "timestamp": timestamp,
    }

    return [
        RequestCheck(
            "alert api accepts alert",
            "POST",
            f"http://{host}:8000/api/alerts",
            (202,),
            payload=alert_payload,
            required_text="accepted",
        ),
        RequestCheck(
            "alert api accepts sensor reading",
            "POST",
            f"http://{host}:8000/api/sensor-readings",
            (202,),
            payload=sensor_payload,
            required_text="accepted",
        ),
        RequestCheck(
            "action layer accepts event",
            "POST",
            f"http://{host}:8080/events",
            (200,),
            payload=action_payload,
            required_text="ok",
        ),
        RequestCheck(
            "action layer metrics expose handled events",
            "GET",
            f"http://{host}:8080/metrics",
            (200,),
            required_text="cctv_events_handled_total",
        ),
        RequestCheck(
            "public api metrics endpoint",
            "GET",
            f"http://{host}:9000/api/v1/metrics",
            (200,),
            required_text="cctv_public_api_http_requests_total",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="CCTV 실행 데이터 경로를 smoke 검사한다.")
    parser.add_argument("--host", default="localhost", help="Docker 포트가 공개된 호스트 주소.")
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP 요청 제한 시간(초).")
    parser.add_argument("--retries", type=int, default=1, help="실패한 검사별 재시도 횟수.")
    parser.add_argument("--retry-delay", type=float, default=0.0, help="재시도 사이 대기 시간(초).")
    parser.add_argument(
        "--internal-token",
        default=os.environ.get("INTERNAL_SERVICE_TOKEN", ""),
        help="Action Layer REST 엔드포인트에 전달할 선택적 X-Internal-Token.",
    )
    args = parser.parse_args()

    headers = {}
    if args.internal_token:
        headers["X-Internal-Token"] = args.internal_token

    results = [
        run_request_check(
            check,
            args.timeout,
            headers=headers,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
        for check in build_checks(args.host)
    ]
    passed = all(result["passed"] for result in results)
    print(json.dumps({"passed": passed, "checks": results}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
