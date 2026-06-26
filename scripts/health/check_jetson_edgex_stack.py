"""Jetson EdgeX 현장 배포 점검 스크립트."""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class HttpCheck:
    """HTTP 헬스체크 정의."""

    name: str
    url: str
    expect_status: int = 200
    headers: Optional[dict[str, str]] = None
    fallback_container: str = ""


@dataclass(frozen=True)
class TcpCheck:
    """TCP 포트 체크 정의."""

    name: str
    host: str
    port: int
    fallback_container: str = ""


def _check_tcp(item: TcpCheck, timeout: float) -> tuple[bool, str]:
    """TCP 포트 연결 가능 여부를 확인한다."""
    try:
        with socket.create_connection((item.host, item.port), timeout=timeout):
            return True, f"{item.host}:{item.port} 연결 성공"
    except OSError as exc:
        return False, f"{item.host}:{item.port} 연결 실패 ({exc})"


def _check_http(item: HttpCheck, timeout: float) -> tuple[bool, str]:
    """HTTP 엔드포인트 응답 상태를 확인한다."""
    request = urllib.request.Request(
        item.url,
        headers={
            "User-Agent": "jetson-edgex-check/1.0",
            **(item.headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            if response.status != item.expect_status:
                return False, f"status={response.status}, body={body[:160]}"
            return True, f"status={response.status}, body={body[:160]}"
    except urllib.error.HTTPError as exc:
        return False, f"status={exc.code}, reason={exc.reason}"
    except OSError as exc:
        return False, str(exc)


def _check_container_health(container_name: str, timeout: float) -> tuple[bool, str]:
    """호스트 포트가 닫힌 compose 내부 서비스의 Docker health 상태를 확인한다."""
    if not container_name:
        return False, "fallback 컨테이너가 지정되지 않음"
    try:
        completed = subprocess.run(
            [
                "docker",
                "inspect",
                container_name,
                "--format",
                "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Docker health 확인 실패 ({exc})"

    detail = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return False, f"Docker inspect 실패: {detail}"

    parts = detail.split()
    status = parts[0] if parts else "unknown"
    health = parts[1] if len(parts) > 1 else "none"
    ok = status == "running" and health in {"healthy", "none"}
    return ok, f"host 포트 미노출, Docker 상태 fallback: status={status}, health={health}"


def _check_tcp_with_fallback(item: TcpCheck, timeout: float) -> tuple[bool, str]:
    """TCP 점검 후 필요하면 컨테이너 health로 fallback한다."""
    ok, detail = _check_tcp(item, timeout)
    if ok or not item.fallback_container:
        return ok, detail
    fallback_ok, fallback_detail = _check_container_health(
        item.fallback_container,
        timeout,
    )
    return fallback_ok, f"{detail}; {fallback_detail}"


def _check_http_with_fallback(item: HttpCheck, timeout: float) -> tuple[bool, str]:
    """HTTP 점검 후 필요하면 컨테이너 health로 fallback한다."""
    ok, detail = _check_http(item, timeout)
    if ok or not item.fallback_container:
        return ok, detail
    fallback_ok, fallback_detail = _check_container_health(
        item.fallback_container,
        timeout,
    )
    return fallback_ok, f"{detail}; {fallback_detail}"


def _print_result(ok: bool, name: str, detail: str) -> None:
    """점검 결과를 한 줄로 출력한다."""
    prefix = "PASS" if ok else "FAIL"
    print(f"[{prefix}] {name}: {detail}")


def _check_python_import(name: str) -> tuple[bool, str]:
    """Python 모듈 import 가능 여부를 확인한다."""
    try:
        importlib.import_module(name)
        return True, "import 성공"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _build_deepstream_checks() -> list[tuple[str, str]]:
    """DeepStream Python 런타임 점검 목록을 구성한다."""
    return [
        ("GStreamer Python", "gi"),
        ("DeepStream pyds", "pyds"),
    ]


def _build_http_checks(
    host: str,
    *,
    public_api_port: int,
    public_api_key: str,
    include_appearance_status: bool,
) -> list[HttpCheck]:
    """기본 HTTP 점검 목록을 구성한다."""
    checks = [
        HttpCheck("EdgeX Core Metadata", f"http://{host}:59881/api/v3/ping"),
        HttpCheck("EdgeX Core Data", f"http://{host}:59880/api/v3/ping"),
        HttpCheck(
            "AIoT Parser",
            f"http://{host}:3500/health",
            fallback_container="aiot-parser",
        ),
        HttpCheck("Alert API", f"http://{host}:8000/health"),
        HttpCheck("Action Layer", f"http://{host}:8080/health"),
        HttpCheck("Public API", f"http://{host}:{public_api_port}/api/v1/health"),
    ]
    if include_appearance_status:
        headers = {"X-API-Key": public_api_key} if public_api_key else None
        checks.append(
            HttpCheck(
                "Public API Appearance Status",
                f"http://{host}:{public_api_port}/api/v1/appearances/status",
                headers=headers,
            )
        )
    return checks


def _build_tcp_checks(
    host: str,
    speaker_host: str,
    speaker_port: int,
    signboard_host: str,
    signboard_port: int,
    siren_host: str,
    siren_port: int,
) -> list[TcpCheck]:
    """기본 TCP 점검 목록을 구성한다."""
    checks = [
        TcpCheck("MQTT Broker", host, 1883),
        TcpCheck("Redis", "127.0.0.1", 6379),
        TcpCheck("AIoT Parser DB", host, 5432, fallback_container="aiot-parser-db"),
    ]
    if speaker_host:
        checks.append(TcpCheck("Speaker Device", speaker_host, speaker_port))
    if signboard_host:
        checks.append(TcpCheck("Signboard Device", signboard_host, signboard_port))
    if siren_host:
        checks.append(TcpCheck("Siren Device", siren_host, siren_port))
    return checks


def _summarize(failures: Iterable[str]) -> int:
    """최종 결과를 요약하고 종료 코드를 반환한다."""
    failed = list(failures)
    if not failed:
        print("\n전체 점검 통과")
        return 0

    print("\n확인이 필요한 항목")
    for item in failed:
        print(f"- {item}")
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="Jetson + EdgeX 스택 점검")
    parser.add_argument("--host", default="127.0.0.1", help="Jetson 호스트 주소")
    parser.add_argument("--timeout", type=float, default=3.0, help="개별 점검 타임아웃(초)")
    parser.add_argument("--public-api-port", type=int, default=9000, help="Public API 포트")
    parser.add_argument("--public-api-key", default="", help="Public API Key (appearances/status 점검 시 사용)")
    parser.add_argument("--speaker-host", default="", help="스피커 장비 주소")
    parser.add_argument("--speaker-port", type=int, default=80, help="스피커 장비 포트")
    parser.add_argument("--signboard-host", default="", help="전광판 장비 주소")
    parser.add_argument("--signboard-port", type=int, default=5000, help="전광판 장비 포트")
    parser.add_argument("--siren-host", default="", help="사이렌 장비 주소")
    parser.add_argument("--siren-port", type=int, default=80, help="사이렌 장비 포트")
    parser.add_argument(
        "--json",
        action="store_true",
        help="사람용 출력 대신 JSON 으로 결과를 반환",
    )
    parser.add_argument(
        "--deepstream",
        action="store_true",
        help="DeepStream Python 바인딩(gi, pyds) import 가능 여부도 점검",
    )
    parser.add_argument(
        "--check-appearance-status",
        action="store_true",
        help="Public API의 /api/v1/appearances/status 도 함께 점검",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    results: list[dict[str, object]] = []

    for item in _build_tcp_checks(
        host=args.host,
        speaker_host=args.speaker_host,
        speaker_port=args.speaker_port,
        signboard_host=args.signboard_host,
        signboard_port=args.signboard_port,
        siren_host=args.siren_host,
        siren_port=args.siren_port,
    ):
        ok, detail = _check_tcp_with_fallback(item, timeout=args.timeout)
        results.append({"type": "tcp", "name": item.name, "ok": ok, "detail": detail})
        if not ok:
            failures.append(item.name)

    for item in _build_http_checks(
        host=args.host,
        public_api_port=args.public_api_port,
        public_api_key=args.public_api_key,
        include_appearance_status=args.check_appearance_status,
    ):
        ok, detail = _check_http_with_fallback(item, timeout=args.timeout)
        results.append({"type": "http", "name": item.name, "ok": ok, "detail": detail})
        if not ok:
            failures.append(item.name)

    if args.deepstream:
        for name, module_name in _build_deepstream_checks():
            ok, detail = _check_python_import(module_name)
            results.append({"type": "python", "name": name, "ok": ok, "detail": detail})
            if not ok:
                failures.append(name)

    if args.json:
        print(json.dumps({"results": results, "failures": failures}, ensure_ascii=False, indent=2))
        return 1 if failures else 0

    for item in results:
        _print_result(bool(item["ok"]), str(item["name"]), str(item["detail"]))
    return _summarize(failures)


if __name__ == "__main__":
    sys.exit(main())
