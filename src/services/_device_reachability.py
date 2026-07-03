"""출력 장치 연결 상태 확인 유틸리티."""

from __future__ import annotations

import socket
from typing import Optional

DEVICE_REACHABILITY_TIMEOUT_SECONDS = 1.5
DEVICE_REACHABILITY_CACHE_SECONDS = 30.0


def check_tcp_reachable(host: str, port: int) -> bool:
    """설정된 출력 장치가 짧은 TCP 연결을 받는지 확인한다."""
    try:
        with socket.create_connection(
            (host, port), timeout=DEVICE_REACHABILITY_TIMEOUT_SECONDS
        ):
            return True
    except OSError:
        return False


def device_status(configured: bool, reachable: Optional[bool]) -> str:
    if not configured:
        return "disabled"
    return "online" if reachable else "unreachable"
