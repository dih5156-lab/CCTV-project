"""Action Layer 프록시 공통 유틸리티."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, status

from .dependencies._settings import INTERNAL_SERVICE_TOKEN as _INTERNAL_TOKEN

_INTERNAL_HEADERS: dict[str, str] = (
    {"X-Internal-Token": _INTERNAL_TOKEN} if _INTERNAL_TOKEN else {}
)

# 요청마다 새 클라이언트 생성 제거 — 커넥션 풀 재사용
_shared_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=5.0,
            headers=_INTERNAL_HEADERS,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )
    return _shared_client


async def close_action_proxy_client() -> None:
    """Public API 종료 시 Action Layer 프록시 HTTP 클라이언트를 닫는다."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


def _extract_error_detail(response: httpx.Response) -> str:
    """Action Layer 응답에서 사용자 친화적인 오류 메시지를 뽑는다."""
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if response.text.strip():
        return response.text.strip()
    return f"Action Layer 요청 실패 ({response.status_code})"


async def proxy_action_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: float = 5.0,
) -> Any:
    """Action Layer REST 서버로 요청을 보내고 JSON 응답을 반환한다."""
    url = f"{base_url.rstrip('/')}{path}"
    client = _get_client()
    try:
        call = getattr(client, method)
        response = await (
            call(url, json=payload, timeout=timeout)
            if payload is not None
            else call(url, timeout=timeout)
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=_extract_error_detail(exc.response),
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Action Layer 응답 시간이 초과되었습니다.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Action Layer에 연결할 수 없습니다.",
        ) from exc
