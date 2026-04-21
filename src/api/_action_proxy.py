"""Action Layer 프록시 공통 유틸리티."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, status

from .dependencies._settings import INTERNAL_SERVICE_TOKEN as _INTERNAL_TOKEN

_INTERNAL_HEADERS: dict[str, str] = (
    {"X-Internal-Token": _INTERNAL_TOKEN} if _INTERNAL_TOKEN else {}
)


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
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_INTERNAL_HEADERS) as client:
            call = getattr(client, method)
            response = await (call(url, json=payload) if payload is not None else call(url))
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
