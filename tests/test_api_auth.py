"""공개 API 인증 단위 테스트."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from src.api.dependencies.auth import verify_api_key


def test_verify_api_key_accepts_header_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_API_KEY", "public-secret")
    monkeypatch.delenv("PUBLIC_API_ALLOW_QUERY_KEY", raising=False)

    asyncio.run(verify_api_key(header_key="public-secret", query_key=None))


def test_verify_api_key_rejects_query_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_API_KEY", "public-secret")
    monkeypatch.delenv("PUBLIC_API_ALLOW_QUERY_KEY", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(verify_api_key(header_key=None, query_key="public-secret"))

    assert exc_info.value.status_code == 401
    assert "X-API-Key" in str(exc_info.value.detail)


def test_verify_api_key_allows_query_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_API_KEY", "public-secret")
    monkeypatch.setenv("PUBLIC_API_ALLOW_QUERY_KEY", "1")

    asyncio.run(verify_api_key(header_key=None, query_key="public-secret"))


def test_verify_api_key_requires_configured_key_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PUBLIC_API_KEY", raising=False)
    monkeypatch.setenv("APP_ENV", "production")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(verify_api_key(header_key=None, query_key=None))

    assert exc_info.value.status_code == 503


def test_verify_api_key_requires_configured_key_when_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PUBLIC_API_KEY", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setenv("REQUIRE_PUBLIC_API_KEY", "1")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(verify_api_key(header_key=None, query_key=None))

    assert exc_info.value.status_code == 503
