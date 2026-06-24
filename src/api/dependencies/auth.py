"""API Key 인증 의존성.

기본적으로 X-API-Key 헤더만 허용한다.
운영상 꼭 필요할 때만 PUBLIC_API_ALLOW_QUERY_KEY=1 로
?api_key= 쿼리 파라미터 인증을 임시 허용할 수 있다.
환경변수 PUBLIC_API_KEY 에 설정된 값과 대조한다.
키가 설정되지 않으면 개발 편의를 위해 통과시키되 경고를 남긴다.
단, APP_ENV=production 또는 REQUIRE_PUBLIC_API_KEY=1이면 키를 반드시 요구한다.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

logger = logging.getLogger(__name__)

_API_KEY_NAME = "X-API-Key"
_QUERY_API_KEY_NAME = "api_key"
_PUBLIC_API_KEY_ENV = "PUBLIC_API_KEY"
_ALLOW_QUERY_KEY_ENV = "PUBLIC_API_ALLOW_QUERY_KEY"
_REQUIRE_PUBLIC_API_KEY_ENV = "REQUIRE_PUBLIC_API_KEY"
_APP_ENV = "APP_ENV"

_MISSING_KEY_DETAIL = "PUBLIC_API_KEY가 설정되지 않았습니다."
_HEADER_REQUIRED_DETAIL = "API Key가 필요합니다. X-API-Key 헤더를 제공하세요."
_HEADER_OR_QUERY_REQUIRED_DETAIL = (
    "API Key가 필요합니다. X-API-Key 헤더 또는 ?api_key= 파라미터를 제공하세요."
)
_INVALID_KEY_DETAIL = "유효하지 않은 API Key입니다."

_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)
_api_key_query = APIKeyQuery(name=_QUERY_API_KEY_NAME, auto_error=False)


def _get_configured_key() -> str | None:
    return os.environ.get(_PUBLIC_API_KEY_ENV) or None


def _get_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _allow_query_api_key() -> bool:
    return _get_env_bool(_ALLOW_QUERY_KEY_ENV, default=False)


def _is_production_env() -> bool:
    return os.environ.get(_APP_ENV, "").strip().lower() in {
        "prod",
        "production",
    }


def _require_public_api_key() -> bool:
    return _is_production_env() or _get_env_bool(
        _REQUIRE_PUBLIC_API_KEY_ENV,
        default=False,
    )


async def verify_api_key(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> None:
    configured = _get_configured_key()

    if configured is None:
        if _require_public_api_key():
            logger.error("PUBLIC_API_KEY가 필수인 환경이지만 값이 설정되지 않았습니다.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_MISSING_KEY_DETAIL,
            )

        logger.warning(
            "PUBLIC_API_KEY 환경변수가 설정되지 않았습니다. "
            "프로덕션 환경에서는 반드시 설정하세요."
        )
        return

    provided = header_key
    if provided is None and _allow_query_api_key():
        provided = query_key
    if provided is None:
        detail = _HEADER_REQUIRED_DETAIL
        if _allow_query_api_key():
            detail = _HEADER_OR_QUERY_REQUIRED_DETAIL
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
        )

    if not secrets.compare_digest(provided, configured):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_INVALID_KEY_DETAIL,
        )
