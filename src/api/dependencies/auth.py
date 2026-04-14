"""API Key 인증 의존성.

X-API-Key 헤더 또는 ?api_key= 쿼리 파라미터로 인증한다.
환경변수 PUBLIC_API_KEY 에 설정된 값과 대조한다.
키가 설정되지 않으면 개발 편의를 위해 통과시키되 경고를 남긴다.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

logger = logging.getLogger(__name__)

_API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)
_api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def _get_configured_key() -> str | None:
    return os.environ.get("PUBLIC_API_KEY") or None


def verify_api_key(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> None:
    """FastAPI Depends 로 사용하는 API Key 검증 함수."""
    configured = _get_configured_key()

    if configured is None:
        # 키가 설정되지 않으면 개발 모드 — 경고만 출력
        logger.warning(
            "PUBLIC_API_KEY 환경변수가 설정되지 않았습니다. "
            "프로덕션 환경에서는 반드시 설정하세요."
        )
        return

    provided = header_key or query_key
    if provided is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key가 필요합니다. X-API-Key 헤더 또는 ?api_key= 파라미터를 제공하세요.",
        )

    # timing-safe 비교로 timing attack 방지
    if not secrets.compare_digest(provided, configured):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="유효하지 않은 API Key입니다.",
        )
