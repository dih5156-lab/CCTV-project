"""slowapi 레이트 리미터 공유 인스턴스.

엔드포인트에서 `from ..dependencies.rate_limit import limiter` 로 임포트하고
데코레이터로 적용한다::

    @router.get("/foo")
    @limiter.limit("60/minute")
    async def foo(request: Request, ...):
        ...

앱 초기화 시 SlowAPIMiddleware와 예외 핸들러 등록이 필요하다 (app.py 참고).

환경변수:
    RATE_LIMIT_ENABLED  : "false" 로 설정하면 전체 비활성화 (기본 true)
"""

from __future__ import annotations

import logging
import os

from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

_enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() not in ("false", "0", "no")

if not _enabled:
    logger.warning("레이트 리밋이 비활성화되어 있습니다 (RATE_LIMIT_ENABLED=false).")

limiter = Limiter(
    key_func=get_remote_address,
    enabled=_enabled,
)
