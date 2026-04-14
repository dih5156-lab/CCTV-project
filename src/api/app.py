"""FastAPI 공개 API 애플리케이션.

서버팀과 공유하는 CCTV 플랫폼 공개 API 엔트리포인트.

설계 원칙:
- 모든 엔드포인트는 /api/v1/ prefix를 가진다
- 통일된 BaseResponse 래퍼로 응답한다
- X-API-Key 헤더로 인증한다
- web/ 대시보드는 포함하지 않는다 (별도 서비스)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .v1 import alerts, appearances, cameras, control, events, health, search, sites

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cctv-public-api")


# ---------------------------------------------------------------------------
# 앱 생성
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "CCTV Public API 시작 — Action Layer: %s | Alert API: %s",
        os.environ.get("ACTION_LAYER_URL", "http://cctv-action-layer:8080"),
        os.environ.get("ALERT_API_URL", "http://cctv-alert-api:8000"),
    )
    yield
    logger.info("CCTV Public API 종료")


app = FastAPI(
    title="CCTV Platform API",
    description=(
        "CCTV 플랫폼 공개 API입니다.\n\n"
        "## 인증\n"
        "`X-API-Key` 헤더에 발급된 API Key를 포함하세요.\n\n"
        "## 응답 형식\n"
        "모든 응답은 `{ success, data, error, timestamp }` 형식의 공통 래퍼를 사용합니다."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS — 서버팀 도메인으로 제한 (환경변수로 설정)
# ---------------------------------------------------------------------------

_origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "").split(",")
    if o.strip()
]
if not _origins:
    _origins = ["*"]  # 개발 환경 기본값

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 전역 예외 핸들러
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("처리되지 않은 예외: %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": "내부 서버 오류가 발생했습니다.",
        },
    )


# ---------------------------------------------------------------------------
# 라우터 등록
# ---------------------------------------------------------------------------

_PREFIX = "/api/v1"

app.include_router(health.router, prefix=_PREFIX)
app.include_router(alerts.router, prefix=_PREFIX)
app.include_router(events.router, prefix=_PREFIX)
app.include_router(cameras.router, prefix=_PREFIX)
app.include_router(sites.router, prefix=_PREFIX)
app.include_router(control.router, prefix=_PREFIX)
app.include_router(appearances.router, prefix=_PREFIX)
app.include_router(search.router, prefix=_PREFIX)
