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

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ._action_proxy import close_action_proxy_client
from ._local_docs import local_api_docs_html
from .dependencies._settings import ACTION_LAYER_URL, ALERT_API_URL
from .dependencies.rate_limit import limiter
from .schemas.common import error_response
from .v1 import (
    alerts,
    appearances,
    cameras,
    control,
    events,
    health,
    metrics,
    search,
    sensor_readings,
    sites,
)
from .v1.alerts import close_alert_client
from .v1.health import close_http_client
from .v1.sensor_readings import close_sensor_client

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

# 라이브러리 모듈에서는 basicConfig를 호출하지 않는다.
# 로깅 설정은 run_public_api.py (진입점 runner) 에서만 수행한다.
logger = logging.getLogger("cctv-public-api")


def _running_under_pytest() -> bool:
    """pytest 실행 중이면 일부 미들웨어를 가볍게 우회한다."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


# ---------------------------------------------------------------------------
# 앱 생성
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "CCTV Public API 시작 — Action Layer: %s | Alert API: %s",
        ACTION_LAYER_URL,
        ALERT_API_URL,
    )
    # NOTE: appearances.set_analyzer()는 단일 프로세스(로컬 개발) 모드에서만 사용한다.
    # Docker 배포 시 ai-engine과 public-api는 별개의 컨테이너이므로
    # 외형 조건 동기화는 SQLite DB(APPEARANCES_DB)를 통해 이뤄진다.
    yield
    await close_http_client()
    await close_action_proxy_client()
    await close_alert_client()
    await close_sensor_client()
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
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)

# slowapi — 앱 상태에 limiter 등록 + 미들웨어 + 예외 핸들러
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
if not _running_under_pytest():
    app.add_middleware(SlowAPIMiddleware)

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


def _metric_path_prefix(path: str) -> str:
    """동적 path 값을 낮은 cardinality의 Prometheus label로 정규화한다."""
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "v1":
        return f"/api/v1/{parts[2]}"
    if path in {"/", "/docs", "/redoc", "/openapi.json"}:
        return path
    return "/other"


@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    """Public API HTTP 요청 수를 Prometheus counter에 기록한다."""
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        metrics.http_requests_total.labels(
            method=request.method,
            path_prefix=_metric_path_prefix(request.url.path),
            status_code=str(status_code),
        ).inc()


# ---------------------------------------------------------------------------
# 전역 예외 핸들러
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root_info() -> dict:
    """브라우저로 루트 경로를 열었을 때 사용할 서비스 안내."""
    return {
        "service": "cctv-public-api",
        "description": "CCTV Platform Public API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "events": "/api/v1/events",
        "sensor_readings": "/api/v1/sensor-readings",
        "cameras": "/api/v1/cameras",
        "sites": "/api/v1/sites",
        "search": "/api/v1/search",
    }


@app.get("/docs", include_in_schema=False)
async def local_api_docs() -> HTMLResponse:
    """외부 CDN 없이 동작하는 가벼운 OpenAPI 문서 페이지."""
    return HTMLResponse(local_api_docs_html())

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTP 예외: %s %s -> %s", request.method, request.url, exc.status_code)
    body = error_response(str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"))


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("입력 검증 실패: %s %s", request.method, request.url)
    details = "; ".join(err.get("msg", "invalid input") for err in exc.errors())
    body = error_response(details or "잘못된 요청입니다.")
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("처리되지 않은 예외: %s %s", request.method, request.url)
    body = error_response("내부 서버 오류가 발생했습니다.")
    return JSONResponse(status_code=500, content=body.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# 라우터 등록
# ---------------------------------------------------------------------------

_PREFIX = "/api/v1"

app.include_router(health.router, prefix=_PREFIX)
app.include_router(alerts.router, prefix=_PREFIX)
app.include_router(events.router, prefix=_PREFIX)
app.include_router(sensor_readings.router, prefix=_PREFIX)
app.include_router(cameras.router, prefix=_PREFIX)
app.include_router(sites.router, prefix=_PREFIX)
app.include_router(control.router, prefix=_PREFIX)
app.include_router(appearances.router, prefix=_PREFIX)
app.include_router(search.router, prefix=_PREFIX)
app.include_router(metrics.router, prefix=_PREFIX)
