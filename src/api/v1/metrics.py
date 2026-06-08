"""GET /api/v1/metrics — Prometheus 메트릭 스크랩 엔드포인트.

prometheus-client 기본 REGISTRY(Python 프로세스 메트릭 포함)를 노출한다.
Public API 서비스에서 HTTP 요청 수, 응답 코드 등을 스크랩하는 데 사용한다.

Prometheus scrape_config 예::

    - job_name: 'cctv-public-api'
      static_configs:
        - targets: ['cctv-public-api:9000']
      metrics_path: '/api/v1/metrics'
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)
from prometheus_client import REGISTRY as _DEFAULT_REGISTRY

from .health import _fd_usage

router = APIRouter(tags=["metrics"])

# HTTP 요청 카운터 (public-api 전용)
http_requests_total: Counter = Counter(
    "cctv_public_api_http_requests_total",
    "Public API HTTP 요청 총 수",
    ["method", "path_prefix", "status_code"],
)

open_file_descriptors: Gauge = Gauge(
    "cctv_public_api_open_file_descriptors",
    "Public API 프로세스가 현재 열고 있는 file descriptor 수",
)

file_descriptor_soft_limit: Gauge = Gauge(
    "cctv_public_api_file_descriptor_soft_limit",
    "Public API 프로세스의 file descriptor soft limit",
)


@router.get("/metrics", include_in_schema=False)
async def get_metrics() -> Response:
    """Prometheus 메트릭 스크랩 엔드포인트.

    Python 프로세스 메트릭(gc, memory, threads 등)과
    ``cctv_public_api_http_requests_total`` 카운터를 노출한다.
    """
    fd_usage = _fd_usage()
    if fd_usage["open"] is not None:
        open_file_descriptors.set(fd_usage["open"])
    if fd_usage["soft_limit"] is not None:
        file_descriptor_soft_limit.set(fd_usage["soft_limit"])

    return Response(
        content=generate_latest(_DEFAULT_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
