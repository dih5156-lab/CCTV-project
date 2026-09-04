"""GET /api/v1/command-results — EdgeX 장치 결과 조회 엔드포인트."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ...edgex.command_result_collector import CommandResultStore
from ..dependencies.auth import verify_api_key
from ..schemas.common import BaseResponse, success_response

router = APIRouter(prefix="/command-results", tags=["command-results"])


def _store() -> CommandResultStore:
    """환경변수로 지정된 EdgeX 결과 저장소를 생성한다."""
    path = os.environ.get(
        "EDGEX_COMMAND_RESULT_DB",
        "/app/data/runtime/edgex_command_results.db",
    )
    return CommandResultStore(Path(path))


@router.get("", response_model=BaseResponse[list[dict]])
async def list_command_results(
    device_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    _: None = Depends(verify_api_key),
) -> BaseResponse[list[dict]]:
    """EdgeX 장치 결과를 장치·상태 조건으로 조회한다."""
    return success_response(
        _store().list_recent(limit, device_id=device_id, status=status)
    )


@router.get("/{request_id}", response_model=BaseResponse[dict])
async def get_command_result(
    request_id: str,
    _: None = Depends(verify_api_key),
) -> BaseResponse[dict]:
    """request_id로 EdgeX 장치 결과 한 건을 조회한다."""
    result = _store().get(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Command 결과를 찾을 수 없습니다.")
    return success_response(result)
