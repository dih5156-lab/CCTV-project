"""외형 기록 검색 엔드포인트.

SQLite에 기록된 인물 외형 속성을 조건부 검색하여
리스트 형태로 반환한다.  각 결과에는 인물 crop 이미지 경로가 포함된다.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..dependencies.auth import verify_api_key
from ..schemas.common import PaginatedResponse
from ...services.appearance_log import AppearanceLog

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])

# ── 싱글턴 DB 인스턴스 ──────────────────────────────────────────────

_log_instance: Optional[AppearanceLog] = None


def _get_log() -> AppearanceLog:
    global _log_instance
    if _log_instance is None:
        _log_instance = AppearanceLog()
    return _log_instance


# ── 응답 스키마 ──────────────────────────────────────────────────────


class AppearanceRecord(BaseModel):
    """검색 결과 개별 레코드."""

    id: int
    timestamp: float
    datetime_str: str = Field(description="사람 읽을 수 있는 시각 문자열")
    camera_id: str
    track_id: Optional[int] = None
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    has_helmet: bool = False
    helmet_color: Optional[str] = None
    has_backpack: bool = False
    has_handbag: bool = False
    has_suitcase: bool = False
    gender: Optional[str] = None
    age_group: Optional[str] = None
    face_name: Optional[str] = None
    attribute_backend: Optional[str] = None
    crop_url: Optional[str] = Field(
        default=None,
        description="인물 crop 이미지 URL (/api/v1/search/crops/...)",
    )


def _to_record(row: dict) -> AppearanceRecord:
    crop_path = row.get("crop_path")
    crop_url: Optional[str] = None
    if crop_path:
        # 저장 위치와 상관없이 파일명만 공개 URL로 노출한다.
        fname = crop_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        crop_url = f"/api/v1/search/crops/{fname}"

    dt = datetime.fromtimestamp(row["timestamp"], tz=timezone.utc)

    return AppearanceRecord(
        id=row["id"],
        timestamp=row["timestamp"],
        datetime_str=dt.strftime("%Y-%m-%d %H:%M:%S"),
        camera_id=row["camera_id"],
        track_id=row.get("track_id"),
        upper_color=row.get("upper_color"),
        lower_color=row.get("lower_color"),
        has_helmet=row.get("has_helmet", False),
        helmet_color=row.get("helmet_color"),
        has_backpack=row.get("has_backpack", False),
        has_handbag=row.get("has_handbag", False),
        has_suitcase=row.get("has_suitcase", False),
        gender=row.get("gender"),
        age_group=row.get("age_group"),
        face_name=row.get("face_name"),
        attribute_backend=row.get("attribute_backend"),
        crop_url=crop_url,
    )


# ── 검색 엔드포인트 ─────────────────────────────────────────────────


@router.get(
    "",
    response_model=PaginatedResponse[AppearanceRecord],
    summary="외형 기록 조건부 검색",
    description=(
        "저장된 인물 외형 기록을 조건으로 검색합니다.\n\n"
        "**예시**: `?upper_color=black&gender=male&time_from=2025-06-01T14:00:00`"
    ),
    dependencies=[Depends(verify_api_key)],
)
async def search_appearances(
    camera_id: Optional[str] = Query(None, description="카메라 ID"),
    upper_color: Optional[str] = Query(None, description="상의 색상"),
    lower_color: Optional[str] = Query(None, description="하의 색상"),
    has_helmet: Optional[bool] = Query(None, description="헬멧 착용 여부"),
    helmet_color: Optional[str] = Query(None, description="헬멧 색상"),
    has_backpack: Optional[bool] = Query(None, description="백팩 소지"),
    has_handbag: Optional[bool] = Query(None, description="핸드백 소지"),
    has_suitcase: Optional[bool] = Query(None, description="캐리어 소지"),
    gender: Optional[str] = Query(None, description="성별 (male/female)"),
    age_group: Optional[str] = Query(None, description="나이대"),
    face_name: Optional[str] = Query(None, description="얼굴 이름 (부분 일치)"),
    time_from: Optional[str] = Query(
        None,
        description="검색 시작 시각 (ISO 형식: 2025-06-01T14:00:00)",
    ),
    time_to: Optional[str] = Query(
        None,
        description="검색 종료 시각 (ISO 형식: 2025-06-01T15:00:00)",
    ),
    limit: int = Query(50, ge=1, le=500, description="페이지 당 결과 수"),
    offset: int = Query(0, ge=0, description="건너뛸 결과 수"),
) -> PaginatedResponse[AppearanceRecord]:
    log = _get_log()

    # ISO datetime → unix timestamp 변환
    ts_from = _parse_datetime(time_from) if time_from else None
    ts_to = _parse_datetime(time_to) if time_to else None

    search_kwargs = dict(
        camera_id=camera_id,
        upper_color=upper_color,
        lower_color=lower_color,
        has_helmet=has_helmet,
        helmet_color=helmet_color,
        has_backpack=has_backpack,
        has_handbag=has_handbag,
        has_suitcase=has_suitcase,
        gender=gender,
        age_group=age_group,
        face_name=face_name,
        time_from=ts_from,
        time_to=ts_to,
    )

    rows = log.search(**search_kwargs, limit=limit, offset=offset)
    total = log.count(**search_kwargs)

    return PaginatedResponse[AppearanceRecord](
        success=True,
        items=[_to_record(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


def _parse_datetime(s: str) -> float:
    """ISO datetime 문자열을 unix timestamp로 변환한다."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    raise HTTPException(status_code=400, detail=f"지원하지 않는 날짜 형식: {s}")


# ── crop 이미지 서빙 ─────────────────────────────────────────────────

_PRIMARY_CROP_DIR = Path(
    os.environ.get("APPEARANCE_CROP_DIR", "data/appearance_crops")
)
# 하위 호환: 기존 테스트/코드에서 _CROP_DIR monkeypatch를 사용한다.
_CROP_DIR = _PRIMARY_CROP_DIR
_LEGACY_CROP_DIR = Path("data/crops")
_SAFE_FNAME = re.compile(r"^[\w\-]+\.jpg$")


@router.get(
    "/crops/{filename}",
    summary="인물 crop 이미지 조회",
    responses={200: {"content": {"image/jpeg": {}}}},
    dependencies=[Depends(verify_api_key)],
)
async def get_crop_image(filename: str):
    """저장된 인물 crop JPEG 이미지를 반환한다."""
    if not _SAFE_FNAME.match(filename):
        raise HTTPException(status_code=400, detail="잘못된 파일명")
    candidate_dirs = [_CROP_DIR]
    if _LEGACY_CROP_DIR != _CROP_DIR:
        candidate_dirs.append(_LEGACY_CROP_DIR)

    for crop_dir in candidate_dirs:
        resolved_dir = crop_dir.resolve()
        fpath = (crop_dir / filename).resolve()
        # path traversal 방지
        if not str(fpath).startswith(str(resolved_dir)):
            continue
        if fpath.is_file():
            return Response(content=fpath.read_bytes(), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
