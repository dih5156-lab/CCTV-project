"""외형 기록 검색 엔드포인트.

SQLite에 기록된 인물 외형 속성을 조건부 검색하여
리스트 형태로 반환한다.  각 결과에는 인물 crop 이미지 경로가 포함된다.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ...services.appearance_log import AppearanceLog
from ...time_utils import KST
from ..dependencies.auth import verify_api_key
from ..schemas.common import PaginatedResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])

_COLOR_ALIASES = {
    "black": ("black", "검정", "검은", "검은색", "검정색", "블랙"),
    "red": ("red", "빨강", "빨간", "빨간색", "붉은", "레드"),
    "blue": ("blue", "파랑", "파란", "파란색", "블루", "청색"),
    "white": ("white", "흰", "흰색", "하얀", "하얀색", "화이트"),
    "gray": ("gray", "회색", "그레이"),
    "yellow": ("yellow", "노랑", "노란", "노란색", "옐로우"),
    "green": ("green", "초록", "초록색", "녹색", "그린"),
    "orange": ("orange", "주황", "주황색", "오렌지"),
    "purple": ("purple", "보라", "보라색", "퍼플"),
}

_UPPER_TERMS = ("상의", "윗옷", "상체", "top", "upper")
_LOWER_TERMS = ("하의", "바지", "하체", "bottom", "lower", "pants")
_BAG_TERMS = {
    "has_backpack": ("백팩", "배낭", "backpack"),
    "has_handbag": ("핸드백", "손가방", "handbag"),
    "has_suitcase": ("캐리어", "여행가방", "suitcase", "luggage"),
}

# ── 싱글턴 DB 인스턴스 ──────────────────────────────────────────────

_log_instance: Optional[AppearanceLog] = None

# ── crop 이미지 경로 ─────────────────────────────────────────────────

_PRIMARY_CROP_DIR = Path(
    os.environ.get("APPEARANCE_CROP_DIR", "data/runtime/appearance_crops")
)
# 하위 호환: 기존 테스트/코드에서 _CROP_DIR monkeypatch를 사용한다.
_CROP_DIR = _PRIMARY_CROP_DIR
_LEGACY_CROP_DIR = Path("data/crops")
_SAFE_FNAME = re.compile(r"^[\w\-]+\.jpg$")


def _get_log() -> AppearanceLog:
    global _log_instance
    if _log_instance is None:
        _log_instance = AppearanceLog()
    return _log_instance


def _find_crop_file(filename: str) -> Optional[Path]:
    """공개 가능한 crop 파일이 실제로 남아 있으면 경로를 반환한다."""
    if not _SAFE_FNAME.match(filename):
        return None

    candidate_dirs = [_CROP_DIR]
    if _LEGACY_CROP_DIR != _CROP_DIR:
        candidate_dirs.append(_LEGACY_CROP_DIR)

    for crop_dir in candidate_dirs:
        resolved_dir = crop_dir.resolve()
        file_path = (crop_dir / filename).resolve()
        if not str(file_path).startswith(str(resolved_dir)):
            continue
        if file_path.is_file():
            return file_path
    return None


# ── 응답 스키마 ──────────────────────────────────────────────────────


class AppearanceRecord(BaseModel):
    """검색 결과 개별 레코드."""

    id: int
    event_id: Optional[str] = None
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
    bbox: Optional[Dict[str, int]] = Field(
        default=None,
        description="검색 기록에 저장된 person bbox",
    )


def _to_record(row: dict) -> AppearanceRecord:
    crop_path = row.get("crop_path")
    crop_url: Optional[str] = None
    if crop_path:
        # 저장 위치와 상관없이 파일명만 공개 URL로 노출한다.
        fname = crop_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if _find_crop_file(fname) is not None:
            crop_url = f"/api/v1/search/crops/{fname}"

    dt = datetime.fromtimestamp(row["timestamp"], tz=KST)

    return AppearanceRecord(
        id=row["id"],
        event_id=row.get("event_id"),
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
        bbox={
            "x": int(row.get("bbox_x") or 0),
            "y": int(row.get("bbox_y") or 0),
            "width": int(row.get("bbox_w") or 0),
            "height": int(row.get("bbox_h") or 0),
        },
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
    q: Optional[str] = Query(
        None,
        description="자연어 조건. 예: 검정색 상의 빨간색 하의 사람",
    ),
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

    parsed_query = _parse_query(q)
    upper_color = upper_color or parsed_query.get("upper_color")
    lower_color = lower_color or parsed_query.get("lower_color")
    has_helmet = has_helmet if has_helmet is not None else parsed_query.get("has_helmet")
    helmet_color = helmet_color or parsed_query.get("helmet_color")
    has_backpack = has_backpack if has_backpack is not None else parsed_query.get("has_backpack")
    has_handbag = has_handbag if has_handbag is not None else parsed_query.get("has_handbag")
    has_suitcase = has_suitcase if has_suitcase is not None else parsed_query.get("has_suitcase")

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


def _parse_query(q: Optional[str]) -> Dict[str, object]:
    """한국어/영어 자연어 검색 문장을 API 필터로 변환한다.

    운영 부담을 줄이기 위해 LLM 호출 없이 색상+부위 키워드만 결정적으로 해석한다.
    """
    if not q:
        return {}

    text = q.strip().lower().replace("색상의", "색 상의")
    parsed: Dict[str, object] = {}
    for color, aliases in _COLOR_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            body_part = _nearest_body_part(text, alias)
            if body_part and body_part not in parsed:
                parsed[body_part] = color

    if any(term in text for term in ("헬멧 미착용", "안전모 미착용", "no helmet", "without helmet")):
        parsed["has_helmet"] = False
    elif any(term in text for term in ("헬멧", "안전모", "helmet")):
        parsed["has_helmet"] = True

    for field, terms in _BAG_TERMS.items():
        if any(term in text for term in terms):
            parsed[field] = True

    return parsed


def _nearest_body_part(text: str, color_alias: str, window: int = 12) -> Optional[str]:
    """색상 키워드와 가장 가까운 상의/하의 키워드를 찾는다."""
    start = 0
    while True:
        color_index = text.find(color_alias, start)
        if color_index < 0:
            return None
        color_end = color_index + len(color_alias)
        candidates = _body_part_candidates(text, color_index, color_end, window)
        if candidates:
            return min(candidates, key=lambda item: (item[0], item[1]))[2]
        start = color_end


def _body_part_candidates(
    text: str,
    color_index: int,
    color_end: int,
    window: int,
) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    for body_part, terms in (("upper_color", _UPPER_TERMS), ("lower_color", _LOWER_TERMS)):
        for term in terms:
            term_index = text.find(
                term,
                max(0, color_index - window),
                color_end + window + len(term),
            )
            if term_index < 0:
                continue
            candidates.append(
                (
                    *_term_distance(color_index, color_end, term_index, len(term)),
                    body_part,
                )
            )
    return candidates


def _term_distance(
    color_index: int,
    color_end: int,
    term_index: int,
    term_length: int,
) -> tuple[int, int]:
    term_end = term_index + term_length
    if term_index >= color_end:
        return 0, term_index - color_end
    if color_index >= term_end:
        return 1, color_index - term_end
    return 0, 0


def _parse_datetime(s: str) -> float:
    """ISO datetime 문자열을 unix timestamp로 변환한다.

    오프셋이 없는 입력은 운영 기준인 KST로 해석한다.
    """
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.timestamp()
    except ValueError:
        pass

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=KST)
            return dt.timestamp()
        except ValueError:
            continue
    raise HTTPException(status_code=400, detail=f"지원하지 않는 날짜 형식: {s}")


# ── crop 이미지 서빙 ─────────────────────────────────────────────────


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
    file_path = _find_crop_file(filename)
    if file_path is not None:
        return Response(content=file_path.read_bytes(), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
