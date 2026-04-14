"""외형 검색 조건 CRUD 엔드포인트.

사용자가 CCTV에서 찾고 싶은 외형 조건(상의 색상, 하의 색상 등)을
등록·조회·삭제할 수 있다.

등록된 조건은 AI 엔진의 AppearanceAnalyzer에 전달되어
실시간 매칭에 사용된다.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..dependencies.auth import verify_api_key
from ..schemas.common import BaseResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/appearances", tags=["appearances"])


# ── Pydantic 스키마 ──────────────────────────────────────────────────

_VALID_COLORS = frozenset({
    "red", "orange", "yellow", "green", "blue",
    "purple", "white", "black", "gray",
})


class AppearanceConditionIn(BaseModel):
    """외형 조건 등록 요청."""

    name: str = Field(min_length=1, max_length=100, description="조건 이름 (예: 의심인물_A)")
    upper_color: Optional[str] = Field(
        default=None,
        description="상의 색상 (red, orange, yellow, green, blue, purple, white, black, gray)",
    )
    lower_color: Optional[str] = Field(
        default=None,
        description="하의 색상",
    )
    hat_color: Optional[str] = Field(
        default=None,
        description="모자 색상",
    )
    has_backpack: Optional[bool] = Field(
        default=None,
        description="백팩 소지 여부",
    )
    has_handbag: Optional[bool] = Field(
        default=None,
        description="핸드백 소지 여부",
    )
    has_suitcase: Optional[bool] = Field(
        default=None,
        description="여행가방 소지 여부",
    )
    threshold: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="매칭 임계값 (기본 0.8)",
    )
    cameras: Optional[List[str]] = Field(
        default=None,
        description="적용 카메라 목록 (null이면 전체)",
    )
    enabled: bool = Field(default=True, description="활성화 여부")

    def validate_colors(self) -> None:
        """최소 1개 조건 필수, 유효한 색상명인지 검증."""
        has_any = (
            self.upper_color
            or self.lower_color
            or self.hat_color
            or self.has_backpack is not None
            or self.has_handbag is not None
            or self.has_suitcase is not None
        )
        if not has_any:
            raise ValueError("색상 또는 소지품 조건 중 최소 1개는 필수입니다.")
        for label, val in [("upper_color", self.upper_color), ("lower_color", self.lower_color), ("hat_color", self.hat_color)]:
            if val and val not in _VALID_COLORS:
                raise ValueError(f"유효하지 않은 색상: {label}={val}")


class AppearanceConditionOut(BaseModel):
    """외형 조건 응답."""

    id: str
    name: str
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    hat_color: Optional[str] = None
    has_backpack: Optional[bool] = None
    has_handbag: Optional[bool] = None
    has_suitcase: Optional[bool] = None
    threshold: float
    cameras: Optional[List[str]] = None
    enabled: bool


class AppearanceConditionList(BaseModel):
    """조건 목록 응답."""

    conditions: List[AppearanceConditionOut]
    total: int


# ── 인메모리 저장소 ──────────────────────────────────────────────────
# 프로덕션에서는 DB/Redis로 교체 가능 — 현재는 단일 프로세스 인메모리
_conditions: List[dict] = []

# AppearanceAnalyzer 인스턴스 참조 (앱 시작 시 주입)
_analyzer_ref = None


def set_analyzer(analyzer) -> None:
    """AIAnalyzer._appearance 인스턴스를 주입한다."""
    global _analyzer_ref
    _analyzer_ref = analyzer


def _sync_to_analyzer() -> None:
    """인메모리 조건을 AppearanceAnalyzer에 동기화한다."""
    if _analyzer_ref is not None:
        _analyzer_ref.set_conditions(_conditions)


# ── 엔드포인트 ───────────────────────────────────────────────────────


@router.get(
    "",
    response_model=BaseResponse[AppearanceConditionList],
    summary="외형 조건 목록 조회",
    description="등록된 외형 검색 조건 전체를 조회합니다.",
)
async def list_conditions(
    _: None = Depends(verify_api_key),
) -> BaseResponse[AppearanceConditionList]:
    return BaseResponse(
        success=True,
        data=AppearanceConditionList(
            conditions=[AppearanceConditionOut(**c) for c in _conditions],
            total=len(_conditions),
        ),
    )


@router.post(
    "",
    response_model=BaseResponse[AppearanceConditionOut],
    status_code=status.HTTP_201_CREATED,
    summary="외형 조건 등록",
    description="새로운 외형 검색 조건을 등록합니다.",
)
async def create_condition(
    body: AppearanceConditionIn,
    _: None = Depends(verify_api_key),
) -> BaseResponse[AppearanceConditionOut]:
    try:
        body.validate_colors()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    entry = {
        "id": str(uuid.uuid4())[:8],
        "name": body.name,
        "upper_color": body.upper_color,
        "lower_color": body.lower_color,
        "hat_color": body.hat_color,
        "has_backpack": body.has_backpack,
        "has_handbag": body.has_handbag,
        "has_suitcase": body.has_suitcase,
        "threshold": body.threshold,
        "cameras": body.cameras,
        "enabled": body.enabled,
    }
    _conditions.append(entry)
    _sync_to_analyzer()
    logger.info("외형 조건 등록: %s (%s)", entry["id"], entry["name"])

    return BaseResponse(success=True, data=AppearanceConditionOut(**entry))


@router.delete(
    "/{condition_id}",
    response_model=BaseResponse[dict],
    summary="외형 조건 삭제",
    description="ID로 외형 검색 조건을 삭제합니다.",
)
async def delete_condition(
    condition_id: str,
    _: None = Depends(verify_api_key),
) -> BaseResponse[dict]:
    global _conditions
    before = len(_conditions)
    _conditions = [c for c in _conditions if c["id"] != condition_id]

    if len(_conditions) == before:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"조건을 찾을 수 없습니다: {condition_id}",
        )

    _sync_to_analyzer()
    logger.info("외형 조건 삭제: %s", condition_id)
    return BaseResponse(success=True, data={"deleted": condition_id})
