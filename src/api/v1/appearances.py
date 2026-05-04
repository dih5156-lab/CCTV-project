"""외형 검색 조건 CRUD 엔드포인트.

사용자가 CCTV에서 찾고 싶은 외형 조건(상의 색상, 하의 색상, 헬멧 착용 여부 등)을
등록·조회·삭제할 수 있다.

등록된 조건은 AI 엔진의 AppearanceAnalyzer에 전달되어
실시간 매칭에 사용된다.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from ..dependencies.auth import verify_api_key
from ..schemas.common import BaseResponse, success_response
from ...services.appearance_conditions import AppearanceConditionStore
from ...services.appearance_status import (
    AppearanceRuntimeStatus,
    build_runtime_status,
)

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
    has_helmet: Optional[bool] = Field(
        default=None,
        description="헬멧 착용 여부",
    )
    helmet_color: Optional[str] = Field(
        default=None,
        description="헬멧 색상",
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

    @model_validator(mode="after")
    def _check_conditions(self) -> "AppearanceConditionIn":
        """최소 1개 조건 필수, 유효한 색상명인지 Pydantic 검증."""
        has_any = (
            self.upper_color
            or self.lower_color
            or self.has_helmet is not None
            or self.helmet_color
            or self.has_backpack is not None
            or self.has_handbag is not None
            or self.has_suitcase is not None
        )
        if not has_any:
            raise ValueError("색상 또는 소지품 조건 중 최소 1개는 필수입니다.")
        for label, val in [
            ("upper_color", self.upper_color),
            ("lower_color", self.lower_color),
            ("helmet_color", self.helmet_color),
        ]:
            if val and val not in _VALID_COLORS:
                raise ValueError(f"유효하지 않은 색상: {label}={val}")
        return self


class AppearanceConditionOut(BaseModel):
    """외형 조건 응답."""

    id: str
    name: str
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    has_helmet: Optional[bool] = None
    helmet_color: Optional[str] = None
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


_DB_PATH = Path(os.environ.get("APPEARANCES_DB", "/app/data/appearances.db"))


def _load_all() -> List[dict]:
    return AppearanceConditionStore(_DB_PATH).list_all()


def _build_runtime_status() -> AppearanceRuntimeStatus:
    return build_runtime_status(_DB_PATH)


# AppearanceAnalyzer 인스턴스 참조 (앱 시작 시 주입)
_analyzer_ref = None


def set_analyzer(analyzer) -> None:
    """AIAnalyzer._appearance 인스턴스를 주입한다."""
    global _analyzer_ref
    _analyzer_ref = analyzer
    # 시작 시 DB에서 조건을 불러와 analyzer에 동기화
    _sync_to_analyzer()


def _sync_to_analyzer() -> None:
    """DB 조건을 AppearanceAnalyzer에 동기화한다."""
    if _analyzer_ref is not None:
        _analyzer_ref.set_conditions(_load_all())


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
    conditions = _load_all()
    return success_response(
        AppearanceConditionList(
            conditions=[AppearanceConditionOut(**c) for c in conditions],
            total=len(conditions),
        )
    )


@router.get(
    "/status",
    response_model=BaseResponse[AppearanceRuntimeStatus],
    summary="외형 검색 준비 상태 조회",
    description=(
        "대시보드에서 외형 검색 필드별 준비 상태와 실제 적재 통계를 함께 조회합니다. "
        "값이 비는 원인이 설정 문제인지, 데이터 부족인지 구분할 때 사용합니다."
    ),
)
async def get_appearance_status(
    _: None = Depends(verify_api_key),
) -> BaseResponse[AppearanceRuntimeStatus]:
    return success_response(_build_runtime_status())


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
    cid = str(uuid.uuid4())[:8]
    payload = {
        "upper_color": body.upper_color,
        "lower_color": body.lower_color,
        "has_helmet": body.has_helmet,
        "helmet_color": body.helmet_color,
        "has_backpack": body.has_backpack,
        "has_handbag": body.has_handbag,
        "has_suitcase": body.has_suitcase,
        "threshold": body.threshold,
        "cameras": body.cameras,
    }
    entry = AppearanceConditionStore(_DB_PATH).create(
        condition_id=cid,
        name=body.name,
        payload=payload,
        enabled=body.enabled,
    )
    _sync_to_analyzer()
    logger.info("외형 조건 등록: %s (%s)", cid, body.name)

    return success_response(AppearanceConditionOut(**entry))


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
    if not AppearanceConditionStore(_DB_PATH).delete(condition_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"조건을 찾을 수 없습니다: {condition_id}",
        )

    _sync_to_analyzer()
    logger.info("외형 조건 삭제: %s", condition_id)
    return success_response({"deleted": condition_id})
