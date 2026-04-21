"""외형 검색 조건 CRUD 엔드포인트.

사용자가 CCTV에서 찾고 싶은 외형 조건(상의 색상, 하의 색상, 헬멧 착용 여부 등)을
등록·조회·삭제할 수 있다.

등록된 조건은 AI 엔진의 AppearanceAnalyzer에 전달되어
실시간 매칭에 사용된다.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from ..dependencies.auth import verify_api_key
from ..schemas.common import BaseResponse, success_response

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


# ── SQLite 영속화 저장소 ─────────────────────────────────────────────
_DB_PATH = Path(os.environ.get("APPEARANCES_DB", "/app/data/appearances.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_conditions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);
"""


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_SCHEMA)
        conn.commit()
        yield conn
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    entry = json.loads(row["payload"])
    entry["id"] = row["id"]
    entry["name"] = row["name"]
    entry["enabled"] = bool(row["enabled"])
    return entry


def _load_all() -> List[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM search_conditions ORDER BY created_at"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


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
    from datetime import datetime, timezone
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
    with _db() as conn:
        conn.execute(
            "INSERT INTO search_conditions (id, name, payload, enabled, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cid, body.name, json.dumps(payload), int(body.enabled),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    _sync_to_analyzer()
    logger.info("외형 조건 등록: %s (%s)", cid, body.name)

    entry = {"id": cid, "name": body.name, "enabled": body.enabled, **payload}
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
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM search_conditions WHERE id = ?", (condition_id,)
        )
        conn.commit()
        deleted = cur.rowcount

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"조건을 찾을 수 없습니다: {condition_id}",
        )

    _sync_to_analyzer()
    logger.info("외형 조건 삭제: %s", condition_id)
    return success_response({"deleted": condition_id})
