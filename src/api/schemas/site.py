"""카메라/사이트/제어 관련 Pydantic 스키마."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 카메라
# ---------------------------------------------------------------------------


class CameraOut(BaseModel):
    """카메라 응답 모델."""

    id: str
    name: Optional[str] = None
    url: Optional[str] = None
    zones: Optional[list] = None


# ---------------------------------------------------------------------------
# 사이트
# ---------------------------------------------------------------------------


class ControlModeOut(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


class AlarmDeviceOut(str, Enum):
    SPEAKER = "speaker"
    SIREN = "siren"
    SIGNBOARD = "signboard"


class SiteOut(BaseModel):
    """사이트 응답 모델."""

    site_id: str
    site_name: str
    site_nickname: str = ""
    camera_ids: List[str] = []
    control_mode: ControlModeOut
    alarm_devices: List[AlarmDeviceOut] = []


class SiteCreateIn(BaseModel):
    """POST /api/v1/sites 요청 바디."""

    site_id: str = Field(min_length=1, max_length=64)
    site_name: str = Field(min_length=1, max_length=128)
    site_nickname: str = ""
    camera_ids: List[str] = []
    control_mode: ControlModeOut = ControlModeOut.AUTO
    alarm_devices: List[AlarmDeviceOut] = [AlarmDeviceOut.SPEAKER]


# ---------------------------------------------------------------------------
# 제어 모드
# ---------------------------------------------------------------------------


class ModeSetIn(BaseModel):
    """POST /api/v1/control/mode 요청 바디."""

    mode: ControlModeOut
    site_id: Optional[str] = Field(
        default=None,
        description="지정 시 해당 사이트만 변경, 생략 시 전체 적용",
    )


class ModeOut(BaseModel):
    """현재 모드 응답."""

    mode: str
    site_id: Optional[str] = None


# ---------------------------------------------------------------------------
# 승인/거부
# ---------------------------------------------------------------------------


class ApprovalOut(BaseModel):
    """이벤트 승인/거부 결과."""

    event_id: str
    status: str  # "approved" | "rejected"
    message: str
