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

    id: str = Field(description="카메라 고유 ID")
    name: Optional[str] = Field(default=None, description="카메라 표시 이름")
    url: Optional[str] = Field(default=None, description="자격증명을 제거한 카메라 URL")
    zones: Optional[list] = Field(default=None, description="cameras.json 기준 구역 설정 목록")


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

    site_id: str = Field(description="사이트 고유 ID")
    site_name: str = Field(description="사이트 이름")
    site_nickname: str = Field(default="", description="현장 표시용 별칭")
    camera_ids: List[str] = Field(default_factory=list, description="이 사이트에 연결된 카메라 ID 목록")
    control_mode: ControlModeOut = Field(description="알람 장치 제어 모드")
    alarm_devices: List[AlarmDeviceOut] = Field(default_factory=list, description="이 사이트에서 사용하는 알람 장치 목록")


class SiteCreateIn(BaseModel):
    """POST /api/v1/sites 요청 바디."""

    site_id: str = Field(min_length=1, max_length=64, description="사이트 고유 ID")
    site_name: str = Field(min_length=1, max_length=128, description="사이트 이름")
    site_nickname: str = Field(default="", description="현장 표시용 별칭")
    camera_ids: List[str] = Field(default_factory=list, description="이 사이트에 연결할 카메라 ID 목록")
    control_mode: ControlModeOut = Field(default=ControlModeOut.AUTO, description="기본 제어 모드")
    alarm_devices: List[AlarmDeviceOut] = Field(
        default_factory=lambda: [
            AlarmDeviceOut.SPEAKER,
            AlarmDeviceOut.SIGNBOARD,
        ],
        description="기본으로 연결할 알람 장치 목록",
    )


# ---------------------------------------------------------------------------
# 제어 모드
# ---------------------------------------------------------------------------


class ModeSetIn(BaseModel):
    """POST /api/v1/control/mode 요청 바디."""

    mode: ControlModeOut = Field(description="변경할 제어 모드")
    site_id: Optional[str] = Field(
        default=None,
        description="지정 시 해당 사이트만 변경, 생략 시 전체 적용",
    )


class ModeOut(BaseModel):
    """현재 모드 응답."""

    mode: str = Field(description="현재 적용된 제어 모드")
    site_id: Optional[str] = Field(default=None, description="사이트 단위 변경인 경우 대상 사이트 ID")


# ---------------------------------------------------------------------------
# 승인/거부
# ---------------------------------------------------------------------------


class ApprovalOut(BaseModel):
    """이벤트 승인/거부 결과."""

    event_id: str = Field(description="승인/거부한 이벤트 ID")
    status: str = Field(description="처리 결과 상태. approved 또는 rejected")
    message: str = Field(description="사용자에게 보여줄 처리 결과 메시지")


class PendingEventOut(BaseModel):
    """수동 승인 대기 이벤트 응답 모델."""

    event_id: str = Field(description="수동 승인 대기 큐에 등록된 이벤트 ID")
    queued_at: Optional[str] = Field(
        default=None,
        description="대기 큐에 적재된 시각(ISO 8601 문자열)",
    )
    site_id: Optional[str] = Field(
        default=None,
        description="사이트 단위 수동 승인인 경우 대상 사이트 ID",
    )
    camera_id: Optional[str] = Field(
        default=None,
        description="이 이벤트가 발생한 카메라 ID",
    )
    event_type: Optional[str] = Field(
        default=None,
        description="정규화된 이벤트 타입",
    )
    severity: Optional[str] = Field(
        default=None,
        description="이벤트 심각도",
    )
    topic: Optional[str] = Field(
        default=None,
        description="원본 MQTT 또는 내부 토픽",
    )
