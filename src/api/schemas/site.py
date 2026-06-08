"""카메라/사이트/제어 관련 Pydantic 스키마."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

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
    confidence_threshold: Optional[float] = Field(default=None, ge=0, le=1, description="이 값 미만 신뢰도 이벤트는 조치하지 않음")
    display_message: str = Field(default="", description="전광판 출력 문구")
    tts_message: str = Field(default="", description="스피커/TTS 출력 문구")


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
    confidence_threshold: Optional[float] = Field(default=None, ge=0, le=1, description="이 값 미만 신뢰도 이벤트는 조치하지 않음")
    display_message: str = Field(default="", max_length=300, description="전광판 출력 문구")
    tts_message: str = Field(default="", max_length=300, description="스피커/TTS 출력 문구")

    @model_validator(mode="after")
    def normalize_output_messages(self) -> "SiteCreateIn":
        if self.display_message and not self.tts_message:
            self.tts_message = self.display_message
        if self.tts_message and not self.display_message:
            self.display_message = self.tts_message
        return self


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
    alarm_devices: Optional[List[AlarmDeviceOut]] = Field(
        default=None,
        description="기본 모드에서 사용할 출력 장치 목록",
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="기본 모드에서 이 값 미만 신뢰도 이벤트는 조치하지 않음",
    )
    display_message: str = Field(default="", max_length=300, description="기본 전광판 출력 문구")
    tts_message: str = Field(default="", max_length=300, description="기본 스피커/TTS 출력 문구")

    @model_validator(mode="after")
    def normalize_output_messages(self) -> "ModeSetIn":
        if self.display_message and not self.tts_message:
            self.tts_message = self.display_message
        if self.tts_message and not self.display_message:
            self.display_message = self.tts_message
        return self


class ModeOut(BaseModel):
    """현재 모드 응답."""

    mode: str = Field(description="현재 적용된 제어 모드")
    site_id: Optional[str] = Field(default=None, description="사이트 단위 변경인 경우 대상 사이트 ID")
    alarm_devices: List[AlarmDeviceOut] = Field(default_factory=list, description="기본 모드 출력 장치 목록")
    confidence_threshold: Optional[float] = Field(default=None, description="기본 모드 신뢰도 임계값")
    display_message: str = Field(default="", description="기본 전광판 출력 문구")
    tts_message: str = Field(default="", description="기본 스피커/TTS 출력 문구")


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
    confidence: Optional[float] = Field(
        default=None,
        description="탐지 신뢰도. 원본 이벤트에 포함된 경우에만 반환",
    )
    severity: Optional[str] = Field(
        default=None,
        description="이벤트 심각도",
    )
    display_message: Optional[str] = Field(
        default=None,
        description="전광판 등 표시 장치에 출력할 문구",
    )
    tts_message: Optional[str] = Field(
        default=None,
        description="스피커/TTS 장치에 출력할 문구",
    )
    topic: Optional[str] = Field(
        default=None,
        description="원본 MQTT 또는 내부 토픽",
    )
