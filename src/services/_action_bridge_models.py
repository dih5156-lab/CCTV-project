"""ActionBridge 도메인 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ControlMode(str, Enum):
    """카메라 사이트별 조치 제어 방식."""

    AUTO = "auto"
    MANUAL = "manual"


class AlarmDevice(str, Enum):
    """사이트에 연결된 알람 장치 종류."""

    SPEAKER = "speaker"
    SIREN = "siren"
    SIGNBOARD = "signboard"


@dataclass
class SiteConfig:
    """IoT 플랫폼 사이트(현장) 설정."""

    site_id: str
    site_name: str
    site_nickname: str = ""
    camera_ids: List[str] = field(default_factory=list)
    control_mode: ControlMode = ControlMode.AUTO
    alarm_devices: List[AlarmDevice] = field(
        default_factory=lambda: [AlarmDevice.SPEAKER, AlarmDevice.SIGNBOARD]
    )
    confidence_threshold: Optional[float] = None
    display_message: str = ""
    tts_message: str = ""

    def to_dict(self) -> Dict:
        return {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "site_nickname": self.site_nickname,
            "camera_ids": self.camera_ids,
            "control_mode": self.control_mode.value,
            "alarm_devices": [device.value for device in self.alarm_devices],
            "confidence_threshold": self.confidence_threshold,
            "display_message": self.display_message,
            "tts_message": self.tts_message,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SiteConfig":
        threshold = data.get("confidence_threshold")
        if threshold in ("", None):
            threshold = None
        else:
            threshold = max(0.0, min(float(threshold), 1.0))
        return cls(
            site_id=data["site_id"],
            site_name=data.get("site_name", ""),
            site_nickname=data.get("site_nickname", ""),
            camera_ids=data.get("camera_ids", []),
            control_mode=ControlMode(data.get("control_mode", "auto")),
            alarm_devices=[
                AlarmDevice(device)
                for device in data.get("alarm_devices", ["speaker", "signboard"])
            ],
            confidence_threshold=threshold,
            display_message=str(data.get("display_message", "") or ""),
            tts_message=str(data.get("tts_message", "") or ""),
        )
